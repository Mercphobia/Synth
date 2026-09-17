"""Tests for tools.py: registry, safe_path, read_file, write_file, bash."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synth.tools import (
    ToolRegistry,
    ToolSpec,
    default_registry,
    make_bash_tool,
    make_read_file_tool,
    make_write_file_tool,
    safe_path,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every tool test inside a temp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- safe_path ---

class TestSafePath:
    def test_relative_path_resolves_inside_base(self, workspace: Path) -> None:
        result = safe_path("hello.txt", workspace)
        assert result == (workspace / "hello.txt").resolve()

    def test_absolute_path_inside_base_allowed(self, workspace: Path) -> None:
        absolute = str(workspace / "nested" / "file.txt")
        result = safe_path(absolute, workspace)
        assert result == Path(absolute).resolve()

    def test_empty_path_rejected(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            safe_path("", workspace)

    def test_none_path_rejected(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            safe_path(None, workspace)  # type: ignore[arg-type]

    def test_parent_traversal_rejected(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="traversal"):
            safe_path("../../etc/passwd", workspace)

    def test_absolute_path_outside_base_rejected(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="outside"):
            safe_path("/etc/passwd", workspace)


# --- read_file ---

class TestReadFile:
    def test_reads_existing_file(self, workspace: Path) -> None:
        target = workspace / "note.txt"
        target.write_text("hello world", encoding="utf-8")
        tool = make_read_file_tool()
        result = tool({"path": "note.txt"})
        assert result.is_error is False
        assert result.text == "hello world"

    def test_missing_file_returns_error(self, workspace: Path) -> None:
        tool = make_read_file_tool()
        result = tool({"path": "ghost.txt"})
        assert result.is_error is True
        assert "not found" in result.text.lower()

    def test_directory_returns_error(self, workspace: Path) -> None:
        (workspace / "adir").mkdir()
        tool = make_read_file_tool()
        result = tool({"path": "adir"})
        assert result.is_error is True
        assert "directory" in result.text.lower()

    def test_oversized_file_rejected(self, workspace: Path) -> None:
        target = workspace / "big.txt"
        target.write_text("a" * 100, encoding="utf-8")
        tool = make_read_file_tool(max_file_size=10)
        result = tool({"path": "big.txt"})
        assert result.is_error is True
        assert "limit" in result.text.lower()

    def test_truncates_long_output(self, workspace: Path) -> None:
        target = workspace / "long.txt"
        target.write_text("b" * 500, encoding="utf-8")
        tool = make_read_file_tool(max_output_size=20)
        result = tool({"path": "long.txt"})
        assert len(result.text) <= 20 + len("\n...[truncated at 20 bytes]")
        assert "truncated" in result.text

    def test_path_outside_workspace_rejected(self, workspace: Path) -> None:
        tool = make_read_file_tool()
        result = tool({"path": "../../etc/hostname"})
        assert result.is_error is True


# --- write_file ---

class TestWriteFile:
    def test_writes_new_file(self, workspace: Path) -> None:
        tool = make_write_file_tool()
        result = tool({"path": "out.txt", "content": "hello"})
        assert result.is_error is False
        assert (workspace / "out.txt").read_text() == "hello"

    def test_overwrites_existing_file(self, workspace: Path) -> None:
        target = workspace / "out.txt"
        target.write_text("old", encoding="utf-8")
        tool = make_write_file_tool()
        tool({"path": "out.txt", "content": "new"})
        assert target.read_text() == "new"

    def test_creates_parent_directories(self, workspace: Path) -> None:
        tool = make_write_file_tool()
        tool({"path": "a/b/c.txt", "content": "deep"})
        assert (workspace / "a" / "b" / "c.txt").read_text() == "deep"

    def test_oversized_content_rejected(self, workspace: Path) -> None:
        tool = make_write_file_tool(max_file_size=5)
        result = tool({"path": "big.txt", "content": "xxxxxxxxxx"})
        assert result.is_error is True
        assert "exceeds" in result.text.lower()

    def test_non_string_content_rejected(self, workspace: Path) -> None:
        tool = make_write_file_tool()
        result = tool({"path": "x.txt", "content": 42})  # type: ignore[dict-item]
        assert result.is_error is True

    def test_success_message_reports_byte_count(self, workspace: Path) -> None:
        tool = make_write_file_tool()
        result = tool({"path": "out.txt", "content": "abcd"})
        assert "4 bytes" in result.text


# --- bash ---

class TestBash:
    def test_runs_command_and_returns_output(self, workspace: Path) -> None:
        tool = make_bash_tool()
        result = tool({"command": "echo synth"})
        assert result.is_error is False
        assert "synth" in result.text
        assert "exit_code=0" in result.text

    def test_captures_stderr_and_exit_code(self, workspace: Path) -> None:
        tool = make_bash_tool()
        result = tool({"command": "no-such-command-xyz"})
        assert result.is_error is False  # command ran; it just failed
        assert "exit_code=" in result.text
        assert result.text.split("exit_code=", 1)[1].splitlines()[0] != "0"

    def test_empty_command_rejected(self, workspace: Path) -> None:
        tool = make_bash_tool()
        result = tool({"command": ""})
        assert result.is_error is True
        assert "non-empty" in result.text.lower()

    def test_timeout_returns_error(self, workspace: Path) -> None:
        tool = make_bash_tool(timeout=1)
        result = tool({"command": "sleep 5"})
        assert result.is_error is True
        assert "timed out" in result.text.lower()

    def test_truncates_huge_output(self, workspace: Path) -> None:
        tool = make_bash_tool(max_output_size=15)
        result = tool({"command": "printf 'y%.0s' {1..500}"})
        assert len(result.text) <= 15 + 40  # cap + truncation marker


# --- registry ---

class TestRegistry:
    def test_default_registry_has_three_tools(self) -> None:
        registry = default_registry()
        assert registry.names() == ["bash", "read_file", "write_file"]

    def test_schemas_match_openai_shape(self) -> None:
        registry = default_registry()
        for schema in registry.schemas():
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    def test_duplicate_registration_rejected(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(
            name="x",
            description="d",
            parameters={"type": "object"},
            func=lambda args: None,  # type: ignore[arg-type]
        )
        registry.register(spec)
        with pytest.raises(ValueError):
            registry.register(spec)

    def test_execute_unknown_tool_returns_error(self) -> None:
        registry = default_registry()
        result = registry.execute("nope", {})
        assert result.is_error is True
        assert "unknown tool" in result.text.lower()

    def test_execute_missing_required_arg_returns_error(self) -> None:
        registry = default_registry()
        result = registry.execute("read_file", {})
        assert result.is_error is True
        assert "missing" in result.text.lower()

    def test_execute_non_dict_args_returns_error(self) -> None:
        registry = default_registry()
        result = registry.execute("bash", "not a dict")  # type: ignore[arg-type]
        assert result.is_error is True

    def test_crashing_tool_returns_error_not_exception(self) -> None:
        def boom(args: dict) -> None:
            raise RuntimeError("kaboom")

        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="boom", description="d", parameters={"type": "object"}, func=boom)
        )
        result = registry.execute("boom", {})
        assert result.is_error is True
        assert "kaboom" in result.text
