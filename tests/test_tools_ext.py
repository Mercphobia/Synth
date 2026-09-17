"""Tests for tools_ext.py: edit_file, glob, grep, extended_registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.tools import ToolRegistry, safe_path
from synth.tools_ext import (
    MAX_GLOB_RESULTS,
    MAX_GREP_RESULTS,
    extended_registry,
    make_edit_file_tool,
    make_glob_tool,
    make_grep_tool,
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every tool test inside a temp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- edit_file ---


class TestEditFile:
    def test_edit_success_replaces_unique_string(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("def foo():\n    return 1\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "code.py", "old_string": "return 1", "new_string": "return 2"})
        assert result.is_error is False
        assert "OK: edited" in result.text
        assert target.read_text() == "def foo():\n    return 2\n"

    def test_edit_reports_line_count_change(self, workspace: Path) -> None:
        target = workspace / "note.txt"
        target.write_text("a\nb\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "note.txt", "old_string": "b", "new_string": "b\nc"})
        assert result.is_error is False
        assert "lines" in result.text

    def test_edit_old_string_not_found_returns_error(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("print('hi')\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "code.py", "old_string": "nope", "new_string": "yes"})
        assert result.is_error is True
        assert "not found" in result.text.lower()
        assert target.read_text() == "print('hi')\n"

    def test_edit_non_unique_string_returns_error(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("x = 1\nx = 1\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "code.py", "old_string": "x = 1", "new_string": "x = 2"})
        assert result.is_error is True
        assert "not unique" in result.text.lower()
        # File must be untouched when the edit is ambiguous.
        assert target.read_text() == "x = 1\nx = 1\n"

    def test_edit_missing_file_returns_error(self, workspace: Path) -> None:
        tool = make_edit_file_tool()
        result = tool({"path": "ghost.txt", "old_string": "a", "new_string": "b"})
        assert result.is_error is True
        assert "not found" in result.text.lower()

    def test_edit_empty_old_string_returns_error(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("hello\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "code.py", "old_string": "", "new_string": "x"})
        assert result.is_error is True
        assert "old_string" in result.text.lower()

    def test_edit_path_outside_workspace_rejected(self, workspace: Path) -> None:
        tool = make_edit_file_tool()
        result = tool({"path": "../../etc/hostname", "old_string": "a", "new_string": "b"})
        assert result.is_error is True
        assert "traversal" in result.text.lower()

    def test_edit_replaces_only_first_occurrence_when_unique_context_given(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("a\nb\nc\nb\n", encoding="utf-8")
        tool = make_edit_file_tool()
        result = tool({"path": "code.py", "old_string": "a\nb", "new_string": "a\nB"})
        assert result.is_error is False
        assert target.read_text() == "a\nB\nc\nb\n"


# --- glob ---


class TestGlob:
    def test_glob_recursive_match_returns_files(self, workspace: Path) -> None:
        deep = workspace / "src" / "nested"
        deep.mkdir(parents=True)
        (deep / "a.py").write_text("", encoding="utf-8")
        (workspace / "b.txt").write_text("", encoding="utf-8")
        tool = make_glob_tool()
        result = tool({"pattern": "**/*.py"})
        assert result.is_error is False
        assert "a.py" in result.text
        assert "b.txt" not in result.text

    def test_glob_no_match_returns_normal_empty_result(self, workspace: Path) -> None:
        tool = make_glob_tool()
        result = tool({"pattern": "*.none"})
        assert result.is_error is False
        assert "No files matched" in result.text

    def test_glob_bad_pattern_returns_error(self, workspace: Path) -> None:
        tool = make_glob_tool()
        # An absolute pattern would search outside the workspace.
        result = tool({"pattern": "/etc/*"})
        assert result.is_error is True
        assert "pattern" in result.text.lower()

    def test_glob_pathological_pattern_returns_error(self, workspace: Path) -> None:
        tool = make_glob_tool()
        # '.' is a pattern pathlib itself rejects; the tool must wrap it.
        result = tool({"pattern": "."})
        assert result.is_error is True
        assert "invalid glob pattern" in result.text.lower()

    def test_glob_empty_pattern_returns_error(self, workspace: Path) -> None:
        tool = make_glob_tool()
        result = tool({"pattern": ""})
        assert result.is_error is True
        assert "pattern" in result.text.lower()

    def test_glob_path_outside_workspace_rejected(self, workspace: Path) -> None:
        tool = make_glob_tool()
        result = tool({"pattern": "*", "path": "../../etc"})
        assert result.is_error is True
        assert "traversal" in result.text.lower()

    def test_glob_truncates_excessive_results(self, workspace: Path) -> None:
        for i in range(10):
            (workspace / f"f{i}.txt").write_text("", encoding="utf-8")
        tool = make_glob_tool(max_results=3, max_output_size=1000)
        result = tool({"pattern": "*.txt"})
        assert result.is_error is False
        assert "more results omitted" in result.text


# --- grep ---


class TestGrep:
    def test_grep_match_returns_path_line_text(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("import os\nprint('x')\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": r"print"})
        assert result.is_error is False
        assert ":2: print('x')" in result.text

    def test_grep_searches_nested_files_recursively(self, workspace: Path) -> None:
        deep = workspace / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "mod.py").write_text("needle = 1\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": "needle"})
        assert result.is_error is False
        assert "mod.py:1: needle = 1" in result.text

    def test_grep_no_match_returns_normal_empty_result(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("nothing here\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": "zzz"})
        assert result.is_error is False
        assert "No matches" in result.text

    def test_grep_invalid_regex_returns_error(self, workspace: Path) -> None:
        target = workspace / "code.py"
        target.write_text("hello\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": "([0-9"})
        assert result.is_error is True
        assert "invalid regex" in result.text.lower()

    def test_grep_binary_file_skipped(self, workspace: Path) -> None:
        binary = workspace / "blob.bin"
        binary.write_bytes(b"\x00\x01\x02needle\x00")
        text_file = workspace / "code.py"
        text_file.write_text("needle found\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": "needle"})
        assert result.is_error is False
        # Only the text file is reported; the binary is skipped silently.
        assert "code.py" in result.text
        assert "blob.bin" not in result.text

    def test_grep_truncates_at_max_results(self, workspace: Path) -> None:
        target = workspace / "many.txt"
        target.write_text("hit\n" * 50, encoding="utf-8")
        tool = make_grep_tool(max_results=5, max_output_size=1000)
        result = tool({"pattern": "hit"})
        assert result.is_error is False
        # Exactly the cap, not all 50 matches.
        assert len(result.text.strip().splitlines()) == 5

    def test_grep_include_filter_limits_files(self, workspace: Path) -> None:
        (workspace / "a.py").write_text("target_line\n", encoding="utf-8")
        (workspace / "b.txt").write_text("target_line\n", encoding="utf-8")
        tool = make_grep_tool()
        result = tool({"pattern": "target_line", "include": "*.py"})
        assert result.is_error is False
        assert "a.py" in result.text
        assert "b.txt" not in result.text

    def test_grep_path_outside_workspace_rejected(self, workspace: Path) -> None:
        tool = make_grep_tool()
        result = tool({"pattern": "x", "path": "../../etc"})
        assert result.is_error is True
        assert "traversal" in result.text.lower()

    def test_grep_missing_path_returns_error(self, workspace: Path) -> None:
        tool = make_grep_tool()
        result = tool({"pattern": "x", "path": "no-such-dir"})
        assert result.is_error is True
        assert "does not exist" in result.text.lower()


# --- extended registry ---


class TestExtendedRegistry:
    def test_extended_registry_has_six_tools(self) -> None:
        registry = extended_registry()
        assert registry.names() == [
            "bash",
            "edit_file",
            "glob",
            "grep",
            "read_file",
            "write_file",
        ]

    def test_extended_registry_is_a_tool_registry(self) -> None:
        registry = extended_registry()
        assert isinstance(registry, ToolRegistry)

    def test_extended_registry_schemas_match_openai_shape(self) -> None:
        registry = extended_registry()
        for schema in registry.schemas():
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    def test_extended_registry_defaults_match_constants(self) -> None:
        registry = extended_registry()
        assert registry.get("glob") is not None
        assert MAX_GLOB_RESULTS > 0
        assert MAX_GREP_RESULTS > 0

    def test_extended_registry_executes_new_tools(self, workspace: Path) -> None:
        (workspace / "findme.txt").write_text("hello\n", encoding="utf-8")
        registry = extended_registry()
        result = registry.execute("glob", {"pattern": "*.txt"})
        assert result.is_error is False
        assert "findme.txt" in result.text

    def test_extended_registry_still_runs_mvp_tools(self, workspace: Path) -> None:
        registry = extended_registry()
        result = registry.execute("write_file", {"path": "x.txt", "content": "hi"})
        assert result.is_error is False
        assert (workspace / "x.txt").read_text() == "hi"


# --- module safety helpers still usable ---


class TestSafetyHelpers:
    def test_safe_path_still_rejects_traversal(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="traversal"):
            safe_path("../../etc/passwd", workspace)
