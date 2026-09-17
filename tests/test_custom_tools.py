"""Tests for synth.custom_tools: tool loader, permission prompts, autocomplete."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from synth.custom_tools import (
    Autocomplete,
    BaseTool,
    DANGEROUS_DEFAULTS,
    MAX_CUSTOM_TOOLS,
    PermissionPrompt,
    SLASH_COMMANDS,
    ToolLoader,
)
from synth.tools import ToolResult, ToolSpec


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def tools_dir(tmp_path: Path) -> Path:
    """Create a temporary tools directory."""
    d = tmp_path / "tools"
    d.mkdir()
    return d


VALID_TOOL_SOURCE = '''
from synth.custom_tools import BaseTool

class HelloTool(BaseTool):
    name = "hello"
    description = "Say hello"
    dangerous = False
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def execute(self, **kwargs):
        return "hello " + kwargs.get("name", "world")
'''

BROKEN_TOOL_SOURCE = '''
class BrokenTool(BaseTool):  # NameError: BaseTool not imported
    name = "broken"
    def execute(self, **kwargs):
        return "x"
'''

DANGEROUS_IMPORT_SOURCE = '''
import os
os.system("echo pwned")

class EvilTool(BaseTool):
    name = "evil"
    description = "evil"
    schema = {"type": "object", "properties": {}}
    def execute(self, **kwargs):
        return "ok"
'''

SUBPROCESS_DANGEROUS_SOURCE = '''
import subprocess
subprocess.run(["ls"], capture_output=True)

class SubTool(BaseTool):
    name = "sub"
    description = "sub"
    schema = {"type": "object", "properties": {}}
    def execute(self, **kwargs):
        return "ok"
'''

NO_BASETOOL_SOURCE = '''
class NotATool:
    name = "nothing"
    def execute(self, **kwargs):
        return "x"
'''

BAD_NAME_SOURCE = '''
from synth.custom_tools import BaseTool

class BadNameTool(BaseTool):
    name = "123 Bad Name"
    description = "bad"
    schema = {"type": "object", "properties": {}}
    def execute(self, **kwargs):
        return "ok"
'''


# --- ToolLoader tests --------------------------------------------------------


def test_loader_finds_valid_tool(tools_dir: Path) -> None:
    """A valid .py file with a BaseTool subclass produces a ToolSpec."""
    (tools_dir / "hello.py").write_text(VALID_TOOL_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert len(specs) == 1
    assert specs[0].name == "hello"
    assert specs[0].description == "Say hello"


def test_loader_valid_tool_executes(tools_dir: Path) -> None:
    """The loaded tool's func actually runs and returns the right text."""
    (tools_dir / "hello.py").write_text(VALID_TOOL_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert len(specs) == 1
    result = specs[0].func({"name": "quasar"})
    assert isinstance(result, ToolResult)
    assert result.text == "hello quasar"
    assert not result.is_error


def test_loader_skips_broken_py(tools_dir: Path) -> None:
    """A .py file that fails to import is skipped, not fatal."""
    (tools_dir / "broken.py").write_text(BROKEN_TOOL_SOURCE)
    (tools_dir / "hello.py").write_text(VALID_TOOL_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    # broken.py fails to import (NameError on BaseTool), hello.py loads fine
    assert len(specs) == 1
    assert specs[0].name == "hello"
    # The log records the failure
    assert any("broken.py" in entry for entry in loader.log)


def test_loader_rejects_dangerous_os_system(tools_dir: Path) -> None:
    """A file with a top-level os.system call is rejected."""
    (tools_dir / "evil.py").write_text(DANGEROUS_IMPORT_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert specs == []
    assert any("rejected" in entry for entry in loader.log)


def test_loader_rejects_dangerous_subprocess(tools_dir: Path) -> None:
    """A file with a top-level subprocess.run call is rejected."""
    (tools_dir / "sub.py").write_text(SUBPROCESS_DANGEROUS_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert specs == []
    assert any("rejected" in entry for entry in loader.log)


def test_loader_skips_file_with_no_basetool(tools_dir: Path) -> None:
    """A .py file with no BaseTool subclass yields no specs and logs a note."""
    (tools_dir / "notool.py").write_text(NO_BASETOOL_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert specs == []
    assert any("no BaseTool" in entry for entry in loader.log)


def test_loader_rejects_bad_tool_name(tools_dir: Path) -> None:
    """A tool whose name isn't a valid identifier is rejected."""
    (tools_dir / "badname.py").write_text(BAD_NAME_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert specs == []
    assert any("not a valid identifier" in entry for entry in loader.log)


def test_loader_empty_dir_returns_empty(tools_dir: Path) -> None:
    """An empty tools directory produces no specs and no log entries."""
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert specs == []
    assert loader.log == []


def test_loader_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A non-existent tools directory yields an empty list, no crash."""
    loader = ToolLoader(tools_dir=tmp_path / "does_not_exist")
    specs = loader.load()
    assert specs == []


def test_loader_toolspec_schema_shape(tools_dir: Path) -> None:
    """The produced ToolSpec.schema() dict has the OpenAI function shape."""
    (tools_dir / "hello.py").write_text(VALID_TOOL_SOURCE)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert len(specs) == 1
    schema = specs[0].schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "hello"
    assert "properties" in schema["function"]["parameters"]


def test_loader_tool_crash_returns_error_result(tools_dir: Path) -> None:
    """If execute() raises, the adapter wraps it in an error ToolResult."""
    source = '''
from synth.custom_tools import BaseTool

class CrashTool(BaseTool):
    name = "crash"
    description = "crashes"
    schema = {"type": "object", "properties": {}}
    def execute(self, **kwargs):
        raise RuntimeError("boom")
'''
    (tools_dir / "crash.py").write_text(source)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert len(specs) == 1
    result = specs[0].func({})
    assert result.is_error
    assert "crashed" in result.text


def test_loader_tool_returning_toolresult_passes_through(tools_dir: Path) -> None:
    """A tool that returns a ToolResult directly gets it passed through."""
    source = '''
from synth.custom_tools import BaseTool
from synth.tools import ToolResult

class PassthroughTool(BaseTool):
    name = "pass"
    description = "pass"
    schema = {"type": "object", "properties": {}}
    def execute(self, **kwargs):
        return ToolResult("custom", is_error=True)
'''
    (tools_dir / "pass.py").write_text(source)
    loader = ToolLoader(tools_dir=tools_dir)
    specs = loader.load()
    assert len(specs) == 1
    result = specs[0].func({})
    assert result.text == "custom"
    assert result.is_error


# --- PermissionPrompt tests --------------------------------------------------


def test_permission_strict_prompts_all() -> None:
    """Strict mode: even non-dangerous tools prompt (returns False)."""
    pp = PermissionPrompt(mode="strict")
    assert pp.check("read_file", {}) is False


def test_permission_balanced_prompts_dangerous_only() -> None:
    """Balanced mode: dangerous tools prompt, safe ones auto-approve."""
    pp = PermissionPrompt(mode="balanced")
    # dangerous tool -> prompt
    assert pp.check("bash", {"command": "ls"}) is False
    assert pp.check("write_file", {}) is False
    assert pp.check("edit_file", {}) is False
    # safe tool -> auto-approve
    assert pp.check("read_file", {}) is True


def test_permission_auto_prompts_none() -> None:
    """Auto mode: nothing prompts, even dangerous tools."""
    pp = PermissionPrompt(mode="auto")
    assert pp.check("bash", {"command": "rm -rf /"}) is True
    assert pp.check("read_file", {}) is True


def test_permission_yolo_prompts_none() -> None:
    """Yolo mode: alias of auto — everything auto-approves."""
    pp = PermissionPrompt(mode="yolo")
    assert pp.check("bash", {}) is True
    assert pp.check("write_file", {}) is True


def test_permission_balanced_custom_dangerous_list() -> None:
    """A caller-supplied dangerous list overrides the default for that call."""
    pp = PermissionPrompt(mode="balanced")
    # 'grep' is not dangerous by default, but caller marks it so for this call
    assert pp.check("grep", {}, dangerous_tools=["grep"]) is False
    # 'bash' is dangerous by default, but caller clears the list for this call
    assert pp.check("bash", {}, dangerous_tools=[]) is True


def test_permission_invalid_mode_raises() -> None:
    """An unknown mode string raises ValueError at construction."""
    with pytest.raises(ValueError):
        PermissionPrompt(mode="paranoid")


def test_permission_defaults_match_constants() -> None:
    """The default dangerous_tools equals DANGEROUS_DEFAULTS."""
    pp = PermissionPrompt(mode="balanced")
    assert pp.dangerous_tools == DANGEROUS_DEFAULTS


# --- Autocomplete tests ------------------------------------------------------


def test_autocomplete_empty_input_returns_empty(tmp_path: Path) -> None:
    """Empty or None input yields no suggestions."""
    ac = Autocomplete(base_dir=tmp_path)
    assert ac.complete("") == []
    assert ac.complete("   ") == []  # non-trigger first char


def test_autocomplete_at_file_glob(tmp_path: Path) -> None:
    """@prefix globs for files under base_dir."""
    (tmp_path / "alpha.txt").write_text("x")
    (tmp_path / "beta.txt").write_text("x")
    (tmp_path / "gamma.log").write_text("x")
    ac = Autocomplete(base_dir=tmp_path)
    # @a should match alpha.txt
    results = ac.complete("@a")
    assert len(results) == 1
    assert results[0] == ("@", "alpha.txt")
    # @ (empty remainder) returns all files
    results = ac.complete("@")
    labels = [label for _, label in results]
    assert "alpha.txt" in labels
    assert "beta.txt" in labels
    assert "gamma.log" in labels


def test_autocomplete_slash_command_match(tmp_path: Path) -> None:
    """/he matches /help."""
    ac = Autocomplete(base_dir=tmp_path)
    results = ac.complete("/he")
    assert results == [("/", "/help")]
    # /m matches /mode and /model
    results = ac.complete("/m")
    labels = [label for _, label in results]
    assert "/mode" in labels
    assert "/model" in labels


def test_autocomplete_slash_empty_returns_all(tmp_path: Path) -> None:
    """/ with no remainder returns every known slash command."""
    ac = Autocomplete(base_dir=tmp_path)
    results = ac.complete("/")
    labels = [label for _, label in results]
    assert set(labels) == set(SLASH_COMMANDS)


def test_autocomplete_shell_passthrough(tmp_path: Path) -> None:
    """! prefix returns a passthrough suggestion."""
    ac = Autocomplete(base_dir=tmp_path)
    results = ac.complete("!ls")
    assert len(results) == 1
    assert results[0][0] == "!"
    assert "ls" in results[0][1]
    # bare ! returns the shell marker
    results = ac.complete("!")
    assert results == [("!", "shell")]


def test_autocomplete_skill_match(tmp_path: Path) -> None:
    """~ prefix matches against loaded skill names."""
    ac = Autocomplete(skills=["deploy", "debug", "docker"], base_dir=tmp_path)
    results = ac.complete("~de")
    labels = [label for _, label in results]
    assert "deploy" in labels
    assert "debug" in labels
    # ~d also matches 'docker'
    results = ac.complete("~d")
    labels = [label for _, label in results]
    assert "docker" in labels
    assert "deploy" in labels
    assert "debug" in labels


def test_autocomplete_skill_no_skills(tmp_path: Path) -> None:
    """~ prefix with no skills loaded returns nothing."""
    ac = Autocomplete(skills=[], base_dir=tmp_path)
    assert ac.complete("~anything") == []


def test_autocomplete_non_trigger_returns_empty(tmp_path: Path) -> None:
    """Input that doesn't start with a known trigger yields no suggestions."""
    ac = Autocomplete(base_dir=tmp_path)
    assert ac.complete("hello") == []
    assert ac.complete("xyz") == []


def test_autocomplete_file_nonexistent_pattern(tmp_path: Path) -> None:
    """A glob prefix with no matches returns an empty list."""
    ac = Autocomplete(base_dir=tmp_path)
    assert ac.complete("@zzz") == []
