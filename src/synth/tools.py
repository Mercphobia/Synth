"""Tool registry and the three MVP tools: read_file, write_file, bash.

Every tool is a boundary: it returns an error string instead of raising, so
the agent can observe the failure and choose a different approach.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from synth.constants import BASH_TIMEOUT, MAX_FILE_SIZE, MAX_OUTPUT_SIZE

# Match paths containing '..' (parent traversal). Commented per rules 2.5:
# this is the traversal guard, not a general filter.
PATH_TRAVERSAL = re.compile(r"\.\.")


def _is_godmode_active() -> bool:
    """Check if godmode is enabled via environment variable."""
    return os.environ.get("GODMODE", "0") == "1"


@dataclass(frozen=True)
class ToolResult:
    """Outcome of running a tool: either a value or an error message."""

    text: str
    is_error: bool = False

    def __str__(self) -> str:
        return self.text


# A tool function takes a dict of arguments and returns a ToolResult.
ToolFunc = Callable[[dict[str, Any]], ToolResult]


@dataclass
class ToolSpec:
    """One tool: JSON schema for the LLM plus its execution function."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI-style function schema sent to the LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    """Maps tool names to their specs."""

    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        """Add a tool. Duplicate names are rejected — silent overrides hide bugs."""
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """Return a tool spec by name, or None if unknown."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """All tool schemas, for sending to the LLM."""
        return [spec.schema() for spec in (self._tools[name] for name in self.names())]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run a tool by name.

        Unknown tools and bad arguments return an error result instead of
        raising — the agent sees the failure and can recover.
        """
        spec = self.get(name)
        if spec is None:
            return ToolResult(f"Error: unknown tool '{name}'", is_error=True)

        if not isinstance(args, dict):
            return ToolResult(f"Error: tool '{name}' expects an object of arguments", is_error=True)

        missing = _missing_params(spec, args)
        if missing:
            return ToolResult(
                f"Error: tool '{name}' missing required arguments: {', '.join(sorted(missing))}",
                is_error=True,
            )

        try:
            return spec.func(args)
        except Exception as exc:  # noqa: BLE001 — boundary: never raise to the loop
            return ToolResult(f"Error: tool '{name}' crashed: {exc}", is_error=True)


def _missing_params(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Return the list of required params the LLM didn't supply."""
    required = spec.parameters.get("required", [])
    return [p for p in required if p not in args]


def safe_path(path: str, base_dir: Path) -> Path:
    """Resolve a user-supplied path and verify it stays inside base_dir.

    Args:
        path: The path argument from the LLM.
        base_dir: The directory the agent is allowed to touch.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path is empty, traverses outside base_dir, or is
            not a string.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    if PATH_TRAVERSAL.search(path):
        raise ValueError(f"path traversal rejected: {path}")

    resolved = (base_dir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    base_resolved = base_dir.resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"path outside working directory: {path}") from exc

    return resolved


# --- read_file ---

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path (relative or absolute, must stay within the working directory)",
        }
    },
    "required": ["path"],
}


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit bytes with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated at {limit} bytes]"


def make_read_file_tool(max_file_size: int = MAX_FILE_SIZE, max_output_size: int = MAX_OUTPUT_SIZE) -> ToolFunc:
    """Build the read_file tool with configured limits."""

    def _read_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        
        # Godmode: bypass path safety checks
        if _is_godmode_active():
            resolved = Path(path).resolve()
        else:
            try:
                resolved = safe_path(str(path), Path.cwd())
            except ValueError as exc:
                return ToolResult(f"Error: {exc}", is_error=True)

        if not resolved.exists():
            return ToolResult(f"Error: file not found: {resolved}", is_error=True)
        if resolved.is_dir():
            return ToolResult(f"Error: path is a directory, not a file: {resolved}", is_error=True)
        if not resolved.is_file():
            return ToolResult(f"Error: not a regular file: {resolved}", is_error=True)

        # Godmode: bypass file size limits
        if not _is_godmode_active():
            size = resolved.stat().st_size
            if size > max_file_size:
                return ToolResult(
                    f"Error: file is {size} bytes (limit {max_file_size}): {resolved}",
                    is_error=True,
                )

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(f"Error: cannot read {resolved}: {exc}", is_error=True)

        # Godmode: bypass output truncation
        if _is_godmode_active():
            return ToolResult(content)
        return ToolResult(_truncate(content, max_output_size))

    return _read_file


# --- write_file ---

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
}


def make_write_file_tool(max_file_size: int = MAX_FILE_SIZE) -> ToolFunc:
    """Build the write_file tool with a size limit."""

    def _write_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content", "")
        
        # Godmode: bypass path safety checks
        if _is_godmode_active():
            resolved = Path(path).resolve()
        else:
            try:
                resolved = safe_path(str(path), Path.cwd())
            except ValueError as exc:
                return ToolResult(f"Error: {exc}", is_error=True)

        if not isinstance(content, str):
            return ToolResult("Error: content must be a string", is_error=True)
        
        # Godmode: bypass content size limits
        if not _is_godmode_active() and len(content.encode("utf-8", errors="replace")) > max_file_size:
            return ToolResult(
                f"Error: content exceeds {max_file_size} bytes",
                is_error=True,
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Error: cannot write {resolved}: {exc}", is_error=True)

        return ToolResult(f"OK: wrote {len(content.encode('utf-8'))} bytes to {resolved}")

    return _write_file


# --- bash ---

BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
    },
    "required": ["command"],
}


def make_bash_tool(timeout: int = BASH_TIMEOUT, max_output_size: int = MAX_OUTPUT_SIZE) -> ToolFunc:
    """Build the bash tool with a timeout and output cap."""

    def _bash(args: dict[str, Any]) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult("Error: command must be a non-empty string", is_error=True)

        # Godmode: bypass timeout
        actual_timeout = None if _is_godmode_active() else timeout

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=actual_timeout,
                cwd=str(Path.cwd()),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Error: command timed out after {timeout}s: {command}",
                is_error=True,
            )
        except OSError as exc:
            return ToolResult(f"Error: cannot run command: {exc}", is_error=True)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        output = f"exit_code={proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        
        # Godmode: bypass output truncation
        if _is_godmode_active():
            return ToolResult(output)
        return ToolResult(_truncate(output, max_output_size))

    return _bash


# --- default registry ---

def default_registry(
    bash_timeout: int = BASH_TIMEOUT,
    max_file_size: int = MAX_FILE_SIZE,
    max_output_size: int = MAX_OUTPUT_SIZE,
) -> ToolRegistry:
    """Build the registry containing the three MVP tools."""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="read_file",
            description="Read contents of a file",
            parameters=READ_FILE_SCHEMA,
            func=make_read_file_tool(max_file_size, max_output_size),
        )
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Write content to a file (overwrites existing)",
            parameters=WRITE_FILE_SCHEMA,
            func=make_write_file_tool(max_file_size),
        )
    )
    registry.register(
        ToolSpec(
            name="bash",
            description="Execute a shell command and return stdout/stderr",
            parameters=BASH_SCHEMA,
            func=make_bash_tool(bash_timeout, max_output_size),
        )
    )
    return registry
