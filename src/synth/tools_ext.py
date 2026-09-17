"""Extended tools: edit_file, glob, grep, plus the extended registry.

Sits alongside tools.py without modifying it: the three MVP tools stay in
tools.py and this module adds the next three, reusing its primitives
(ToolResult/ToolSpec/ToolRegistry, safe_path, _truncate, default_registry).

Every tool here is a boundary like the others: it returns an error string
instead of raising, so the agent can observe the failure and adapt. The
regex and glob patterns come straight from the LLM and are untrusted, so
they are compiled defensively and their output is always truncated.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from synth.constants import (
    BASH_TIMEOUT,
    MAX_FILE_SIZE,
    MAX_OUTPUT_SIZE,
)
from synth.tools import (
    ToolFunc,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    _truncate,
    default_registry,
    safe_path,
)

# --- Limits (module-local: specific to the search/edit tools) ---

# Cap on glob results. A '**/*' over a large tree can yield thousands of
# entries; this keeps one listing from eating the whole context window.
MAX_GLOB_RESULTS = 500

# Cap on grep matches. Same reasoning: a common pattern in a big codebase
# matches everywhere, and the agent only needs the first few hits to act.
MAX_GREP_RESULTS = 200

# Bytes of a file to sniff when deciding it is binary. A NUL byte in the
# first 2 KB is a reliable binary signal, and reading a prefix avoids
# loading whole logs into memory just to skip them.
BINARY_SNIFF_BYTES = 2048


# --- edit_file ---


EDIT_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "File path (relative or absolute, must stay within the working directory)",
        },
        "old_string": {
            "type": "string",
            "description": "The exact text to replace; must be unique in the file",
        },
        "new_string": {
            "type": "string",
            "description": "The text to replace old_string with",
        },
    },
    "required": ["path", "old_string", "new_string"],
}


def make_edit_file_tool(max_file_size: int = MAX_FILE_SIZE) -> ToolFunc:
    """Build the edit_file tool: patch-style exact replacement.

    The file is always read before it is written (prompt rule 3), and the
    edit is only applied when old_string occurs exactly once — ambiguity is
    an error, never a guess about which occurrence to change.
    """

    def _edit_file(args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        old_string = args.get("old_string")
        new_string = args.get("new_string")

        if not isinstance(old_string, str) or not old_string:
            return ToolResult("Error: old_string must be a non-empty string", is_error=True)
        if not isinstance(new_string, str):
            return ToolResult("Error: new_string must be a string", is_error=True)

        try:
            resolved = safe_path(str(path), Path.cwd())
        except ValueError as exc:
            return ToolResult(f"Error: {exc}", is_error=True)

        if not resolved.exists():
            return ToolResult(f"Error: file not found: {resolved}", is_error=True)
        if not resolved.is_file():
            return ToolResult(f"Error: not a regular file: {resolved}", is_error=True)

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

        # Ambiguity is an error: with multiple matches the agent must supply
        # more context rather than let the tool pick one silently.
        occurrences = content.count(old_string)
        if occurrences == 0:
            return ToolResult(
                f"Error: old_string not found in {resolved}",
                is_error=True,
            )
        if occurrences > 1:
            return ToolResult(
                f"Error: old_string is not unique ({occurrences} matches) in {resolved}; "
                "include more surrounding context",
                is_error=True,
            )

        updated = content.replace(old_string, new_string, 1)

        try:
            resolved.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(f"Error: cannot write {resolved}: {exc}", is_error=True)

        old_lines = content.count("\n")
        new_lines = updated.count("\n")
        delta = new_lines - old_lines
        return ToolResult(
            f"OK: edited {resolved} ({new_lines} lines, "
            f"{'+' if delta >= 0 else ''}{delta} lines)"
        )

    return _edit_file


# --- glob ---


GLOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern, e.g. '**/*.py' (recursive) or 'src/*.txt'",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (defaults to the working directory)",
        },
    },
    "required": ["pattern"],
}


def make_glob_tool(
    max_results: int = MAX_GLOB_RESULTS,
    max_output_size: int = MAX_OUTPUT_SIZE,
) -> ToolFunc:
    """Build the glob tool: list files matching a pattern.

    No matches is a normal empty result, not an error — the agent often
    globs precisely to confirm a file is absent.
    """

    def _glob(args: dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        search_path = args.get("path", ".")

        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult("Error: pattern must be a non-empty string", is_error=True)

        # An absolute pattern would escape the base directory entirely, so
        # it is rejected up front rather than passed to pathlib.
        if Path(pattern).is_absolute():
            return ToolResult(
                f"Error: pattern must be relative, not absolute: {pattern}",
                is_error=True,
            )

        try:
            base = safe_path(str(search_path), Path.cwd())
        except ValueError as exc:
            return ToolResult(f"Error: {exc}", is_error=True)

        if not base.is_dir():
            return ToolResult(f"Error: not a directory: {base}", is_error=True)

        # Path.glob understands '**' natively; a malformed pattern surfaces
        # as ValueError/NotImplementedError rather than a crash.
        try:
            matches = sorted(p for p in base.glob(pattern) if p.is_file())
        except (ValueError, NotImplementedError, OSError) as exc:
            return ToolResult(f"Error: invalid glob pattern '{pattern}': {exc}", is_error=True)

        if not matches:
            return ToolResult("No files matched")

        # Cap the count first, then the byte length, so both bounds hold.
        capped = matches[:max_results]
        text = "\n".join(str(p) for p in capped)
        if len(matches) > max_results:
            text += f"\n...[{len(matches) - max_results} more results omitted]"
        return ToolResult(_truncate(text, max_output_size))

    return _glob


# --- grep ---


GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regular expression to search for inside files",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in (defaults to the working directory)",
        },
        "include": {
            "type": "string",
            "description": "Filename glob filter, e.g. '*.py' (default: all files)",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matching lines to return",
        },
    },
    "required": ["pattern"],
}


def _is_binary(path: Path, sniff_bytes: int = BINARY_SNIFF_BYTES) -> bool:
    """Return True if the file looks binary (NUL byte in the first chunk)."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(sniff_bytes)
    except OSError:
        # If the file cannot be read at all, treat it as unreadable/binary
        # so the caller skips it instead of crashing.
        return True


def make_grep_tool(
    max_results: int = MAX_GREP_RESULTS,
    max_output_size: int = MAX_OUTPUT_SIZE,
    max_file_size: int = MAX_FILE_SIZE,
) -> ToolFunc:
    """Build the grep tool: regex search inside files, ripgrep-style output.

    The regex arrives from the LLM, so it is compiled defensively: an
    invalid pattern is reported as an error rather than raised. Binary files
    are skipped, and output is bounded by both match count and byte length.
    """

    def _grep(args: dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        search_path = args.get("path", ".")
        include = args.get("include")
        limit = args.get("max_results")

        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult("Error: pattern must be a non-empty string", is_error=True)

        # Untrusted input: a broken regex is a user error, not a crash.
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(f"Error: invalid regex '{pattern}': {exc}", is_error=True)

        try:
            base = safe_path(str(search_path), Path.cwd())
        except ValueError as exc:
            return ToolResult(f"Error: {exc}", is_error=True)

        if isinstance(limit, bool) or not isinstance(limit, int):
            limit = max_results
        limit = max(0, min(limit, max_results))

        # A single file is searched directly; a directory is walked.
        if base.is_file():
            candidates = [base]
        elif base.is_dir():
            try:
                candidates = sorted(p for p in base.rglob("*") if p.is_file())
            except OSError as exc:
                return ToolResult(f"Error: cannot walk {base}: {exc}", is_error=True)
        else:
            return ToolResult(f"Error: path does not exist: {base}", is_error=True)

        lines_out: list[str] = []
        matched = 0
        for candidate in candidates:
            if matched >= limit:
                break

            if include is not None and not fnmatch.fnmatch(candidate.name, str(include)):
                continue

            try:
                if candidate.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue

            if _is_binary(candidate):
                continue

            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    lines_out.append(f"{candidate}:{lineno}: {line}")
                    matched += 1
                    if matched >= limit:
                        break

        if not lines_out:
            return ToolResult("No matches")

        output = "\n".join(lines_out)
        return ToolResult(_truncate(output, max_output_size))

    return _grep


# --- extended registry ---


def extended_registry(
    bash_timeout: int = BASH_TIMEOUT,
    max_file_size: int = MAX_FILE_SIZE,
    max_output_size: int = MAX_OUTPUT_SIZE,
    max_glob_results: int = MAX_GLOB_RESULTS,
    max_grep_results: int = MAX_GREP_RESULTS,
) -> ToolRegistry:
    """Build the registry with all six tools: the three MVP tools plus the
    edit_file, glob, and grep tools added by this module.
    """
    registry = default_registry(
        bash_timeout=bash_timeout,
        max_file_size=max_file_size,
        max_output_size=max_output_size,
    )
    registry.register(
        ToolSpec(
            name="edit_file",
            description="Replace the unique occurrence of old_string with new_string in a file",
            parameters=EDIT_FILE_SCHEMA,
            func=make_edit_file_tool(max_file_size),
        )
    )
    registry.register(
        ToolSpec(
            name="glob",
            description="Find files matching a glob pattern (supports ** recursive)",
            parameters=GLOB_SCHEMA,
            func=make_glob_tool(max_glob_results, max_output_size),
        )
    )
    registry.register(
        ToolSpec(
            name="grep",
            description="Search file contents with a regular expression (ripgrep-style output)",
            parameters=GREP_SCHEMA,
            func=make_grep_tool(max_grep_results, max_output_size, max_file_size),
        )
    )
    return registry
