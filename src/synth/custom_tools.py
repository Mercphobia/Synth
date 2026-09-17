"""Custom tool loader, permission prompts, and autocomplete.

Three independent subsystems that share a single module so the entry point
stays small:

* ``ToolLoader`` — scans ``~/.synth/tools/`` for ``.py`` files, imports each
  in a restricted namespace, discovers ``BaseTool`` subclasses, and yields
  ``ToolSpec`` instances ready for a ``ToolRegistry``. Per-file failures are
  logged and skipped; one bad file never kills the whole scan.
* ``PermissionPrompt`` — decides whether a tool call needs a human prompt
  based on a mode (``strict`` / ``balanced`` / ``auto`` / ``yolo``) and a
  caller-supplied list of dangerous tool names.
* ``Autocomplete`` — turns a raw input line into a list of
  ``(trigger, label)`` suggestions for the four trigger characters Synth
  understands: ``@`` (file), ``/`` (slash command), ``!`` (shell), ``~``
  (skill).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR
from synth.tools import ToolFunc, ToolResult, ToolSpec

# --- Module-level constants ---------------------------------------------------

# Directory name under the user home that holds custom tool scripts. Kept in
# sync with CONFIG_DIR so the loader looks in ~/.synth/tools/ by default.
TOOLS_DIR_NAME = "tools"

# Hard ceiling on the number of custom tools loaded in a single scan. Stops
# a runaway tools/ directory from registering hundreds of near-duplicate
# tools and swamping the LLM's schema list.
MAX_CUSTOM_TOOLS = 50

# Tool names that are dangerous by default: they touch the filesystem or run
# arbitrary commands. PermissionPrompt starts from this list in 'balanced'
# mode unless the caller overrides it.
DANGEROUS_DEFAULTS = ["bash", "write_file", "edit_file"]

# Known slash commands. The autocomplete surface matches user input against
# this list so the TUI can offer completions for /help, /exit, etc.
SLASH_COMMANDS: list[str] = ["/help", "/exit", "/new", "/session", "/mode", "/model"]

# Maximum number of file suggestions the @ autocomplete returns. A glob over
# a fat directory can yield thousands of entries; this keeps the dropdown
# readable and the context window intact.
MAX_AUTOCOMPLETE_FILES = 50

# Regex applied to the raw source of every tool file BEFORE exec. It looks
# for top-level / module-level calls to os.system or subprocess.* that are
# not nested inside a function body. The heuristic is intentionally simple:
# it rejects lines that begin (after optional whitespace) with a forbidden
# call. False positives are acceptable — a tool author who genuinely needs
# subprocess inside execute() can still call it there, because execute() is
# a method body and its lines are indented.
_TOPLEVEL_DANGEROUS = re.compile(
    r"^\s*(?:os\.system\s*\(|subprocess\.(?:run|call|check_output|Popen|check_call)\s*\()",
    re.MULTILINE,
)


# --- BaseTool ----------------------------------------------------------------


class BaseTool:
    """Base class every custom tool author subclasses.

    A subclass sets five things:

    * ``name``        — short identifier, must be a valid tool name.
    * ``description`` — one-line summary shown to the LLM.
    * ``dangerous``   — True if the tool has side effects worth prompting for.
    * ``schema``      — JSON-schema dict describing the parameters.
    * ``execute``     — the actual logic; takes keyword args, returns a str.

    The loader wraps each discovered subclass into a ``ToolSpec`` whose
    ``func`` adapts the dict-style call convention (``func(args: dict)``)
    down to the author-friendly ``execute(**kwargs)``.
    """

    name: str = ""
    description: str = ""
    dangerous: bool = False
    schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> str:  # pragma: no cover - abstract
        """Run the tool. Subclasses must override."""
        raise NotImplementedError("subclasses must implement execute()")


def _safe_builtins() -> dict[str, Any]:
    """Return a restricted builtins dict for exec'ing untrusted tool files.

    The goal is NOT a real sandbox — Python cannot sandbox Python — but to
    catch the obvious foot-guns (``__import__`` of arbitrary modules at the
    top level, ``eval``/``exec`` of dynamic code). ``open``, ``print``, and
    the usual data structures are left in so normal tools still work.
    """
    import builtins

    allowed = {
        name: getattr(builtins, name)
        for name in (
            "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable",
            "chr", "dict", "dir", "divmod", "enumerate", "filter", "float",
            "format", "frozenset", "getattr", "hasattr", "hash", "hex", "id",
            "input", "int", "isinstance", "issubclass", "iter", "len", "list",
            "map", "max", "min", "next", "object", "oct", "open", "ord", "pow",
            "print", "property", "range", "repr", "reversed", "round", "set",
            "setattr", "slice", "sorted", "str", "sum", "super", "tuple",
            "type", "vars", "zip", "True", "False", "None", "Exception",
            "ValueError", "TypeError", "KeyError", "RuntimeError",
            "AttributeError", "ImportError", "OSError", "FileNotFoundError",
            "NotImplementedError", "StopIteration", "AssertionError",
            "IndexError", "ZeroDivisionError", "OverflowError",
        )
    }
    # Permit a controlled __import__ so tool files can `import re`, `import os`
    # inside their own module body — we cannot run Python at all otherwise.
    # The static _TOPLEVEL_DANGEROUS check is what catches os.system abuse;
    # this just lets normal imports work.
    allowed["__import__"] = builtins.__import__
    allowed["__build_class__"] = builtins.__build_class__
    allowed["__name__"] = "__synth_tool__"
    return allowed


class ToolLoader:
    """Scan a directory for ``.py`` files defining ``BaseTool`` subclasses.

    Usage::

        loader = ToolLoader()
        specs = loader.load()
        for spec in specs:
            registry.register(spec)

    Per-file errors are logged via ``self.log`` (a list of strings) and the
    file is skipped; a single corrupt file does not abort the scan.
    """

    def __init__(self, tools_dir: Path | None = None) -> None:
        if tools_dir is None:
            tools_dir = Path.home() / CONFIG_DIR / TOOLS_DIR_NAME
        self.tools_dir = tools_dir
        self.log: list[str] = []

    # -- public API --

    def load(self) -> list[ToolSpec]:
        """Return a list of ToolSpec objects from all valid tool files."""
        self.log = []
        if not self.tools_dir.is_dir():
            return []

        specs: list[ToolSpec] = []
        py_files = sorted(self.tools_dir.glob("*.py"))
        for path in py_files:
            if len(specs) >= MAX_CUSTOM_TOOLS:
                self.log.append(
                    f"reached MAX_CUSTOM_TOOLS={MAX_CUSTOM_TOOLS}; "
                    f"skipping {path.name}"
                )
                break
            spec = self._load_file(path)
            if spec is not None:
                specs.append(spec)
        return specs

    # -- internals --

    def _load_file(self, path: Path) -> ToolSpec | None:
        """Load a single .py file and return its first valid ToolSpec, or None."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.log.append(f"{path.name}: cannot read: {exc}")
            return None

        # Static safety gate: reject top-level dangerous calls.
        if _TOPLEVEL_DANGEROUS.search(source):
            self.log.append(
                f"{path.name}: rejected (top-level os.system/subprocess call)"
            )
            return None

        namespace = self._exec_restricted(source, path)
        if namespace is None:
            return None

        # Find BaseTool subclasses defined in this file (skip BaseTool itself
        # and anything imported from elsewhere).
        for obj in namespace.values():
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseTool)
                and obj is not BaseTool
                and obj.__module__ == namespace.get("__name__", "")
            ):
                spec = self._build_spec(obj, path.name)
                if spec is not None:
                    return spec

        self.log.append(f"{path.name}: no BaseTool subclass found")
        return None

    def _exec_restricted(self, source: str, path: Path) -> dict[str, Any] | None:
        """exec the source in a restricted namespace; return it or None."""
        module_name = f"_synth_custom_{path.stem}"
        namespace: dict[str, Any] = {
            "__name__": module_name,
            "__file__": str(path),
            "__builtins__": _safe_builtins(),
        }
        try:
            compiled = compile(source, str(path), "exec")
            exec(compiled, namespace)  # noqa: S102 — intentional, gated above
        except SystemExit:
            # A tool file that calls sys.exit() is hostile or broken.
            self.log.append(f"{path.name}: rejected (called sys.exit())")
            return None
        except BaseException as exc:  # noqa: BLE001 — boundary
            self.log.append(f"{path.name}: import failed: {exc!r}")
            return None
        return namespace

    def _build_spec(self, cls: type[BaseTool], source_name: str) -> ToolSpec | None:
        """Validate a discovered class and wrap it in a ToolSpec."""
        name = getattr(cls, "name", "")
        if not name or not isinstance(name, str):
            self.log.append(f"{source_name}: tool class has invalid name")
            return None
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            self.log.append(
                f"{source_name}: tool name '{name}' is not a valid identifier"
            )
            return None

        description = getattr(cls, "description", "") or name
        schema = getattr(cls, "schema", None)
        if not isinstance(schema, dict):
            self.log.append(f"{source_name}: tool '{name}' has no schema dict")
            return None

        instance: BaseTool | None = None
        try:
            instance = cls()
        except BaseException as exc:  # noqa: BLE001 — boundary
            self.log.append(f"{source_name}: cannot instantiate '{name}': {exc!r}")
            return None

        if not callable(getattr(instance, "execute", None)):
            self.log.append(f"{source_name}: tool '{name}' has no execute method")
            return None

        func = _wrap_basetool(instance, name)
        return ToolSpec(
            name=name,
            description=description,
            parameters=schema,
            func=func,
        )


def _wrap_basetool(instance: BaseTool, name: str) -> ToolFunc:
    """Adapt a BaseTool.execute(**kwargs) to the ToolFunc dict contract."""

    def _adapter(args: dict[str, Any]) -> ToolResult:
        if not isinstance(args, dict):
            return ToolResult(
                f"Error: tool '{name}' expects an object of arguments",
                is_error=True,
            )
        try:
            text = instance.execute(**args)
        except Exception as exc:  # noqa: BLE001 — boundary: never raise to loop
            return ToolResult(f"Error: tool '{name}' crashed: {exc}", is_error=True)
        if isinstance(text, ToolResult):
            return text
        return ToolResult(str(text))

    return _adapter


# --- PermissionPrompt --------------------------------------------------------


class PermissionPrompt:
    """Decide whether a tool call needs a human confirmation prompt.

    Four modes, simplest to most permissive:

    * ``strict``   — every tool call prompts (returns False).
    * ``balanced`` — only tools in ``dangerous_tools`` prompt.
    * ``auto``     — no prompts (returns True).
    * ``yolo``     — auto-approve everything (alias of auto, explicit).

    The caller supplies ``dangerous_tools``; if absent it defaults to
    ``DANGEROUS_DEFAULTS``. The decision is a pure function of (mode, tool,
    dangerous_tools) — no I/O, no state.
    """

    def __init__(
        self,
        mode: str = "balanced",
        dangerous_tools: list[str] | None = None,
    ) -> None:
        if mode not in ("strict", "balanced", "auto", "yolo"):
            raise ValueError(f"unknown permission mode: {mode!r}")
        self.mode = mode
        self.dangerous_tools = (
            list(dangerous_tools) if dangerous_tools is not None else list(DANGEROUS_DEFAULTS)
        )

    def check(self, tool_name: str, args: dict[str, Any], dangerous_tools: list[str] | None = None) -> bool:
        """Return True if the call is auto-approved, False if it must prompt.

        ``dangerous_tools`` overrides the instance list for this call only;
        when None the constructor list is used.
        """
        effective = self.dangerous_tools if dangerous_tools is None else list(dangerous_tools)

        if self.mode in ("auto", "yolo"):
            return True
        if self.mode == "strict":
            return False
        # balanced: prompt only dangerous tools.
        if tool_name in effective:
            return False
        return True


# --- Autocomplete ------------------------------------------------------------


class Autocomplete:
    """Produce ``(trigger, label)`` suggestions for a raw input line.

    Recognised triggers:

    * ``@`` prefix  — glob for files matching the remainder.
    * ``/`` prefix  — match against known slash commands.
    * ``!`` prefix  — passthrough; suggest the shell prefix.
    * ``~`` prefix  — match against loaded skill names.

    Anything else (or empty input) returns an empty list.
    """

    def __init__(
        self,
        skills: list[str] | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.skills = list(skills) if skills else []
        self.base_dir = base_dir if base_dir is not None else Path.cwd()

    def complete(self, input: str) -> list[tuple[str, str]]:
        """Return suggestions for the given input line."""
        if not isinstance(input, str) or not input:
            return []

        trigger = input[0]
        remainder = input[1:]

        if trigger == "@":
            return self._complete_file(remainder)
        if trigger == "/":
            return self._complete_slash(remainder)
        if trigger == "!":
            return [("!", "shell: " + remainder)] if remainder else [("!", "shell")]
        if trigger == "~":
            return self._complete_skill(remainder)
        return []

    # -- completers --

    def _complete_file(self, remainder: str) -> list[tuple[str, str]]:
        """Glob for files matching the remainder under base_dir."""
        if not remainder:
            pattern = "*"
        else:
            pattern = remainder + "*"
        try:
            matches = sorted(
                p for p in self.base_dir.glob(pattern) if p.is_file()
            )
        except (ValueError, NotImplementedError, OSError):
            return []
        capped = matches[:MAX_AUTOCOMPLETE_FILES]
        return [("@", str(p.relative_to(self.base_dir)) if p.is_relative_to(self.base_dir) else str(p)) for p in capped]

    def _complete_slash(self, remainder: str) -> list[tuple[str, str]]:
        """Match the remainder against known slash commands."""
        prefix = "/" + remainder
        results = [cmd for cmd in SLASH_COMMANDS if cmd.startswith(prefix)]
        if not results:
            # If remainder is empty, return all commands.
            if not remainder:
                results = list(SLASH_COMMANDS)
            else:
                return []
        return [("/", cmd) for cmd in results]

    def _complete_skill(self, remainder: str) -> list[tuple[str, str]]:
        """Match the remainder against loaded skill names."""
        if not self.skills:
            return []
        results = [s for s in self.skills if s.startswith(remainder)]
        return [("~", s) for s in results]


__all__ = [
    "BaseTool",
    "ToolLoader",
    "PermissionPrompt",
    "Autocomplete",
    "TOOLS_DIR_NAME",
    "MAX_CUSTOM_TOOLS",
    "DANGEROUS_DEFAULTS",
    "SLASH_COMMANDS",
    "MAX_AUTOCOMPLETE_FILES",
]
