"""Skills engine: persistent store, loader, executor, and refiner (spec 6.4).

A 'Skill' is a named, documented Python snippet the agent can save, load, and
execute against a context dict. Skills live under ~/.synth/skills/ in SQLite
and can be pulled from GitHub raw URLs or YAML playbooks. Code is validated
before execution (static ban of dangerous builtins) and executed in a
restricted environment.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR
from synth.llm import LLMClient
from synth.tools import ToolResult

logger = logging.getLogger(__name__)

# --- constants -----------------------------------------------------------
# Where skills are persisted (SQLite). Matches SessionStore/MemoryStore layout.
DEFAULT_SKILLS_DB_PATH: Path = Path(CONFIG_DIR) / "skills.db"

# Hard cap on the number of skill code lines — keeps exec bounded.
MAX_SKILL_LINES: int = 400

# Builtins that are disallowed in skill code. Chosen conservatively:
# __import__ blocks dynamic imports; eval/exec block runtime code injection;
# open/globals/locals protect the host filesystem and runtime state.
BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {"__import__", "eval", "exec", "open", "globals", "locals", "compile"}
)

# Builtins whitelisted into the restricted execution namespace. Everything
# else is stripped. This is a defense-in-depth layer on top of BLOCKED_BUILTINS.
ALLOWED_BUILTINS: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "print": print,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "type": type,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "True": True,
    "False": False,
    "None": None,
}


class SkillError(Exception):
    """Raised when a skill cannot be saved, loaded, validated, or executed."""


@dataclass
class Skill:
    """A named, documented snippet of Python the agent can execute.

    Attributes:
        name: Unique identifier (lowercase, dash/dot/underscore allowed).
        description: One-line human description shown in list/search.
        code: The Python snippet to execute (plain, not compiled).
        created_at: When the skill was first saved (UTC, ISO-8601).
        usage_count: How many times this skill has been executed.
    """

    name: str
    description: str
    code: str
    created_at: str = ""
    usage_count: int = 0


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _validate_name(name: str) -> str:
    """Return a normalized name or raise SkillError on invalid input."""
    if not isinstance(name, str) or not name.strip():
        raise SkillError("skill name must be a non-empty string")
    lowered = name.strip().lower()
    if not _NAME_RE.match(lowered):
        raise SkillError(
            f"skill name {name!r} must match { _NAME_RE.pattern!r}"
        )
    return lowered


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SkillStore:
    """Persists skills under ~/.synth/skills.db with parameterized SQL.

    Args:
        db_path: Optional override for the SQLite database path. Defaults to
            ``CONFIG_DIR / "skills.db"``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_SKILLS_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path))
        except sqlite3.Error as exc:
            raise SkillError(f"cannot open skills db: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    # --- context manager ---
    def __enter__(self) -> "SkillStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with contextlib.suppress(Exception):
            self._conn.close()

    # --- schema ---
    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                created_at TEXT NOT NULL,
                usage_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    # --- CRUD ---
    def save_skill(self, skill: Skill) -> None:
        """Upsert a skill by name, keeping the original created_at."""
        name = _validate_name(skill.name)
        if not skill.description.strip():
            raise SkillError("skill description must be non-empty")
        if not skill.code.strip():
            raise SkillError("skill code must be non-empty")
        line_count = len(skill.code.splitlines())
        if line_count > MAX_SKILL_LINES:
            raise SkillError(
                f"skill code too long ({line_count} lines > {MAX_SKILL_LINES})"
            )

        existing = self.load_skill(name)
        created_at = skill.created_at or (existing.created_at if existing else _now_iso())

        try:
            self._conn.execute(
                """
                INSERT INTO skills (name, description, code, created_at, usage_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    code=excluded.code,
                    created_at=excluded.created_at,
                    usage_count=excluded.usage_count
                """,
                (name, skill.description.strip(), skill.code, created_at, skill.usage_count),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SkillError(f"failed to save skill {name!r}: {exc}") from exc

    def load_skill(self, name: str) -> Skill | None:
        """Return the skill with the given name or None when absent."""
        lowered = _validate_name(name)
        try:
            row = self._conn.execute(
                "SELECT name, description, code, created_at, usage_count FROM skills WHERE name = ?",
                (lowered,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise SkillError(f"failed to load skill {name!r}: {exc}") from exc
        if row is None:
            return None
        return Skill(
            name=row["name"],
            description=row["description"],
            code=row["code"],
            created_at=row["created_at"],
            usage_count=row["usage_count"],
        )

    def list_skills(self) -> list[Skill]:
        """Return all skills ordered by name ascending."""
        try:
            rows = self._conn.execute(
                "SELECT name, description, code, created_at, usage_count FROM skills ORDER BY name"
            ).fetchall()
        except sqlite3.Error as exc:
            raise SkillError(f"failed to list skills: {exc}") from exc
        return [
            Skill(
                name=r["name"],
                description=r["description"],
                code=r["code"],
                created_at=r["created_at"],
                usage_count=r["usage_count"],
            )
            for r in rows
        ]

    def delete_skill(self, name: str) -> bool:
        """Delete the skill with the given name; return True iff a row was removed."""
        lowered = _validate_name(name)
        try:
            cur = self._conn.execute("DELETE FROM skills WHERE name = ?", (lowered,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SkillError(f"failed to delete skill {name!r}: {exc}") from exc
        return cur.rowcount > 0

    def search_skills(self, query: str) -> list[Skill]:
        """Return skills whose name or description contain query (case-insensitive)."""
        if not isinstance(query, str) or not query.strip():
            raise SkillError("search query must be a non-empty string")
        like = f"%{query.strip()}%"
        try:
            rows = self._conn.execute(
                "SELECT name, description, code, created_at, usage_count FROM skills "
                "WHERE name LIKE ? OR description LIKE ? ORDER BY name",
                (like, like),
            ).fetchall()
        except sqlite3.Error as exc:
            raise SkillError(f"failed to search skills: {exc}") from exc
        return [
            Skill(
                name=r["name"],
                description=r["description"],
                code=r["code"],
                created_at=r["created_at"],
                usage_count=r["usage_count"],
            )
            for r in rows
        ]

    def increment_usage(self, name: str) -> None:
        """Increment usage_count for the given skill (no-op if missing)."""
        lowered = _validate_name(name)
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute(
                "UPDATE skills SET usage_count = usage_count + 1 WHERE name = ?",
                (lowered,),
            )
            self._conn.commit()


class SkillExecutor:
    """Execute skill code in a restricted environment.

    Args:
        store: The backing SkillStore (used to bump usage_count on execute).
    """

    def __init__(self, store: SkillStore | None = None) -> None:
        self.store = store

    def validate_code(self, code: str) -> bool:
        """Return True when code passes static safety checks.

        Rejects: dangerous builtins, lines over MAX_SKILL_LINES, and code
        that cannot be compiled. The check is heuristic — it catches the
        common attack vectors, not every possible exploit.
        """
        if not isinstance(code, str) or not code.strip():
            return False
        if len(code.splitlines()) > MAX_SKILL_LINES:
            return False
        for bad in BLOCKED_BUILTINS:
            # Match whole-token usage; substring matches in identifiers are OK.
            if re.search(rf"\b{re.escape(bad)}\b", code):
                return False
        try:
            compile(code, "<skill>", "exec")
        except SyntaxError:
            return False
        return True

    def execute(self, skill: Skill, context: dict[str, Any] | None = None) -> ToolResult:
        """Run skill.code with context injected into globals['context'].

        The restricted environment exposes only ALLOWED_BUILTINS plus the
        context dict. stdout/stderr are captured and returned as the result
        text (with the function's return value appended, if any). Errors
        are returned as ToolResult(is_error=True).
        """
        if not self.validate_code(skill.code):
            return ToolResult(
                f"skill {skill.name!r} failed validation", is_error=True
            )

        stdout = io.StringIO()
        captured: dict[str, Any] = {}
        globals_ns: dict[str, Any] = {
            "__builtins__": ALLOWED_BUILTINS,
            "context": dict(context or {}),
        }

        old_stdout = None
        try:
            old_stdout = sys.stdout
            sys.stdout = stdout  # type: ignore[name-defined]  # noqa: F821
            exec(compile(skill.code, f"<skill:{skill.name}>", "exec"), globals_ns)
        except Exception as exc:
            captured_text = stdout.getvalue()
            return ToolResult(
                f"skill {skill.name!r} raised {type(exc).__name__}: {exc}\n"
                f"{captured_text}".rstrip(),
                is_error=True,
            )
        finally:
            if old_stdout is not None:
                sys.stdout = old_stdout  # type: ignore[name-defined]  # noqa: F821

        captured_text = stdout.getvalue().rstrip()
        ret = globals_ns.get("result") or globals_ns.get("_result")
        parts = []
        if captured_text:
            parts.append(captured_text)
        if ret is not None:
            parts.append(repr(ret))
        if not parts:
            parts.append(f"skill {skill.name!r} executed with no output")

        if self.store is not None:
            with contextlib.suppress(Exception):
                self.store.increment_usage(skill.name)
        return ToolResult("\n".join(parts), is_error=False)


# --- loader --------------------------------------------------------------

class SkillLoader:
    """Fetch skill snippets from remote URLs or YAML playbooks.

    Args:
        timeout: HTTP request timeout in seconds for from_github.
    """

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def from_github(self, repo: str, path: str, name: str | None = None) -> Skill:
        """Fetch a skill from a GitHub raw content URL.

        Args:
            repo: GitHub slug like "Mercphobia/Synth".
            path: Path inside the repo like "skills/foo.py".
            name: Optional skill name (default: stem of path, lowercased).

        Returns:
            A Skill with code = fetched content.

        Raises:
            SkillError: On HTTP failure or empty response.
        """
        if not isinstance(repo, str) or "/" not in repo:
            raise SkillError(f"repo must be 'owner/name', got {repo!r}")
        if not isinstance(path, str) or not path.strip():
            raise SkillError("path must be a non-empty string")
        url = f"https://raw.githubusercontent.com/{repo}/main/{path.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = resp.read()
        except Exception as exc:
            raise SkillError(f"failed to fetch {url}: {exc}") from exc
        try:
            code = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"skill code at {url} is not UTF-8") from exc
        if not code.strip():
            raise SkillError(f"skill code at {url} is empty")
        skill_name = name or Path(path).stem.lower()
        return Skill(
            name=_validate_name(skill_name),
            description=f"imported from {repo} @ {path}",
            code=code,
        )

    def from_playbook(self, yaml_str: str) -> Skill:
        """Parse a YAML playbook into a Skill.

        The playbook format is:

            name: my-skill
            description: What it does
            steps:
              - action: read_file
                args: {path: foo.txt}
              - action: write_file
                args: {path: bar.txt, content: hi}

        Each step becomes one line of Python calling the action as a
        function with the args dict spread in.
        """
        if not isinstance(yaml_str, str) or not yaml_str.strip():
            raise SkillError("playbook must be a non-empty YAML string")

        # Minimal YAML parser that handles the documented subset without
        # depending on PyYAML.
        name = ""
        description = ""
        steps: list[dict[str, Any]] = []
        try:
            parsed = _parse_minimal_yaml(yaml_str)
        except ValueError as exc:
            raise SkillError(f"invalid playbook YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SkillError("playbook YAML must be a mapping")

        name = str(parsed.get("name", "")).strip()
        description = str(parsed.get("description", "")).strip()
        raw_steps = parsed.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            raise SkillError("playbook must have a non-empty 'steps' list")

        for step in raw_steps:
            if not isinstance(step, dict):
                raise SkillError(f"playbook step must be a mapping, got {type(step).__name__}")
            action = step.get("action")
            if not action:
                raise SkillError("playbook step missing 'action'")
            args = step.get("args") or {}
            if not isinstance(args, dict):
                raise SkillError(f"step {action!r} args must be a mapping")
            steps.append({"action": str(action), "args": args})

        # Build Python code that calls into the tools registry.
        lines = ["def run(context):", "    tools = context['tools']"]
        for i, step in enumerate(steps):
            args_repr = repr(step["args"])
            lines.append(
                f"    _step_{i} = tools.execute({step['action']!r}, {args_repr})"
            )
            lines.append(
                f"    if _step_{i}.is_error: return _step_{i}.text"
            )
        lines.append("    return 'playbook completed'")
        code = "\n".join(lines) + "\n"

        return Skill(
            name=_validate_name(name) if name else "playbook",
            description=description or "imported from playbook YAML",
            code=code,
        )


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the minimal YAML subset used by from_playbook.

    Handles top-level key/value strings and a top-level 'steps:' list of
    inline dicts. Raises ValueError on anything outside that subset.
    """
    import re

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    result: dict[str, Any] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_-]*)\s*:\s*(.*)$", line)
        if not m:
            raise ValueError(f"unrecognized line: {line!r}")
        key, value = m.group(1), m.group(2).strip()
        if value == "":
            # expect a list or block below
            items = []
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(_parse_inline_mapping(lines[i]))
                i += 1
            result[key] = items
            continue
        result[key] = _parse_scalar(value)
        i += 1
    return result


def _parse_inline_mapping(line: str) -> dict[str, Any]:
    """Parse a line like '  - {path: foo.txt, content: hi}' into a dict."""
    import re

    body = line.lstrip().lstrip("- ").strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    out: dict[str, Any] = {}
    for pair in re.split(r",\s*", body):
        if not pair.strip():
            continue
        k, _, v = pair.partition(":")
        out[k.strip()] = _parse_scalar(v.strip())
    return out


def _parse_scalar(value: str) -> Any:
    """Best-effort scalar parsing: int, float, bool, quoted str, or bare str."""
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "none"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# --- refiner --------------------------------------------------------------

class SkillRefiner:
    """Use an LLMClient to rewrite a skill's code based on feedback.

    Args:
        llm: A configured LLMClient to call for the rewrite.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def refine(self, skill: Skill, feedback: str) -> Skill:
        """Return a new Skill with code rewritten per feedback.

        Raises:
            SkillError: When the LLM returns no text or the result fails
                validate_code.
        """
        if not isinstance(feedback, str) or not feedback.strip():
            raise SkillError("refine feedback must be a non-empty string")

        prompt = (
            "Rewrite the following Python skill code based on the feedback. "
            "Output ONLY the new code, no explanation.\n\n"
            f"Skill name: {skill.name}\n"
            f"Current code:\n```\n{skill.code}\n```\n\n"
            f"Feedback: {feedback.strip()}"
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                stream=False,
            )
        except Exception as exc:
            raise SkillError(f"refine LLM call failed: {exc}") from exc

        new_code = (response.text or "").strip()
        # Strip markdown fences if present.
        if new_code.startswith("```"):
            first_nl = new_code.find("\n")
            last_fence = new_code.rfind("```")
            if first_nl != -1 and last_fence > first_nl:
                new_code = new_code[first_nl + 1 : last_fence].strip()

        if not new_code:
            raise SkillError("refine LLM returned no code")

        # Validate before returning — the caller is responsible for saving.
        if not SkillExecutor().validate_code(new_code):
            raise SkillError(
                "refine LLM produced code that fails validation; rejected"
            )

        return Skill(
            name=skill.name,
            description=skill.description + " (refined)",
            code=new_code,
            created_at=skill.created_at,
            usage_count=skill.usage_count,
        )
