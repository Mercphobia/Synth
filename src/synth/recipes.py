"""Recipes, self-update, and live metrics (spec #55, #124, #135).

A 'Recipe' is a reusable prompt template with {variable} placeholders that can
be stored, rendered, and shared. LiveMetrics records model call telemetry and
formats it as an ASCII table. SelfUpdater checks the git remote for newer tags
and pulls the main branch.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from synth.constants import CONFIG_DIR

# --- constants -----------------------------------------------------------

RECIPE_DB_FILENAME = "recipes.db"
# Keep only the last N calls for rolling average latency. Bounds memory and
# keeps the average responsive to recent performance, not historical drift.
METRICS_WINDOW = 100


class RecipeError(Exception):
    """Raised when a recipe cannot be saved, loaded, rendered, or parsed."""


# --- Recipe dataclass ---------------------------------------------------

@dataclass
class Recipe:
    """A reusable prompt template with {variable} placeholders.

    Attributes:
        name: Unique identifier for the recipe.
        template: Prompt text containing zero or more {variable} markers.
        variables: List of variable names extracted from the template.
        description: One-line human description.
    """

    name: str
    template: str
    variables: list[str] = field(default_factory=list)
    description: str = ""


_VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_variables(template: str) -> list[str]:
    """Return unique variable names found in {var} patterns, in order of first appearance."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _VAR_RE.finditer(template):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# --- RecipeStore --------------------------------------------------------

class RecipeStore:
    """Persists recipes in SQLite under ~/.synth/recipes.db.

    Args:
        db_path: Optional override for the SQLite database path. Defaults to
            ``CONFIG_DIR / "recipes.db"``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or Path(CONFIG_DIR) / RECIPE_DB_FILENAME
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path))
        except sqlite3.Error as exc:
            raise RecipeError(f"cannot open recipes db: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                name TEXT PRIMARY KEY,
                template TEXT NOT NULL,
                variables TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.commit()

    def save(self, recipe: Recipe) -> None:
        """Upsert a recipe by name. Variables are re-extracted from the template."""
        if not isinstance(recipe.name, str) or not recipe.name.strip():
            raise RecipeError("recipe name must be a non-empty string")
        if not isinstance(recipe.template, str) or not recipe.template.strip():
            raise RecipeError("recipe template must be non-empty")
        name = recipe.name.strip()
        variables = _extract_variables(recipe.template)
        import json
        try:
            self._conn.execute(
                """
                INSERT INTO recipes (name, template, variables, description)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    template=excluded.template,
                    variables=excluded.variables,
                    description=excluded.description
                """,
                (name, recipe.template, json.dumps(variables), recipe.description),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RecipeError(f"failed to save recipe {name!r}: {exc}") from exc

    def load(self, name: str) -> Recipe | None:
        """Return the recipe with the given name or None when absent."""
        if not isinstance(name, str) or not name.strip():
            return None
        import json
        try:
            row = self._conn.execute(
                "SELECT name, template, variables, description FROM recipes WHERE name = ?",
                (name.strip(),),
            ).fetchone()
        except sqlite3.Error as exc:
            raise RecipeError(f"failed to load recipe {name!r}: {exc}") from exc
        if row is None:
            return None
        try:
            variables = json.loads(row["variables"]) if row["variables"] else []
        except (ValueError, TypeError):
            variables = []
        return Recipe(
            name=row["name"],
            template=row["template"],
            variables=variables,
            description=row["description"],
        )

    def list_all(self) -> list[Recipe]:
        """Return all recipes ordered by name ascending."""
        import json
        try:
            rows = self._conn.execute(
                "SELECT name, template, variables, description FROM recipes ORDER BY name"
            ).fetchall()
        except sqlite3.Error as exc:
            raise RecipeError(f"failed to list recipes: {exc}") from exc
        recipes: list[Recipe] = []
        for r in rows:
            try:
                variables = json.loads(r["variables"]) if r["variables"] else []
            except (ValueError, TypeError):
                variables = []
            recipes.append(
                Recipe(
                    name=r["name"],
                    template=r["template"],
                    variables=variables,
                    description=r["description"],
                )
            )
        return recipes

    def render(self, name: str, variables: dict[str, str]) -> str:
        """Substitute {var} in the recipe's template with provided values.

        Raises:
            RecipeError: When the recipe is not found.
            ValueError: When a required variable is missing from the dict.
        """
        recipe = self.load(name)
        if recipe is None:
            raise RecipeError(f"recipe {name!r} not found")
        result = recipe.template
        for var in recipe.variables:
            if var not in variables:
                raise ValueError(f"missing variable: {var!r}")
            result = result.replace(f"{{{var}}}", str(variables[var]))
        return result

    def delete(self, name: str) -> bool:
        """Delete the recipe with the given name; return True iff a row was removed."""
        if not isinstance(name, str) or not name.strip():
            return False
        try:
            cur = self._conn.execute("DELETE FROM recipes WHERE name = ?", (name.strip(),))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RecipeError(f"failed to delete recipe {name!r}: {exc}") from exc
        return cur.rowcount > 0

    @staticmethod
    def from_text(text: str) -> Recipe:
        """Parse a text block into a Recipe.

        Format:
            # name: <name>
            description: <desc>
            <rest is the template>

        The first line must start with '# name:' or 'name:' to supply the name.
        Raises ValueError on missing name.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        lines = text.strip().splitlines()
        if not lines:
            raise ValueError("empty text block — cannot parse recipe name")

        # First line: accept '# name: <name>' or 'name: <name>'
        first = lines[0].strip()
        name = ""
        if first.startswith("#"):
            first = first[1:].strip()
        m = re.match(r"^name\s*:\s*(.+)$", first, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
        if not name:
            raise ValueError("missing recipe name: first line must be '# name: <name>'")

        # Second line: 'description: <desc>' (optional)
        description = ""
        template_start = 1
        if len(lines) > 1:
            second = lines[1].strip()
            m2 = re.match(r"^#?\s*description\s*:\s*(.*)$", second, re.IGNORECASE)
            if m2:
                description = m2.group(1).strip()
                template_start = 2

        template = "\n".join(lines[template_start:]).strip()
        if not template:
            raise ValueError("recipe template is empty")
        return Recipe(
            name=name,
            template=template,
            variables=_extract_variables(template),
            description=description,
        )


# --- SelfUpdater --------------------------------------------------------

class SelfUpdater:
    """Check the git remote for newer tags and pull the main branch.

    Args:
        repo_path: Repository root directory. Defaults to the current working directory.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        self.repo_path = repo_path or Path.cwd()

    def _run(self, args: list[str]) -> str:
        """Run a git command in repo_path and return stdout. Raise on failure."""
        try:
            result = subprocess.run(
                args,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"git not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"git command timed out: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    def _current_version(self) -> str:
        """Read the version field from pyproject.toml in the repo root."""
        pyproject = self.repo_path / "pyproject.toml"
        if not pyproject.exists():
            return "0.0.0"
        content = pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        return m.group(1) if m else "0.0.0"

    @staticmethod
    def _parse_tag_version(ref: str) -> str | None:
        """Extract a semantic version from a git ref like 'refs/tags/v1.2.3'."""
        m = re.search(r"v?(\d+\.\d+\.\d+)", ref)
        return m.group(1) if m else None

    def check_for_update(self) -> dict | None:
        """Check the git remote for newer tags.

        Returns:
            ``{'current': str, 'latest': str, 'update_available': bool}`` or
            ``None`` if not in a git repo.
        """
        # Verify we are in a git repo.
        try:
            self._run(["git", "rev-parse", "--git-dir"])
        except RuntimeError:
            return None

        current = self._current_version()
        try:
            raw = self._run(["git", "ls-remote", "--tags", "origin"])
        except RuntimeError:
            return None

        versions: list[str] = []
        for line in raw.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ref = parts[1]
            v = self._parse_tag_version(ref)
            if v:
                versions.append(v)

        if not versions:
            return {
                "current": current,
                "latest": current,
                "update_available": False,
            }

        latest = _max_semver(versions)
        return {
            "current": current,
            "latest": latest,
            "update_available": _semver_gt(latest, current),
        }

    def update(self) -> bool:
        """Run ``git pull origin main`` in the repo root.

        Returns:
            True on success.

        Raises:
            RuntimeError: If not in a git repo or the pull fails.
        """
        try:
            self._run(["git", "rev-parse", "--git-dir"])
        except RuntimeError as exc:
            raise RuntimeError(f"not in a git repo: {exc}") from exc
        self._run(["git", "pull", "origin", "main"])
        return True


def _semver_tuple(v: str) -> tuple[int, ...]:
    """Parse '1.2.3' into (1, 2, 3). Non-numeric parts become 0."""
    parts: list[int] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _semver_gt(a: str, b: str) -> bool:
    """Return True if version a is strictly greater than version b."""
    return _semver_tuple(a) > _semver_tuple(b)


def _max_semver(versions: list[str]) -> str:
    """Return the maximum version from a list of semantic version strings."""
    best = versions[0]
    for v in versions[1:]:
        if _semver_gt(v, best):
            best = v
    return best


# --- LiveMetrics --------------------------------------------------------

class LiveMetrics:
    """Record and report model call telemetry.

    Keeps the last ``METRICS_WINDOW`` calls for a rolling average latency,
    while maintaining cumulative totals.
    """

    def __init__(self, window: int = METRICS_WINDOW) -> None:
        self._window = window
        self._calls: deque[dict] = deque(maxlen=window)
        self._total_calls = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model: dict[str, dict] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        """Record a single model call."""
        self._total_calls += 1
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        entry = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
        self._calls.append(entry)
        if model not in self._by_model:
            self._by_model[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latencies": deque(maxlen=self._window),
            }
        m = self._by_model[model]
        m["calls"] += 1
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["latencies"].append(latency_ms)

    def snapshot(self) -> dict:
        """Return a snapshot of current cumulative + rolling metrics."""
        latencies = [c["latency_ms"] for c in self._calls]
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        by_model: dict[str, dict] = {}
        for model, m in self._by_model.items():
            m_lat = list(m["latencies"])
            m_avg = sum(m_lat) / len(m_lat) if m_lat else 0.0
            by_model[model] = {
                "calls": m["calls"],
                "input_tokens": m["input_tokens"],
                "output_tokens": m["output_tokens"],
                "avg_latency_ms": round(m_avg, 2),
            }
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_latency_ms": round(avg, 2),
            "by_model": by_model,
        }

    def format_table(self) -> str:
        """Render current metrics as an ASCII table (like ``synth stats``)."""
        snap = self.snapshot()
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("  Synth Live Metrics")
        lines.append("=" * 72)
        lines.append(f"  Total calls:          {snap['total_calls']}")
        lines.append(f"  Total input tokens:   {snap['total_input_tokens']}")
        lines.append(f"  Total output tokens:  {snap['total_output_tokens']}")
        lines.append(f"  Avg latency (ms):     {snap['avg_latency_ms']}")
        lines.append("-" * 72)
        if snap["by_model"]:
            header = f"  {'Model':<30} {'Calls':>8} {'In':>10} {'Out':>10} {'Avg ms':>10}"
            lines.append(header)
            lines.append("  " + "-" * 70)
            for model, m in sorted(snap["by_model"].items()):
                lines.append(
                    f"  {model:<30} {m['calls']:>8} {m['input_tokens']:>10} "
                    f"{m['output_tokens']:>10} {m['avg_latency_ms']:>10}"
                )
        else:
            lines.append("  (no model calls recorded)")
        lines.append("=" * 72)
        return "\n".join(lines)
