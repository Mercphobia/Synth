"""Developer-experience helpers for the CLI (spec 6.12: doctor, init, config).

Pure stdlib. Each function returns an exit code so cli.py stays thin.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

from synth.config import ConfigError, config_path, load_config
from synth.constants import CONFIG_DIR

# AGENTS.md template written by `synth init` (spec 6.12: /init).
AGENTS_TEMPLATE = """# AGENTS.md — Synth project guide

## Build & test
- venv: `python -m venv .venv && .venv/bin/pip install -e .`
- tests: `.venv/bin/python -m pytest tests/ -q`
- coverage gate: `pytest --cov=src/synth` must stay >= 80%

## Conventions
- Python 3.12+, full type hints, docstrings on public functions.
- No hardcoded limits — add named constants with a rationale comment.
- Tools return `ToolResult(text, is_error)` and never raise (boundary rule).
- SQL is always parameterized; file paths go through `safe_path()`.

## Commit style
- Conventional commits: `feat:` / `fix:` / `test:` / `docs:` / `refactor:`.
"""


def run_doctor(console) -> int:
    """Check the environment and report what is broken. Returns exit code.

    Checks: python version, config validity, API key, session db, git,
    ollama, disk space. FAIL -> exit 1; warnings alone -> exit 0.
    """
    import sys

    failures = 0

    def report(ok: bool, label: str, detail: str) -> None:
        nonlocal failures
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            failures += 1
        console.print(f"  {mark:<12} {label}: {detail}")

    console.print("[bold]Synth doctor[/bold]")

    # Python >= 3.12 (spec 9.1)
    py_ok = sys.version_info >= (3, 12)
    report(py_ok, "python", sys.version.split()[0] if py_ok else
           f"{sys.version.split()[0]} (need >= 3.12; fix: use a newer interpreter)")

    # Config parses
    try:
        config = load_config()
        report(True, "config", str(config_path()))
    except ConfigError as exc:
        report(False, "config", f"{exc} (fix: edit {config_path()})")
        config = None

    # API key resolves for the configured model
    if config is not None:
        try:
            config.resolve_api_key()
            report(True, "api key", f"set for model {config.model}")
        except ConfigError as exc:
            report(False, "api key", f"{exc} (fix: export the env var)")
    else:
        report(False, "api key", "skipped (config invalid)")

    # Session db directory writable
    if config is not None:
        db = Path(os.path.expanduser(config.session.db_path))
        try:
            db.parent.mkdir(parents=True, exist_ok=True)
            probe = db.parent / ".doctor-probe"
            probe.write_text("x")
            probe.unlink()
            report(True, "session db", str(db))
        except OSError as exc:
            report(False, "session db", f"{db}: {exc} (fix: check dir permissions)")

    # git on PATH (needed for git integration features)
    git = shutil.which("git")
    report(git is not None, "git", git or "not found (fix: install git)")

    # ollama reachable (optional)
    try:
        from synth.local import OllamaClient

        available = OllamaClient().is_available()
    except Exception:  # noqa: BLE001 — optional dependency path
        available = False
    if available:
        report(True, "ollama", "reachable at localhost:11434")
    else:
        console.print("  [yellow]WARN[/yellow] ollama: not running (optional; "
                      "start with `ollama serve` for local models)")

    # Disk space under ~/.synth
    home_cfg = Path(CONFIG_DIR)
    try:
        target = home_cfg if home_cfg.is_absolute() else Path.home()
        usage = shutil.disk_usage(target)
        free_gb = usage.free / (1024**3)
        ok = free_gb > 1.0
        report(ok, "disk", f"{free_gb:.1f} GB free" + ("" if ok else " (< 1 GB; fix: free space)"))
    except OSError as exc:
        report(False, "disk", str(exc))

    if failures:
        console.print(f"[red]{failures} problem(s) found.[/red]")
        return 1
    console.print("[green]All essential checks passed.[/green]")
    return 0


def run_init(console, force: bool = False) -> int:
    """Write an AGENTS.md project guide to the current directory (spec 6.12)."""
    target = Path.cwd() / "AGENTS.md"
    if target.exists() and not force:
        console.print(f"[yellow]{target} exists.[/yellow] Re-run with --force to overwrite.")
        return 0
    try:
        target.write_text(AGENTS_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        console.print(f"[red]Error:[/red] cannot write {target}: {exc}")
        return 1
    console.print(f"[green]✓[/green] wrote {target}")
    return 0


def _split_key(key: str) -> tuple[str | None, str]:
    """Split a dotted config key into (section, option)."""
    if "." in key:
        section, option = key.rsplit(".", 1)
        return section, option
    return None, key


def run_config(console, action: str, key: str | None = None,
               value: str | None = None) -> int:
    """Handle `synth config path|get|set` (spec 6.12 config editing)."""
    path = config_path()

    if action == "path":
        console.print(str(path))
        return 0

    # get/set: bootstrap defaults on first use so `config set` never dead-ends.
    if not path.exists():
        try:
            load_config()
        except ConfigError as exc:
            console.print(f"[red]Error:[/red] cannot create config {path}: {exc}")
            return 3

    raw_text = path.read_text(encoding="utf-8")

    if action == "get":
        if not key:
            # markup=False: config values legitimately contain [sections] which
            # rich would otherwise swallow as style tags.
            console.print(raw_text.rstrip(), markup=False)
            return 0
        try:
            parsed = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as exc:
            console.print(f"[red]Error:[/red] invalid config TOML: {exc}")
            return 3
        section, option = _split_key(key)
        node = parsed.get(section, {}) if section else parsed
        if not isinstance(node, dict) or option not in node:
            console.print(f"[red]Error:[/red] key '{key}' not found")
            return 2
        console.print(repr(node[option]))
        return 0

    if action == "set":
        if not key or value is None:
            console.print("[red]Error:[/red] usage: synth config set <key> <value>")
            return 2
        section, option = _split_key(key)
        new_text = _set_toml_option(raw_text, section, option, value)
        # Validate before replacing the file — never write a broken config.
        try:
            tomllib.loads(new_text)
        except tomllib.TOMLDecodeError as exc:
            console.print(f"[red]Error:[/red] resulting config would be invalid: {exc}")
            return 3
        tmp = path.with_suffix(".toml.tmp")
        try:
            tmp.write_text(new_text, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            console.print(f"[red]Error:[/red] cannot write {path}: {exc}")
            return 3
        console.print(f"[green]✓[/green] {key} = {value}")
        return 0

    console.print(f"[red]Error:[/red] unknown config action: {action}")
    return 2


def _set_toml_option(text: str, section: str | None, option: str, value: str) -> str:
    """Rewrite one option in TOML text, section-aware, preserving comments.

    Replaces the option line inside the target [section] (or top-level when
    section is None). Appends it under the section when missing; appends a new
    section header + option when the section itself is absent.
    """
    def _fmt(v: str) -> str:
        if v.lower() in {"true", "false"}:
            return v.lower()
        # Already a quoted TOML string — keep it verbatim.
        if len(v) >= 2 and v[0] == v[-1] and v[0] in {"\"", "'"}:
            return v
        try:
            return str(int(v))
        except ValueError:
            pass
        try:
            return str(float(v))
        except ValueError:
            pass
        return repr(v)

    new_line = f"{option} = {_fmt(value)}"
    lines = text.splitlines()

    def _header_of(line: str) -> str | None:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped[1:-1].strip()
        return None

    # Pass 1: replace an existing option line inside the wanted section.
    current: str | None = None
    replaced = False
    out: list[str] = []
    for line in lines:
        header = _header_of(line)
        if header is not None:
            current = header
        elif current == section:
            stripped = line.strip()
            if stripped == option or stripped.startswith(f"{option} ")\
                    or stripped.startswith(f"{option}="):
                out.append(new_line)
                replaced = True
                continue
        out.append(line)

    if replaced:
        return "\n".join(out) + "\n"

    # Pass 2: append under the section (or create it).
    if section is not None:
        section_exists = any(_header_of(l) == section for l in out)
        if section_exists:
            last = -1
            current = None
            for i, line in enumerate(out):
                header = _header_of(line)
                if header is not None:
                    current = header
                if current == section:
                    last = i
            out.insert(last + 1, new_line)
        else:
            out.extend([f"[{section}]", new_line])
        return "\n".join(out) + "\n"

    # Top-level option: must sit before the first [section] header.
    for i, line in enumerate(out):
        if _header_of(line) is not None:
            out.insert(i, new_line)
            break
    else:
        out.append(new_line)
    return "\n".join(out) + "\n"
