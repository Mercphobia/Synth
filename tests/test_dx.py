"""Tests for the DX helpers: doctor, init, config get/set (spec 6.12)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rich.console import Console

from synth.config import config_path
from synth.dx import _set_toml_option, run_config, run_init


@pytest.fixture
def console() -> Console:
    return Console(record=True, width=200)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME (and therefore ~/.synth) at a temp dir."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- _set_toml_option ------------------------------------------------------

BASE_TOML = (
    '[default]\nmodel = "a"\nmax_iterations = 20\nstream = true\n\n'
    '[session]\ndb_path = "x"\n'
)


def test_set_replaces_existing_option():
    out = _set_toml_option(BASE_TOML, "default", "model", '"gpt-4o"')
    assert tomllib.loads(out)["default"]["model"] == "gpt-4o"


def test_set_preserves_other_options_and_comments():
    toml = '# keep me\n[default]\nmodel = "a"\n'
    out = _set_toml_option(toml, "default", "max_iterations", "50")
    assert "# keep me" in out
    parsed = tomllib.loads(out)
    assert parsed["default"]["model"] == "a"
    assert parsed["default"]["max_iterations"] == 50


def test_set_int_value_unquoted():
    out = _set_toml_option(BASE_TOML, "default", "max_iterations", "50")
    assert tomllib.loads(out)["default"]["max_iterations"] == 50


def test_set_bool_value():
    out = _set_toml_option(BASE_TOML, "session", "checkpoints", "true")
    assert tomllib.loads(out)["session"]["checkpoints"] is True


def test_set_new_option_in_existing_section():
    out = _set_toml_option(BASE_TOML, "session", "new_opt", '"v"')
    parsed = tomllib.loads(out)
    assert parsed["session"]["new_opt"] == "v"
    assert parsed["session"]["db_path"] == "x"


def test_set_creates_missing_section():
    out = _set_toml_option(BASE_TOML, "git", "auto_commit", "false")
    parsed = tomllib.loads(out)
    assert parsed["git"]["auto_commit"] is False
    assert parsed["default"]["model"] == "a"  # untouched


def test_set_top_level_before_first_section():
    out = _set_toml_option("x = 1\n[sec]\ny = 2\n", None, "x", "5")
    parsed = tomllib.loads(out)
    assert parsed["x"] == 5 and parsed["sec"]["y"] == 2


def test_set_string_value_is_quoted():
    out = _set_toml_option(BASE_TOML, "default", "model", "gpt-4o")
    assert tomllib.loads(out)["default"]["model"] == "gpt-4o"


# --- run_config ------------------------------------------------------------

def test_config_path_bootstraps_and_prints(console):
    rc = run_config(console, "path")
    assert rc == 0
    assert "config.toml" in console.export_text()


def test_config_get_after_set(console):
    assert run_config(console, "set", "default.model", "gpt-4o") == 0
    console.record = True
    assert run_config(console, "get", "default.model") == 0
    assert "gpt-4o" in console.export_text()


def test_config_get_missing_key_returns_2(console):
    run_config(console, "path")  # bootstrap
    assert run_config(console, "get", "default.nonexistent") == 2


def test_config_get_without_key_prints_file(console):
    run_config(console, "path")
    assert run_config(console, "get") == 0
    assert "[default]" in console.export_text()


def test_config_set_requires_value(console):
    run_config(console, "path")
    assert run_config(console, "set", "default.model") == 2


def test_config_set_rejects_invalid_toml_result(console, monkeypatch):
    """A value that breaks TOML parsing must not be written to disk."""
    # First valid set bootstraps the config file.
    assert run_config(console, "set", "default.model", "x") == 0
    before = config_path().read_text()
    # Force the validity check to fail.
    monkeypatch.setattr(
        "synth.dx.tomllib.loads",
        lambda _s: (_ for _ in ()).throw(tomllib.TOMLDecodeError(msg="boom")),
    )
    rc = run_config(console, "set", "default.model", "y")
    assert rc == 3
    assert config_path().read_text() == before


# --- run_init --------------------------------------------------------------

def test_init_writes_agents_md(console):
    assert run_init(console) == 0
    agents = Path.cwd() / "AGENTS.md"
    assert agents.exists()
    assert "pytest" in agents.read_text()


def test_init_refuses_overwrite_without_force(console):
    run_init(console)
    (Path.cwd() / "AGENTS.md").write_text("custom")
    assert run_init(console) == 0
    assert (Path.cwd() / "AGENTS.md").read_text() == "custom"


def test_init_force_overwrites(console):
    run_init(console)
    (Path.cwd() / "AGENTS.md").write_text("custom")
    assert run_init(console, force=True) == 0
    assert "pytest" in (Path.cwd() / "AGENTS.md").read_text()


# --- run_doctor ------------------------------------------------------------

def test_doctor_reports_python_and_config(console):
    from synth.dx import run_doctor

    rc = run_doctor(console)
    text = console.export_text()
    assert "python" in text and "config" in text
    # rc depends on env (API key may be missing); just assert valid code.
    assert rc in (0, 1)


def test_doctor_flags_invalid_config(console, monkeypatch):
    from synth.dx import run_doctor

    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text("not [ valid toml")
    assert run_doctor(console) == 1
    assert "FAIL" in console.export_text()
