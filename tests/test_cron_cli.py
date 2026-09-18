"""Tests for cron CLI integration (cron_cli.py) — was 15% coverage."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest
from rich.console import Console

from synth.cli import main
from synth.cron_cli import (
    handle_cron_command,
    integrate_cron_cli,
    setup_cron_parser,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME at tmp so the cron DB is isolated per test.

    DEFAULT_CRON_DB_PATH is resolved at import time, so patching HOME alone
    is not enough — the module constant must be redirected too.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    # Redirect the cron DB into the per-test tmp dir.
    import synth.cron as cron_mod
    monkeypatch.setattr(cron_mod, "DEFAULT_CRON_DB_PATH", tmp_path / ".synth" / "cron.db")
    return tmp_path


def _ns(**kwargs) -> argparse.Namespace:
    """Build a cron args namespace with sane defaults."""
    defaults = {
        "cron_command": "list",
        "expression": "*/5 * * * *",
        "command": "echo hi",
        "job_id": 1,
        "show_disabled": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# --- setup_cron_parser -------------------------------------------------------

def test_setup_cron_parser_registers_all_six_subcommands():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    choices = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            choices.update(action.choices)
    assert "cron" in choices
    cron_parser = choices["cron"]
    cron_choices = {}
    for action in cron_parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            cron_choices.update(action.choices)
    assert set(cron_choices) == {"add", "remove", "list", "enable", "disable", "run"}


def test_setup_cron_parser_add_takes_expression_and_command():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    args = parser.parse_args(["cron", "add", "*/5 * * * *", "echo hi"])
    assert args.expression == "*/5 * * * *"
    assert args.command == "echo hi"


def test_setup_cron_parser_remove_takes_int_job_id():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    args = parser.parse_args(["cron", "remove", "7"])
    assert args.job_id == 7


def test_setup_cron_parser_list_has_show_disabled_flag():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    args = parser.parse_args(["cron", "list", "--show-disabled"])
    assert args.show_disabled is True


def test_setup_cron_parser_list_defaults_show_disabled_false():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    args = parser.parse_args(["cron", "list"])
    assert args.show_disabled is False


def test_setup_cron_parser_enable_disable_take_job_id():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    assert parser.parse_args(["cron", "enable", "3"]).job_id == 3
    assert parser.parse_args(["cron", "disable", "4"]).job_id == 4


def test_setup_cron_parser_run_has_no_extra_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup_cron_parser(sub)
    args = parser.parse_args(["cron", "run"])
    assert args.cron_command == "run"


# --- handle_cron_command dispatch --------------------------------------------

def test_handle_add_success(capsys):
    rc = handle_cron_command(_ns(cron_command="add"))
    assert rc == 0
    assert "Added job" in capsys.readouterr().out


def test_handle_add_invalid_expression(capsys):
    rc = handle_cron_command(
        _ns(cron_command="add", expression="not a cron", command="x")
    )
    assert rc == 1
    assert "Error" in capsys.readouterr().out


def test_handle_list_empty(capsys):
    rc = handle_cron_command(_ns(cron_command="list"))
    assert rc == 0
    assert "No cron jobs" in capsys.readouterr().out


def test_handle_list_with_job(capsys):
    handle_cron_command(_ns(cron_command="add"))
    rc = handle_cron_command(_ns(cron_command="list"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "*/5 * * * *" in out


def test_handle_list_disabled_filtered(capsys):
    handle_cron_command(_ns(cron_command="add"))
    handle_cron_command(_ns(cron_command="disable", job_id=1))
    capsys.readouterr()
    # Default (no --show-disabled) hides disabled jobs.
    handle_cron_command(_ns(cron_command="list"))
    out = capsys.readouterr().out
    assert "No enabled cron jobs" in out


def test_handle_list_show_disabled_flag(capsys):
    handle_cron_command(_ns(cron_command="add"))
    handle_cron_command(_ns(cron_command="disable", job_id=1))
    capsys.readouterr()
    handle_cron_command(_ns(cron_command="list", show_disabled=True))
    out = capsys.readouterr().out
    assert "*/5 * * * *" in out


def test_handle_list_truncates_long_command(capsys):
    long_cmd = "echo " + "x" * 80
    handle_cron_command(_ns(cron_command="add", command=long_cmd))
    capsys.readouterr()
    handle_cron_command(_ns(cron_command="list"))
    out = capsys.readouterr().out
    assert "…" in out
    assert "x" * 80 not in out


def test_handle_enable_success(capsys):
    handle_cron_command(_ns(cron_command="add"))
    handle_cron_command(_ns(cron_command="disable", job_id=1))
    capsys.readouterr()
    rc = handle_cron_command(_ns(cron_command="enable", job_id=1))
    assert rc == 0
    assert "Enabled job 1" in capsys.readouterr().out


def test_handle_disable_success(capsys):
    handle_cron_command(_ns(cron_command="add"))
    capsys.readouterr()
    rc = handle_cron_command(_ns(cron_command="disable", job_id=1))
    assert rc == 0
    assert "Disabled job 1" in capsys.readouterr().out


def test_handle_remove_success(capsys):
    handle_cron_command(_ns(cron_command="add"))
    capsys.readouterr()
    rc = handle_cron_command(_ns(cron_command="remove", job_id=1))
    assert rc == 0
    assert "Removed job 1" in capsys.readouterr().out


def test_handle_remove_missing_job(capsys):
    rc = handle_cron_command(_ns(cron_command="remove", job_id=999))
    assert rc == 1
    assert "Error" in capsys.readouterr().out


def test_handle_enable_missing_job(capsys):
    rc = handle_cron_command(_ns(cron_command="enable", job_id=999))
    assert rc == 1
    assert "Error" in capsys.readouterr().out


def test_handle_disable_missing_job(capsys):
    rc = handle_cron_command(_ns(cron_command="disable", job_id=999))
    assert rc == 1
    assert "Error" in capsys.readouterr().out


def test_handle_run_succeeds(capsys):
    rc = handle_cron_command(_ns(cron_command="run"))
    assert rc == 0
    assert "Ran pending jobs" in capsys.readouterr().out


def test_handle_unknown_command_returns_zero():
    # No matching branch falls through to the final return 0.
    assert handle_cron_command(_ns(cron_command="bogus")) == 0


# --- integrate_cron_cli ------------------------------------------------------

def test_integrate_cron_cli_with_subparsers():
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="cmd")
    integrate_cron_cli(parser)  # must not raise
    args = parser.parse_args(["cron", "list"])
    assert args.cron_command == "list"


def test_integrate_cron_cli_without_subparsers_is_noop():
    parser = argparse.ArgumentParser()
    integrate_cron_cli(parser)  # no subparsers -> silently does nothing
    assert parser.parse_args([]) is not None


# --- end-to-end through the real CLI dispatcher ------------------------------

def test_cli_cron_list_end_to_end(capsys):
    assert main(["cron", "list"]) == 0
    assert "No cron jobs" in capsys.readouterr().out


def test_cli_cron_add_then_list_end_to_end(capsys):
    assert main(["cron", "add", "0 9 * * 1", "echo morning"]) == 0
    capsys.readouterr()
    assert main(["cron", "list"]) == 0
    assert "echo morning" in capsys.readouterr().out


def test_cli_cron_run_end_to_end(capsys):
    assert main(["cron", "run"]) == 0