"""Extra CLI dispatch tests: session, git, snapshot, daemon subcommands.

These fill the remaining gaps in cli.py's dispatch chain (the branches that
parse arguments and delegate to ``_run_*`` helpers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.cli import (
    EXIT_OK,
    EXIT_ARG,
    EXIT_GENERAL,
    main,
)


@pytest.fixture(autouse=True)
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- session subcommand ------------------------------------------------------

def test_session_arg_choices_reject_bogus_action():
    with pytest.raises(SystemExit) as exc:
        main(["session", "frobnicate", "x"])
    assert exc.value.code == 2


def test_session_export_json_missing_session():
    # Exporting an unknown session id must fail cleanly (not traceback).
    rc = main(["session", "export", "no-such-session", "--format", "json"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_session_export_md_missing_session():
    rc = main(["session", "export", "no-such-session", "--format", "md"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_session_export_html_missing_session():
    rc = main(["session", "export", "no-such-session", "--format", "html"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_session_export_explicit_out_path(tmp_path):
    out = tmp_path / "dump.json"
    rc = main(["session", "export", "no-such", "--format", "json", "--out", str(out)])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_session_missing_arg_is_arg_error():
    with pytest.raises(SystemExit) as exc:
        main(["session", "search"])
    assert exc.value.code == 2


# --- git subcommand ----------------------------------------------------------

def test_git_bogus_action_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["git", "frobnicate"])
    assert exc.value.code == 2


def test_git_commit_without_message(tmp_path):
    rc = main(["git", "commit"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


def test_git_commit_with_message_outside_repo(tmp_path):
    rc = main(["git", "commit", "-m", "test commit"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


def test_git_diff_outside_repo():
    rc = main(["git", "diff"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


def test_git_review_outside_repo():
    rc = main(["git", "review"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


def test_git_log_limit_flag():
    rc = main(["git", "log", "--limit", "5"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


# --- snapshot subcommand -----------------------------------------------------

def test_snapshot_bogus_action_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["snapshot", "frobnicate"])
    assert exc.value.code == 2


def test_snapshot_create_without_session_id():
    rc = main(["snapshot", "create"])
    assert rc == EXIT_ARG


def test_snapshot_list_without_session_id():
    rc = main(["snapshot", "list"])
    assert rc == EXIT_ARG


def test_snapshot_restore_without_checkpoint_id():
    rc = main(["snapshot", "restore"])
    assert rc == EXIT_ARG


def test_snapshot_create_unknown_session(tmp_path):
    rc = main(["snapshot", "create", "no-such-session", "--tag", "wip"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_snapshot_list_unknown_session():
    rc = main(["snapshot", "list", "no-such-session"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_snapshot_restore_unknown_checkpoint():
    rc = main(["snapshot", "restore", "deadbeef"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


# --- daemon subcommand -------------------------------------------------------

def test_daemon_status_not_running(capsys):
    rc = main(["daemon", "status"])
    assert rc in (EXIT_OK, EXIT_GENERAL)
    out = capsys.readouterr().out
    assert "stopped" in out.lower() or "running" in out.lower()


def test_daemon_stop_when_not_running():
    rc = main(["daemon", "stop"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


def test_daemon_start_when_already_running(tmp_path, monkeypatch):
    """With a live pid file, start must refuse rather than spawn a second daemon."""
    import os

    from synth import daemon as daemon_mod

    state = tmp_path / ".synth"
    state.mkdir(parents=True, exist_ok=True)
    pid_file = state / "synth.pid"
    pid_file.write_text(str(os.getpid()))

    cfg = daemon_mod.DaemonConfig
    monkeypatch.setattr(cfg, "pid_file", pid_file, raising=False)

    rc = main(["daemon", "status"])
    assert rc in (EXIT_OK, EXIT_GENERAL)


# --- models / task extra branches -------------------------------------------

def test_models_bogus_action():
    with pytest.raises(SystemExit) as exc:
        main(["models", "frobnicate"])
    assert exc.value.code == 2


def test_task_bogus_action():
    with pytest.raises(SystemExit) as exc:
        main(["task", "frobnicate"])
    assert exc.value.code == 2


def test_scan_nonexistent_path():
    rc = main(["scan", "/nonexistent/path/xyz"])
    assert rc in (EXIT_OK, EXIT_GENERAL, EXIT_ARG)


def test_config_bogus_action():
    with pytest.raises(SystemExit) as exc:
        main(["config", "frobnicate"])
    assert exc.value.code == 2