"""Shared pytest fixtures for the Synth test suite.

Several modules resolve their default state paths at *import* time
(``Path.home() / CONFIG_DIR / ...``). That means patching ``HOME`` inside a
test is not enough to isolate them — the module-level constant has already
been frozen. This conftest redirects those constants for every test so that
no test can ever touch the developer's real ``~/.synth`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_synth_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every import-time default state path into ``tmp_path``."""
    state = tmp_path / ".synth"
    # sqlite3 requires the parent directory to exist before opening a DB.
    # Tests that create this dir themselves must use exist_ok=True.
    state.mkdir(parents=True, exist_ok=True)

    db = state / "sessions.db"
    ckpt = state / "checkpoints.db"
    cron_db = state / "cron.db"

    # session.py / session_plus.py resolve DEFAULT_SESSION_DIR at import time.
    import synth.session as session_mod
    import synth.session_plus as session_plus_mod
    import synth.session_extras as session_extras_mod
    import synth.checkpoints as checkpoints_mod
    import synth.cron as cron_mod
    import synth.daemon as daemon_mod
    import synth.memory as memory_mod
    import synth.recipes as recipes_mod

    monkeypatch.setattr(session_mod, "DEFAULT_SESSION_DIR", state, raising=False)
    monkeypatch.setattr(session_plus_mod, "DEFAULT_SESSION_DIR", state, raising=False)
    if hasattr(session_extras_mod, "DEFAULT_SESSION_DIR"):
        monkeypatch.setattr(
            session_extras_mod, "DEFAULT_SESSION_DIR", state, raising=False
        )
    monkeypatch.setattr(
        checkpoints_mod, "DEFAULT_CHECKPOINT_DB_PATH", ckpt, raising=False
    )
    monkeypatch.setattr(cron_mod, "DEFAULT_CRON_DB_PATH", cron_db, raising=False)

    return state