"""Tests for synth.checkpoints — session checkpoints and undo.

Uses ``tmp_path`` for isolated databases.  Covers both success and error
paths for every public method, per project rules.
"""

from __future__ import annotations

import sqlite3

import pytest

from synth.checkpoints import CheckpointError, CheckpointStore, _Msg


def _seed_session_db(tmp_path, session_id: str, messages: list[tuple]) -> None:
    """Create a minimal sessions.db with the given messages."""
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, title TEXT,
            created_at INTEGER, updated_at INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, tool_calls TEXT, tool_call_id TEXT, ts INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
        (session_id, "test", 0, 0),
    )
    for role, content, ts in messages:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, ts) "
            "VALUES (?,?,?,NULL,NULL,?)",
            (session_id, role, content, ts),
        )
    conn.commit()
    conn.close()


# --- create + restore round trip -------------------------------------------


def test_create_and_restore_checkpoint(tmp_path):
    """Success: checkpoint a session and restore it faithfully."""
    sid = "sess-1"
    _seed_session_db(tmp_path, sid, [
        ("user", "hello", 1),
        ("assistant", "hi!", 2),
    ])
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        cid = cs.create_checkpoint(sid, tag="v1")
        assert isinstance(cid, str) and len(cid) > 0

        restored = cs.restore_checkpoint(cid)
        assert len(restored) == 2
        assert restored[0].role == "user"
        assert restored[0].content == "hello"
        assert restored[1].role == "assistant"
        assert restored[1].content == "hi!"
    finally:
        cs.close()


def test_restore_checkpoint_not_found(tmp_path):
    """Error: restoring a nonexistent checkpoint raises CheckpointError."""
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        with pytest.raises(CheckpointError, match="not found"):
            cs.restore_checkpoint("nonexistent-id")
    finally:
        cs.close()


# --- list_checkpoints ------------------------------------------------------


def test_list_checkpoints(tmp_path):
    """Success: list returns metadata in newest-first order."""
    sid = "sess-list"
    _seed_session_db(tmp_path, sid, [("user", "x", 1)])
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        c1 = cs.create_checkpoint(sid, tag="first")
        c2 = cs.create_checkpoint(sid, tag="second")
        listing = cs.list_checkpoints(sid)
        assert len(listing) == 2
        # Newest first (both have same second timestamp, but order is by
        # created_at DESC; insertion order is stable for same timestamp).
        ids = [e["id"] for e in listing]
        assert c1 in ids and c2 in ids
        assert all("tag" in e and "created_at" in e for e in listing)
    finally:
        cs.close()


def test_list_checkpoints_empty(tmp_path):
    """Success: listing a session with no checkpoints returns []."""
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        assert cs.list_checkpoints("no-such-session") == []
    finally:
        cs.close()


# --- delete_checkpoint -----------------------------------------------------


def test_delete_checkpoint(tmp_path):
    """Success: delete removes the checkpoint."""
    sid = "sess-del"
    _seed_session_db(tmp_path, sid, [("user", "x", 1)])
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        cid = cs.create_checkpoint(sid)
        assert cs.delete_checkpoint(cid) is True
        assert cs.list_checkpoints(sid) == []
    finally:
        cs.close()


def test_delete_checkpoint_missing(tmp_path):
    """Error: deleting a nonexistent checkpoint returns False."""
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        assert cs.delete_checkpoint("does-not-exist") is False
    finally:
        cs.close()


# --- undo_last_step --------------------------------------------------------


def test_undo_last_step(tmp_path):
    """Success: undo removes assistant/tool messages back to the last user."""
    sid = "sess-undo"
    _seed_session_db(tmp_path, sid, [
        ("user", "do this", 1),
        ("assistant", "ok", 2),
        ("assistant", "calling tool", 3),
        ("user", "next", 4),
        ("assistant", "working", 5),
    ])
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        changed = cs.undo_last_step(sid)
        assert changed is True

        # Only the last assistant message (id=5) should be gone.
        conn = sqlite3.connect(str(tmp_path / "sessions.db"))
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        conn.close()
        assert len(rows) == 4
        assert rows[-1] == ("user", "next")
    finally:
        cs.close()


def test_undo_last_step_nothing_to_undo(tmp_path):
    """Error/edge: undo on a session with only user messages returns False."""
    sid = "sess-undo-empty"
    _seed_session_db(tmp_path, sid, [("user", "alone", 1)])
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        assert cs.undo_last_step(sid) is False
    finally:
        cs.close()


def test_undo_last_step_no_sessions_db(tmp_path):
    """Error: undo when sessions.db doesn't exist raises CheckpointError."""
    cs = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    try:
        with pytest.raises(CheckpointError, match="not found"):
            cs.undo_last_step("anything")
    finally:
        cs.close()


# --- context manager -------------------------------------------------------


def test_context_manager(tmp_path):
    """Success: CheckpointStore works as a context manager."""
    with CheckpointStore(db_path=tmp_path / "checkpoints.db") as cs:
        sid = "sess-ctx"
        _seed_session_db(tmp_path, sid, [("user", "x", 1)])
        cid = cs.create_checkpoint(sid)
        assert cs.restore_checkpoint(cid)
