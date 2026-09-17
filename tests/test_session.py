"""Tests for synth.session — SQLite-backed session storage.

Every test opens its own SessionStore on a temp DB file. The real
~/.synth/sessions.db is never touched.
"""

from __future__ import annotations

import sqlite3

import pytest

from synth.constants import SESSION_LIST_LIMIT
from synth.session import Message, SessionError, SessionStore


@pytest.fixture
def store(tmp_path):
    """A SessionStore on a throwaway DB file."""
    s = SessionStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


def _make_messages(n: int, base_ts: int = 1000) -> list[Message]:
    return [Message(role="user", content=f"msg-{i}", ts=base_ts + i) for i in range(n)]


# --- create_session ---

def test_create_session_with_title_returns_uuid(store):
    sid = store.create_session(title="chat about cats")
    assert isinstance(sid, str)
    # uuid4 format
    assert len(sid) == 36 and sid.count("-") == 4
    row = store._conn.execute(
        "SELECT id, title FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["id"] == sid
    assert row["title"] == "chat about cats"


def test_create_session_without_title_stores_null(store):
    sid = store.create_session()
    row = store._conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["title"] is None


def test_create_session_two_ids_are_unique(store):
    a = store.create_session()
    b = store.create_session()
    assert a != b


# --- create + append + load round trip ---

def test_create_append_load_roundtrip_preserves_messages(store):
    sid = store.create_session(title="round trip")
    msgs = [
        Message(role="user", content="hello", ts=100),
        Message(role="assistant", content="hi there", ts=200),
        Message(role="user", content="bye", ts=300),
    ]
    for m in msgs:
        store.append_message(sid, m)

    loaded = store.load_session(sid)
    assert len(loaded) == 3
    assert [m.role for m in loaded] == ["user", "assistant", "user"]
    assert [m.content for m in loaded] == ["hello", "hi there", "bye"]
    assert [m.ts for m in loaded] == [100, 200, 300]


def test_create_append_load_roundtrip_preserves_all_fields(store):
    sid = store.create_session()
    original = Message(
        role="assistant",
        content=None,
        tool_calls='[{"id": "call-1", "name": "bash"}]',
        tool_call_id="call-1",
        ts=42,
    )
    store.append_message(sid, original)

    loaded = store.load_session(sid)
    assert loaded == [original]


def test_append_message_returns_increasing_row_ids(store):
    sid = store.create_session()
    ids = [store.append_message(sid, m) for m in _make_messages(3)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 3
    assert all(isinstance(i, int) for i in ids)


def test_append_message_uses_current_time_when_ts_is_zero(store):
    sid = store.create_session()
    rid = store.append_message(sid, Message(role="user", content="x"))
    row = store._conn.execute(
        "SELECT ts FROM messages WHERE id = ?", (rid,)
    ).fetchone()
    assert row["ts"] > 0


def test_append_message_to_nonexistent_session_raises_fk_error(store):
    bogus = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(SessionError):
        store.append_message(bogus, Message(role="user", content="nope"))
    # nothing was written
    assert store._conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"] == 0


# --- load_session ---

def test_load_session_empty_returns_empty_list(store):
    sid = store.create_session(title="empty")
    assert store.load_session(sid) == []


def test_load_session_nonexistent_returns_empty_list(store):
    # No session exists; loading its (absent) messages is [] not an error.
    assert store.load_session("does-not-exist") == []


def test_load_session_returns_oldest_first(store):
    sid = store.create_session()
    # Append out of chronological order; load must sort by ts.
    store.append_message(sid, Message(role="user", content="third", ts=300))
    store.append_message(sid, Message(role="user", content="first", ts=100))
    store.append_message(sid, Message(role="user", content="second", ts=200))

    loaded = store.load_session(sid)
    assert [m.content for m in loaded] == ["first", "second", "third"]


def test_load_session_ties_break_by_id(store):
    sid = store.create_session()
    # Same ts: insertion order (id ASC) must be preserved.
    for text in ("a", "b", "c"):
        store.append_message(sid, Message(role="user", content=text, ts=500))

    loaded = store.load_session(sid)
    assert [m.content for m in loaded] == ["a", "b", "c"]


# --- list_sessions ---

def test_list_sessions_orders_newest_first(store):
    # created_at/updated_at use wall-clock time; space sessions apart so the
    # ordering is deterministic rather than a coin flip on equal timestamps.
    first = store.create_session(title="old")
    _bump_updated(store, first, 1000)
    second = store.create_session(title="new")
    _bump_updated(store, second, 2000)

    rows = store.list_sessions()
    assert [r["title"] for r in rows] == ["new", "old"]


def test_list_sessions_default_limit_is_constants_value(store):
    for _ in range(SESSION_LIST_LIMIT + 5):
        _bump_updated(store, store.create_session(), 1000)

    rows = store.list_sessions()
    assert len(rows) == SESSION_LIST_LIMIT


def test_list_sessions_respects_explicit_limit(store):
    for _ in range(5):
        _bump_updated(store, store.create_session(), 1000)

    assert len(store.list_sessions(limit=3)) == 3
    assert len(store.list_sessions(limit=1)) == 1


def test_list_sessions_empty_db_returns_empty_list(store):
    assert store.list_sessions() == []


def test_list_sessions_invalid_limit_raises_value_error(store):
    with pytest.raises(ValueError):
        store.list_sessions(limit=0)
    with pytest.raises(ValueError):
        store.list_sessions(limit=-5)


# --- update_session_title ---

def test_update_session_title_success(store):
    sid = store.create_session(title="old title")
    store.update_session_title(sid, "new title")
    row = store._conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["title"] == "new title"


def test_update_session_title_updates_timestamp(store):
    sid = store.create_session(title="t")
    _bump_updated(store, sid, 1000)
    store.update_session_title(sid, "renamed")
    row = store._conn.execute(
        "SELECT updated_at FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["updated_at"] > 1000


def test_update_session_title_not_found_raises_session_error(store):
    bogus = "11111111-1111-1111-1111-111111111111"
    with pytest.raises(SessionError, match="not found"):
        store.update_session_title(bogus, "nope")


def test_update_session_title_not_found_leaves_other_sessions(store):
    keep = store.create_session(title="keep me")
    with pytest.raises(SessionError):
        store.update_session_title("nonexistent-id", "nope")
    row = store._conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (keep,)
    ).fetchone()
    assert row["title"] == "keep me"


# --- close / context manager ---

def test_context_manager_closes_database(tmp_path):
    db_path = tmp_path / "ctx.db"
    with SessionStore(db_path=db_path) as s:
        sid = s.create_session(title="ctx")
        s.append_message(sid, Message(role="user", content="hi", ts=1))
        assert s.load_session(sid)
    # Connection is closed: further use raises a SessionError (our boundary
    # wraps the underlying sqlite3 closed-database error).
    with pytest.raises(SessionError):
        s.list_sessions()


def test_context_manager_commits_on_exit(tmp_path):
    db_path = tmp_path / "ctx2.db"
    with SessionStore(db_path=db_path) as s:
        s.create_session(title="committed")
    # A fresh store on the same file must see the row.
    with SessionStore(db_path=db_path) as s2:
        rows = s2.list_sessions()
    assert [r["title"] for r in rows] == ["committed"]


def test_close_is_idempotent(store):
    store.close()
    store.close()  # second close is best-effort, must not raise


# --- helpers ---

def _bump_updated(store: SessionStore, session_id: str, ts: int) -> None:
    """Pin a session's updated_at so list_sessions ordering is deterministic."""
    store._conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?", (ts, session_id)
    )
    store._conn.commit()
