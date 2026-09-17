"""Tests for session_extras.py — time travel, watch mode, fuzzy picker.

Spec #73 (time travel), #80 (watch mode), #127 (session picker).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from synth.checkpoints import CheckpointStore
from synth.session import Message, SessionStore
from synth.session_extras import (
    SessionPicker,
    TimeTravel,
    WatchMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stores(tmp_path: Path) -> tuple[CheckpointStore, SessionStore, str]:
    """Create a CheckpointStore + SessionStore sharing one config dir."""
    cfg = tmp_path / ".synth"
    cfg.mkdir()
    cp_store = CheckpointStore(cfg / "checkpoints.db")
    sess_store = SessionStore(cfg / "sessions.db")
    sess_id = sess_store.create_session(title="test session")
    return cp_store, sess_store, sess_id


def _seed_messages(store: SessionStore, sid: str) -> None:
    """Append a fixed conversation to the session."""
    for msg in (
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="Do a thing"),
        Message(role="assistant", content="Done"),
    ):
        store.append_message(sid, msg)


# ---------------------------------------------------------------------------
# TimeTravel.replay — callback fired per message
# ---------------------------------------------------------------------------


def test_replay_calls_callback_per_message(stores: tuple) -> None:
    """replay() should invoke the callback once per message, in order."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid, tag="snap")

    seen: list[tuple[str, str | None]] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.replay(sid, cp_id, lambda role, content: seen.append((role, content)))

    assert len(seen) == 4
    assert seen[0] == ("user", "Hello")
    assert seen[1] == ("assistant", "Hi there")
    assert seen[2] == ("user", "Do a thing")
    assert seen[3] == ("assistant", "Done")


def test_replay_callback_signature(stores: tuple) -> None:
    """Callback receives exactly two positional args: role, content."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid)

    captured: list[Any] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.replay(sid, cp_id, lambda *args: captured.append(args))
    assert all(len(c) == 2 for c in captured)


def test_replay_empty_checkpoint(stores: tuple) -> None:
    """Replaying a checkpoint taken on an empty session calls callback 0 times."""
    cp_store, sess_store, sid = stores
    cp_id = cp_store.create_checkpoint(sid, tag="empty")
    seen: list[Any] = []
    TimeTravel(cp_store, sess_store).replay(sid, cp_id, lambda r, c: seen.append(r))
    assert seen == []


# ---------------------------------------------------------------------------
# TimeTravel.pause_at
# ---------------------------------------------------------------------------


def test_pause_at_stops_early(stores: tuple) -> None:
    """pause_at(N) should stop replay after the Nth (0-indexed) message."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid)

    seen: list[tuple[str, str | None]] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.pause_at(1)
    tt.replay(sid, cp_id, lambda r, c: seen.append((r, c)))
    assert len(seen) == 2
    assert seen[0][0] == "user"
    assert seen[1][0] == "assistant"


def test_pause_at_zero(stores: tuple) -> None:
    """pause_at(0) replays only the first message."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid)

    seen: list[str] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.pause_at(0)
    tt.replay(sid, cp_id, lambda r, c: seen.append(r))
    assert seen == ["user"]


def test_pause_at_clears_with_negative(stores: tuple) -> None:
    """A negative pause_at resets the cap so all messages replay."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid)

    seen: list[str] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.pause_at(1)
    tt.pause_at(-1)
    tt.replay(sid, cp_id, lambda r, c: seen.append(r))
    assert len(seen) == 4


def test_pause_at_last_index(stores: tuple) -> None:
    """pause_at equal to last index replays everything."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_id = cp_store.create_checkpoint(sid)

    seen: list[str] = []
    tt = TimeTravel(cp_store, sess_store)
    tt.pause_at(3)
    tt.replay(sid, cp_id, lambda r, c: seen.append(r))
    assert len(seen) == 4


# ---------------------------------------------------------------------------
# TimeTravel.get_timeline
# ---------------------------------------------------------------------------


def test_timeline_returns_all_messages(stores: tuple) -> None:
    """get_timeline returns one entry per message in order."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    tt = TimeTravel(cp_store, sess_store)
    timeline = tt.get_timeline(sid)
    assert len(timeline) == 4
    assert [e["index"] for e in timeline] == [0, 1, 2, 3]


def test_timeline_roles_and_previews(stores: tuple) -> None:
    """Timeline entries carry role and a content_preview."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    tt = TimeTravel(cp_store, sess_store)
    timeline = tt.get_timeline(sid)
    assert timeline[0]["role"] == "user"
    assert timeline[0]["content_preview"] == "Hello"
    assert timeline[1]["role"] == "assistant"
    assert timeline[1]["content_preview"] == "Hi there"


def test_timeline_has_checkpoint_flag(stores: tuple) -> None:
    """The index where a checkpoint snapshot ends should be flagged True."""
    cp_store, sess_store, sid = stores
    _seed_messages(sess_store, sid)
    cp_store.create_checkpoint(sid, tag="snap")  # snapshot of 4 msgs
    tt = TimeTravel(cp_store, sess_store)
    timeline = tt.get_timeline(sid)
    # Snapshot of 4 messages → has_checkpoint True at index 3 (last).
    assert timeline[3]["has_checkpoint"] is True
    assert timeline[0]["has_checkpoint"] is False


def test_timeline_preview_truncation(stores: tuple) -> None:
    """Long content is truncated to TIMELINE_PREVIEW_CHARS."""
    from synth.constants import TIMELINE_PREVIEW_CHARS

    cp_store, sess_store, sid = stores
    long_text = "X" * (TIMELINE_PREVIEW_CHARS + 50)
    sess_store.append_message(sid, Message(role="user", content=long_text))
    tt = TimeTravel(cp_store, sess_store)
    timeline = tt.get_timeline(sid)
    assert len(timeline[0]["content_preview"]) == TIMELINE_PREVIEW_CHARS


def test_timeline_empty_session(stores: tuple) -> None:
    """An empty session yields an empty timeline."""
    cp_store, sess_store, sid = stores
    tt = TimeTravel(cp_store, sess_store)
    assert tt.get_timeline(sid) == []


# ---------------------------------------------------------------------------
# WatchMode
# ---------------------------------------------------------------------------


def test_watch_mode_detects_file_change(tmp_path: Path) -> None:
    """Modifying a file between polls fires the callback."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    seen: list[Path] = []
    w = WatchMode(lambda p: seen.append(p))
    # First poll: initial state, file is "new" → reported.
    w.poll_once(tmp_path)
    assert any(p == f for p in seen)
    seen.clear()
    # No change: second poll reports nothing.
    w.poll_once(tmp_path)
    assert seen == []
    # Mutate mtime + content.
    time.sleep(0.05)
    f.write_text("x = 2\n")
    os.utime(f, None)
    w.poll_once(tmp_path)
    assert any(p == f for p in seen)


def test_watch_mode_respects_pattern(tmp_path: Path) -> None:
    """Files not matching the pattern are ignored."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    seen: list[Path] = []
    w = WatchMode(lambda p: seen.append(p))
    w.poll_once(tmp_path, pattern="*.py")
    names = {p.name for p in seen}
    assert "a.py" in names
    assert "b.txt" not in names


def test_watch_mode_stop_exits_loop(tmp_path: Path) -> None:
    """stop() makes watch() return within a reasonable time."""
    import threading

    w = WatchMode(lambda p: None)
    (tmp_path / "a.py").write_text("x")
    # Start watch in a thread, stop it, ensure it terminates.
    t = threading.Thread(target=w.watch, args=(tmp_path,), kwargs={"interval": 0.2})
    t.start()
    time.sleep(0.1)
    w.stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "watch loop did not stop"


def test_watch_mode_nonexistent_dir(tmp_path: Path) -> None:
    """Watching a path that doesn't exist doesn't blow up."""
    seen: list[Path] = []
    w = WatchMode(lambda p: seen.append(p))
    w.poll_once(tmp_path / "nope")
    assert seen == []


# ---------------------------------------------------------------------------
# SessionPicker — scoring
# ---------------------------------------------------------------------------


def test_picker_exact_score() -> None:
    """Exact match (case-insensitive) scores 100 and sorts first."""
    sessions = [{"id": "alpha", "title": "first"}, {"id": "beta", "title": "second"}]
    res = SessionPicker.fuzzy_search(sessions, "alpha")
    assert res[0]["id"] == "alpha"


def test_picker_prefix_score() -> None:
    """Prefix match (but not exact) sorts above substring matches."""
    sessions = [
        {"id": "abcdef", "title": ""},
        {"id": "zzz", "title": "abcdef"},
    ]
    res = SessionPicker.fuzzy_search(sessions, "abc")
    # Both match, but id has prefix, title only has substring.
    assert res[0]["id"] == "abcdef"


def test_picker_substring_score() -> None:
    """Substring (non-prefix) match still returns the session."""
    sessions = [{"id": "xabcdefx", "title": ""}]
    res = SessionPicker.fuzzy_search(sessions, "abc")
    assert len(res) == 1
    assert res[0]["id"] == "xabcdefx"


def test_picker_word_boundary_score() -> None:
    """Needle matching start of a word (after separator) is a word-boundary hit."""
    sessions = [{"id": "foo_bar_baz", "title": ""}]
    res = SessionPicker.fuzzy_search(sessions, "baz")
    assert len(res) == 1
    assert res[0]["id"] == "foo_bar_baz"


def test_picker_no_match_excluded() -> None:
    """Sessions with zero score are not returned."""
    sessions = [{"id": "alpha", "title": "first"}, {"id": "zzz", "title": "zzz"}]
    res = SessionPicker.fuzzy_search(sessions, "nomatch")
    assert res == []


def test_picker_limit_enforced() -> None:
    """The limit parameter caps the result count."""
    sessions = [{"id": f"match{i}", "title": ""} for i in range(20)]
    res = SessionPicker.fuzzy_search(sessions, "match", limit=5)
    assert len(res) == 5


def test_picker_sorted_by_score_desc() -> None:
    """Results are sorted by score, highest first."""
    sessions = [
        {"id": "zz", "title": "zz"},  # exact
        {"id": "zzz-prefix", "title": ""},  # prefix
        {"id": "xxzzz", "title": ""},  # substring
    ]
    res = SessionPicker.fuzzy_search(sessions, "zz")
    assert [s["id"] for s in res] == ["zz", "zzz-prefix", "xxzzz"]


def test_picker_default_limit() -> None:
    """Default limit is FUZZY_LIMIT_DEFAULT (10)."""
    from synth.constants import FUZZY_LIMIT_DEFAULT

    sessions = [{"id": f"s{i}", "title": ""} for i in range(30)]
    res = SessionPicker.fuzzy_search(sessions, "s")
    assert len(res) == FUZZY_LIMIT_DEFAULT


def test_picker_empty_query() -> None:
    """Empty query returns no results."""
    sessions = [{"id": "alpha", "title": ""}]
    assert SessionPicker.fuzzy_search(sessions, "") == []


def test_picker_missing_fields_safe() -> None:
    """Sessions missing id/title fields don't crash the picker."""
    sessions = [{}, {"id": "alpha"}]
    res = SessionPicker.fuzzy_search(sessions, "alpha")
    assert len(res) == 1
    assert res[0]["id"] == "alpha"


# ---------------------------------------------------------------------------
# SessionPicker.pick_interactive
# ---------------------------------------------------------------------------


def test_pick_interactive_returns_best() -> None:
    """pick_interactive returns the single highest-scoring match."""
    sessions = [
        {"id": "alpha", "title": ""},
        {"id": "alpha-beta", "title": ""},
        {"id": "zzz", "title": ""},
    ]
    best = SessionPicker.pick_interactive(sessions, "alpha")
    assert best is not None
    assert best["id"] == "alpha"  # exact beats prefix


def test_pick_interactive_no_match_returns_none() -> None:
    """When nothing matches, pick_interactive returns None."""
    sessions = [{"id": "alpha", "title": ""}]
    assert SessionPicker.pick_interactive(sessions, "nomatch") is None


def test_pick_interactive_empty_sessions() -> None:
    """An empty session list yields None."""
    assert SessionPicker.pick_interactive([], "x") is None
