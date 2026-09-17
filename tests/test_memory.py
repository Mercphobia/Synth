"""Tests for synth.memory — L1 prompt memory and L2 lessons store.

Every test points base_dir at a temp directory. The real ~/.synth/memory/
is never touched.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from synth.constants import CONFIG_DIR
from synth.memory import (
    LESSON_PROMPT_LIMIT,
    MEMORY_DB_FILENAME,
    MEMORY_NOTES_FILENAME,
    NOTE_TIMESTAMP_FORMAT,
    USER_FACTS_FILENAME,
    Lesson,
    MemoryError,
    MemoryStore,
    build_memory_prompt,
)


@pytest.fixture
def store(tmp_path):
    """A MemoryStore on a throwaway directory."""
    s = MemoryStore(base_dir=tmp_path / "mem")
    yield s
    s.close()


# --- construction ---

def test_default_base_dir_is_under_config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with MemoryStore() as s:
        assert s.base_dir == tmp_path / CONFIG_DIR / "memory"
        assert s.memory_notes_path.name == MEMORY_NOTES_FILENAME
        assert s.user_facts_path.name == USER_FACTS_FILENAME
        assert s.db_path.name == MEMORY_DB_FILENAME


def test_base_dir_escaping_via_dotdot_raises_memory_error(tmp_path):
    with pytest.raises(MemoryError, match=r"\.\."):
        MemoryStore(base_dir=tmp_path / "mem" / ".." / "evil")


def test_base_dir_escaping_in_the_middle_raises_memory_error(tmp_path):
    with pytest.raises(MemoryError):
        MemoryStore(base_dir=Path("../escape"))


# --- L1: load_prompt_memory ---

def test_load_prompt_memory_empty_store_returns_empty_string(store):
    assert store.load_prompt_memory() == ""


def test_load_prompt_memory_reads_user_facts(store):
    store.save_user_fact("prefers tabs over spaces")
    assert "prefers tabs over spaces" in store.load_prompt_memory()


def test_load_prompt_memory_reads_notes_and_facts(store):
    store.save_memory_note("deploy failed on missing env var")
    store.save_user_fact("works on the Synth CLI")
    loaded = store.load_prompt_memory()
    assert "deploy failed on missing env var" in loaded
    assert "works on the Synth CLI" in loaded


def test_load_prompt_memory_unreadable_file_raises_memory_error(store):
    # A directory where the notes file should be cannot be read.
    store.memory_notes_path.mkdir()
    with pytest.raises(MemoryError):
        store.load_prompt_memory()


# --- L1: save_user_fact ---

def test_save_user_fact_round_trips_through_load(store):
    store.save_user_fact("likes concise answers")
    assert store.user_facts_path.read_text(encoding="utf-8").strip() == "likes concise answers"
    assert store.load_prompt_memory().endswith("likes concise answers")


def test_save_user_fact_appends_multiple_lines(store):
    store.save_user_fact("first fact")
    store.save_user_fact("second fact")
    lines = store.user_facts_path.read_text(encoding="utf-8").splitlines()
    assert lines == ["first fact", "second fact"]


def test_save_user_fact_empty_raises_value_error(store):
    with pytest.raises(ValueError, match="empty"):
        store.save_user_fact("   ")


# --- L1: save_memory_note ---

def test_save_memory_note_writes_timestamped_line(store):
    store.save_memory_note("auth bug was a stale token")
    line = store.memory_notes_path.read_text(encoding="utf-8").strip()
    # Line format: "- [YYYY-MM-DD HH:MM:SS] auth bug was a stale token"
    assert line.startswith("- [")
    assert line.endswith("] auth bug was a stale token")
    stamp = line[len("- ["): line.index("]")]
    # A real, parseable timestamp rather than a placeholder.
    assert datetime.strptime(stamp, NOTE_TIMESTAMP_FORMAT).year >= 2024


def test_save_memory_note_appears_in_prompt_memory(store):
    store.save_memory_note("regression came from the timeout change")
    assert "regression came from the timeout change" in store.load_prompt_memory()


def test_save_memory_note_empty_raises_value_error(store):
    with pytest.raises(ValueError, match="empty"):
        store.save_memory_note("")


# --- L2: save_lesson ---

def test_save_lesson_inserts_row_and_returns_id(store):
    rid = store.save_lesson("always read the error before retrying")
    assert isinstance(rid, int)
    rows = store.load_recent_lessons()
    assert len(rows) == 1
    assert rows[0].text == "always read the error before retrying"
    assert rows[0].occurrence_count == 1


def test_save_lesson_upsert_increments_occurrence_count(store):
    store.save_lesson("check the config path first")
    store.save_lesson("check the config path first")
    store.save_lesson("check the config path first")
    lessons = store.load_recent_lessons()
    assert len(lessons) == 1
    assert lessons[0].occurrence_count == 3


def test_save_lesson_upsert_does_not_create_duplicate_rows(store):
    store.save_lesson("same lesson")
    store.save_lesson("same lesson")
    count = store._conn.execute("SELECT COUNT(*) AS c FROM lessons").fetchone()["c"]
    assert count == 1


def test_save_lesson_trims_whitespace_before_upsert(store):
    store.save_lesson("  padded lesson  ")
    store.save_lesson("padded lesson")
    lessons = store.load_recent_lessons()
    assert len(lessons) == 1
    assert lessons[0].text == "padded lesson"


def test_save_lesson_empty_raises_value_error(store):
    with pytest.raises(ValueError, match="empty"):
        store.save_lesson("   ")


# --- L2: load_recent_lessons ---

def test_load_recent_lessons_empty_store_returns_empty_list(store):
    assert store.load_recent_lessons() == []


def test_load_recent_lessons_orders_newest_first(store):
    store.save_lesson("oldest")
    _pin_created_at(store, "oldest", 1000)
    store.save_lesson("newest")
    _pin_created_at(store, "newest", 2000)
    assert [lesson.text for lesson in store.load_recent_lessons()] == ["newest", "oldest"]


def test_load_recent_lessons_respects_explicit_limit(store):
    for i in range(5):
        store.save_lesson(f"lesson-{i}")
    assert len(store.load_recent_lessons(limit=2)) == 2
    assert len(store.load_recent_lessons(limit=10)) == 5


def test_load_recent_lessons_invalid_limit_raises_value_error(store):
    with pytest.raises(ValueError):
        store.load_recent_lessons(limit=0)
    with pytest.raises(ValueError):
        store.load_recent_lessons(limit=-1)


# --- L2: search_lessons ---

def test_search_lessons_matches_substring(store):
    store.save_lesson("always commit before deploying")
    store.save_lesson("drink water")
    hits = store.search_lessons("deploy")
    assert len(hits) == 1
    assert hits[0].text == "always commit before deploying"


def test_search_lessons_is_case_insensitive(store):
    store.save_lesson("Retry On Rate Limit")
    assert len(store.search_lessons("rate limit")) == 1


def test_search_lessons_no_match_returns_empty_list(store):
    store.save_lesson("unrelated lesson")
    assert store.search_lessons("deploy") == []


def test_search_lessons_empty_query_raises_value_error(store):
    with pytest.raises(ValueError, match="empty"):
        store.search_lessons("  ")


def test_search_lessons_treats_wildcards_literally(store):
    # '%' and '_' are LIKE wildcards; a query containing them must not match
    # everything.
    store.save_lesson("100 percent coverage")
    assert store.search_lessons("%") == []
    assert store.search_lessons("_") == []
    assert len(store.search_lessons("percent")) == 1


# --- build_memory_prompt ---

def test_build_memory_prompt_empty_store_returns_empty_string(store):
    assert build_memory_prompt(store) == ""


def test_build_memory_prompt_includes_heading_facts_and_lessons(store):
    store.save_user_fact("user fact line")
    store.save_lesson("a lesson worth keeping")
    prompt = build_memory_prompt(store)
    assert prompt.startswith("## Memory")
    assert "user fact line" in prompt
    assert "a lesson worth keeping" in prompt


def test_build_memory_prompt_caps_lessons_at_prompt_limit(store):
    for i in range(LESSON_PROMPT_LIMIT + 5):
        store.save_lesson(f"lesson-{i}")
    prompt = build_memory_prompt(store)
    assert prompt.count("- lesson-") == LESSON_PROMPT_LIMIT


def test_build_memory_prompt_respects_explicit_lesson_limit(store):
    for i in range(6):
        store.save_lesson(f"lesson-{i}")
    assert build_memory_prompt(store, lesson_limit=2).count("- lesson-") == 2


# --- L3 / L4 stubs ---

def test_l3_save_skill_raises_not_implemented(store):
    with pytest.raises(NotImplementedError, match="Spec 6.3"):
        store.save_skill("deploy", "step one")


def test_l3_load_skill_raises_not_implemented(store):
    with pytest.raises(NotImplementedError, match="Spec 6.3"):
        store.load_skill("deploy")


def test_l4_search_external_raises_not_implemented(store):
    with pytest.raises(NotImplementedError, match="Spec 6.3"):
        store.search_external("deploy checklist")


# --- close / context manager ---

def test_context_manager_closes_database(tmp_path):
    with MemoryStore(base_dir=tmp_path / "ctx") as s:
        s.save_lesson("lesson inside the context")
        assert s.load_recent_lessons()
    # Connection closed: further use raises MemoryError at our boundary.
    with pytest.raises(MemoryError):
        s.load_recent_lessons()


def test_context_manager_persists_writes_on_exit(tmp_path):
    base = tmp_path / "ctx2"
    with MemoryStore(base_dir=base) as s:
        s.save_user_fact("persisted fact")
        s.save_lesson("persisted lesson")
    with MemoryStore(base_dir=base) as s2:
        assert "persisted fact" in s2.load_prompt_memory()
        assert [lesson.text for lesson in s2.load_recent_lessons()] == ["persisted lesson"]


def test_close_is_idempotent(store):
    store.close()
    store.close()  # second close is best-effort, must not raise


def test_lesson_dataclass_carries_all_columns(store):
    store.save_lesson("structural lesson")
    lesson = store.load_recent_lessons()[0]
    assert isinstance(lesson, Lesson)
    assert isinstance(lesson.id, int)
    assert isinstance(lesson.created_at, int)
    assert lesson.created_at > 0
    assert lesson.occurrence_count == 1


# --- helpers ---

def _pin_created_at(store: MemoryStore, text: str, ts: int) -> None:
    """Pin a lesson's created_at so load_recent_lessons ordering is deterministic."""
    store._conn.execute(
        "UPDATE lessons SET created_at = ? WHERE text = ?", (ts, text)
    )
    store._conn.commit()
