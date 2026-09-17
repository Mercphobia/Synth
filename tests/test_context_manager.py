"""Tests for synth.context_manager (spec #84, #86, #87, #159).

Stdlib only, no network. Uses tmp_path for both the repo root and the
offload directory so nothing touches the real ~/.synth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth.context_manager import (
    AUTO_CONTEXT_MAX_RESULTS,
    MAX_CONTEXT_TOKENS_DEFAULT,
    OFFLOAD_DIR_NAME,
    OFFLOAD_KEEP_RECENT,
    TOKEN_RATIO,
    ContextError,
    ContextManager,
)


# --- fixtures ---

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repo with a couple of source files."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "alpha.py").write_text(
        "def authenticate(user, token):\n"
        "    '''Authenticate a user by token.'''\n"
        "    return token == 'secret'\n",
        encoding="utf-8",
    )
    (root / "beta.py").write_text(
        "class Database:\n"
        "    '''Database connection handler.'''\n"
        "    def connect(self):\n"
        "        return True\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Demo\n\nThis project has an authenticate function.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def offload_dir(tmp_path: Path) -> Path:
    d = tmp_path / "offloads"
    d.mkdir()
    return d


@pytest.fixture
def cm(repo: Path, offload_dir: Path) -> ContextManager:
    return ContextManager(repo, max_context_tokens=200, offload_dir=offload_dir)


# --- construction ---

def test_init_defaults(repo: Path, offload_dir: Path):
    mgr = ContextManager(repo, offload_dir=offload_dir)
    assert mgr.repo_root == repo.resolve()
    assert mgr.max_context_tokens == MAX_CONTEXT_TOKENS_DEFAULT
    assert mgr.offload_dir == offload_dir
    assert mgr._offload_id is None


def test_init_missing_repo_raises(tmp_path: Path, offload_dir: Path):
    with pytest.raises(ContextError):
        ContextManager(tmp_path / "nope", offload_dir=offload_dir)


def test_init_non_positive_tokens_raises(repo: Path, offload_dir: Path):
    with pytest.raises(ContextError):
        ContextManager(repo, max_context_tokens=0, offload_dir=offload_dir)


# --- estimate_tokens ---

def test_estimate_tokens_math(cm: ContextManager):
    text = "a" * (TOKEN_RATIO * 10)  # 40 chars -> 10 tokens
    assert cm.estimate_tokens(text) == 10


def test_estimate_tokens_empty(cm: ContextManager):
    assert cm.estimate_tokens("") == 0


def test_estimate_tokens_non_string_raises(cm: ContextManager):
    with pytest.raises(ContextError):
        cm.estimate_tokens(123)  # type: ignore[arg-type]


# --- context_fit ---

def test_context_fit_true_small(cm: ContextManager):
    msgs = [{"role": "user", "content": "hi"}]
    assert cm.context_fit(msgs) is True


def test_context_fit_false_large(cm: ContextManager):
    # 200 token cap / TOKEN_RATIO=4 -> 800 chars fit; exceed that.
    big = "x" * 1000
    msgs = [{"role": "user", "content": big}]
    assert cm.context_fit(msgs) is False


# --- auto_context ---

def test_auto_context_finds_matching_files(cm: ContextManager, repo: Path):
    results = cm.auto_context("authenticate token")
    assert len(results) >= 1
    assert results[0]["path"] == "alpha.py"
    assert results[0]["score"] >= 2  # 'authenticate' + 'token' both present
    assert "snippet" in results[0]
    assert isinstance(results[0]["snippet"], str)


def test_auto_context_ranks_relevant_first(cm: ContextManager):
    # 'authenticate' appears in alpha.py and README.md; alpha has it in code
    # plus the word 'token', so it should rank above README.
    results = cm.auto_context("authenticate")
    assert results[0]["path"] == "alpha.py"


def test_auto_context_empty_query(cm: ContextManager):
    assert cm.auto_context("") == []


def test_auto_context_short_words_ignored(cm: ContextManager):
    # 'a' / 'to' are < 3 chars and dropped; no long words -> no results.
    assert cm.auto_context("a to") == []


def test_auto_context_no_matches(cm: ContextManager):
    assert cm.auto_context("nonexistentword") == []


def test_auto_context_capped_at_max(cm: ContextManager):
    # Create many matching files so the AUTO_CONTEXT_MAX_RESULTS cap bites.
    for i in range(20):
        (cm.repo_root / f"file_{i}.py").write_text(
            f"# authenticate module {i}\n", encoding="utf-8"
        )
    results = cm.auto_context("authenticate")
    assert len(results) == AUTO_CONTEXT_MAX_RESULTS


def test_auto_context_non_string_query_raises(cm: ContextManager):
    with pytest.raises(ContextError):
        cm.auto_context(123)  # type: ignore[arg-type]


# --- build_context ---

def test_build_context_returns_nonempty_with_repo_map(cm: ContextManager, repo: Path):
    ctx = cm.build_context("authenticate", [{"role": "user", "content": "hi"}])
    assert isinstance(ctx, str)
    assert len(ctx) > 0
    assert "Repository" in ctx


def test_build_context_includes_search_results(cm: ContextManager):
    ctx = cm.build_context("authenticate", [{"role": "user", "content": "hi"}])
    assert "Relevant files" in ctx
    assert "alpha.py" in ctx


def test_build_context_empty_query_omits_relevant(cm: ContextManager):
    ctx = cm.build_context("", [{"role": "user", "content": "hi"}])
    assert "Relevant files" not in ctx
    assert "Repository" in ctx


def test_build_context_offload_reference(cm: ContextManager):
    # Force an offload then confirm build_context surfaces the pointer.
    msgs = [{"role": "user", "content": "x" * 500} for _ in range(5)]
    trimmed, oid = cm.offload(msgs, keep_recent=1)
    assert oid
    ctx = cm.build_context("authenticate", trimmed)
    assert "Offloaded context" in ctx
    assert oid in ctx


# --- offload / load_offload ---

def test_offload_no_action_when_fits(cm: ContextManager):
    msgs = [{"role": "user", "content": "hi"}]
    trimmed, oid = cm.offload(msgs, keep_recent=OFFLOAD_KEEP_RECENT)
    assert oid == ""
    assert trimmed == msgs


def test_offload_writes_file_and_returns_trimmed(cm: ContextManager, offload_dir: Path):
    msgs = [{"role": "user", "content": "x" * 300} for _ in range(6)]
    trimmed, oid = cm.offload(msgs, keep_recent=2)
    assert oid
    assert len(oid) == 64  # sha256 hex
    target = offload_dir / f"{oid}.json"
    assert target.exists()
    assert len(trimmed) == 2
    # Trimmed tail is the last keep_recent messages.
    assert trimmed == msgs[-2:]


def test_offload_keep_recent_zero(cm: ContextManager):
    msgs = [{"role": "user", "content": "x" * 300} for _ in range(3)]
    trimmed, oid = cm.offload(msgs, keep_recent=0)
    assert oid
    assert trimmed == []


def test_offload_keep_recent_exceeds_length(cm: ContextManager):
    # More keep_recent than messages but still over budget -> no spill.
    msgs = [{"role": "user", "content": "x" * 500} for _ in range(3)]
    trimmed, oid = cm.offload(msgs, keep_recent=10)
    assert oid == ""
    assert trimmed == msgs


def test_offload_id_is_content_hash(cm: ContextManager, offload_dir: Path):
    msgs = [{"role": "user", "content": "x" * 300} for _ in range(4)]
    _, oid = cm.offload(msgs, keep_recent=1)
    # Same content -> same id; reload and compare to verify determinism.
    loaded = cm.load_offload(oid)
    assert loaded == msgs[:-1]


def test_load_offload_round_trip(cm: ContextManager):
    msgs = [{"role": "user", "content": "x" * 300} for _ in range(4)]
    _, oid = cm.offload(msgs, keep_recent=1)
    loaded = cm.load_offload(oid)
    assert isinstance(loaded, list)
    assert loaded == msgs[:-1]
    assert all(isinstance(m, dict) for m in loaded)


def test_load_offload_missing_raises(cm: ContextManager):
    fake = "a" * 64
    with pytest.raises(ContextError):
        cm.load_offload(fake)


def test_load_offload_bad_id_format_raises(cm: ContextManager):
    with pytest.raises(ContextError):
        cm.load_offload("not-a-hash")
    with pytest.raises(ContextError):
        cm.load_offload("")


def test_load_offload_traversal_rejected(cm: ContextManager):
    # 64 chars of path-segment characters: must still be rejected because it
    # contains chars outside the hex charset.
    bad = "g" * 64
    with pytest.raises(ContextError):
        cm.load_offload(bad)


def test_offload_non_list_raises(cm: ContextManager):
    with pytest.raises(ContextError):
        cm.offload("not a list")  # type: ignore[arg-type]


def test_offload_negative_keep_recent_raises(cm: ContextManager):
    with pytest.raises(ContextError):
        cm.offload([], keep_recent=-1)


# --- window_indicator ---

def test_window_indicator_green(cm: ContextManager):
    # max=200 tokens -> 800 chars. 100 chars = 25 tokens = 12.5% -> green.
    msgs = [{"role": "user", "content": "a" * 100}]
    info = cm.window_indicator(msgs)
    assert info["used_tokens"] == 25
    assert info["max_tokens"] == 200
    assert info["percentage"] == 12.5
    assert info["status"] == "green"


def test_window_indicator_yellow(cm: ContextManager):
    # 50% of 200 tokens = 100 tokens = 400 chars.
    msgs = [{"role": "user", "content": "a" * 400}]
    info = cm.window_indicator(msgs)
    assert info["status"] == "yellow"
    assert 50.0 <= info["percentage"] < 80.0


def test_window_indicator_red(cm: ContextManager):
    # 81% of 200 tokens -> 162 tokens = 648 chars.
    msgs = [{"role": "user", "content": "a" * 700}]
    info = cm.window_indicator(msgs)
    assert info["status"] == "red"
    assert info["percentage"] > 80.0


def test_window_indicator_empty(cm: ContextManager):
    info = cm.window_indicator([])
    assert info["used_tokens"] == 0
    assert info["percentage"] == 0.0
    assert info["status"] == "green"


def test_window_indicator_keys(cm: ContextManager):
    info = cm.window_indicator([{"role": "user", "content": "hi"}])
    assert set(info.keys()) == {"used_tokens", "max_tokens", "percentage", "status"}


# --- constants sanity ---

def test_constants_values():
    assert MAX_CONTEXT_TOKENS_DEFAULT == 200_000
    assert OFFLOAD_DIR_NAME == "contexts"
    assert OFFLOAD_KEEP_RECENT == 10
    assert AUTO_CONTEXT_MAX_RESULTS == 5
    assert TOKEN_RATIO == 4
