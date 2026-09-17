"""Tests for synth.recipes — RecipeStore, SelfUpdater, LiveMetrics (spec #55, #124, #135)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from synth.recipes import (
    LiveMetrics,
    Recipe,
    RecipeStore,
    SelfUpdater,
)


# ---------------------------------------------------------------------------
# RecipeStore — save / load round trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    r = Recipe(
        name="greet",
        template="Hello {name}, you are {role}.",
        description="Greeting template",
    )
    store.save(r)
    loaded = store.load("greet")
    assert loaded is not None
    assert loaded.name == "greet"
    assert loaded.template == "Hello {name}, you are {role}."
    assert loaded.variables == ["name", "role"]
    assert loaded.description == "Greeting template"


def test_save_upsert_overwrites(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="r1", template="v1={val}", description="first"))
    store.save(Recipe(name="r1", template="v2={val}", description="second"))
    loaded = store.load("r1")
    assert loaded is not None
    assert loaded.template == "v2={val}"
    assert loaded.description == "second"


def test_load_missing_returns_none(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    assert store.load("nonexistent") is None


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_substitutes_variables(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="greet", template="Hi {name}!", description=""))
    out = store.render("greet", {"name": "Alice"})
    assert out == "Hi Alice!"


def test_render_multiple_variables(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="r", template="{a} and {b}", description=""))
    out = store.render("r", {"a": "X", "b": "Y"})
    assert out == "X and Y"


def test_render_missing_variable_raises(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="r", template="Hi {name}!", description=""))
    with pytest.raises(ValueError, match="missing variable"):
        store.render("r", {})


def test_render_extra_variables_ignored(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="r", template="Hi {name}!", description=""))
    out = store.render("r", {"name": "Bob", "extra": "ignored"})
    assert out == "Hi Bob!"


def test_render_no_variables(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="static", template="Just text.", description=""))
    out = store.render("static", {})
    assert out == "Just text."


def test_render_missing_recipe_raises(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    with pytest.raises(Exception, match="not found"):
        store.render("nope", {})


# ---------------------------------------------------------------------------
# from_text
# ---------------------------------------------------------------------------

def test_from_text_parses(tmp_path):
    text = "# name: my-recipe\ndescription: A test recipe\nHello {name}!"
    r = RecipeStore.from_text(text)
    assert r.name == "my-recipe"
    assert r.description == "A test recipe"
    assert r.template == "Hello {name}!"
    assert r.variables == ["name"]


def test_from_text_without_description(tmp_path):
    text = "# name: simple\nJust a template {x}."
    r = RecipeStore.from_text(text)
    assert r.name == "simple"
    assert r.description == ""
    assert r.template == "Just a template {x}."
    assert r.variables == ["x"]


def test_from_text_missing_name_raises():
    with pytest.raises(ValueError, match="missing recipe name"):
        RecipeStore.from_text("no name here\njust text")


def test_from_text_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        RecipeStore.from_text("")


# ---------------------------------------------------------------------------
# list_all / delete
# ---------------------------------------------------------------------------

def test_list_all(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="a", template="ta", description=""))
    store.save(Recipe(name="b", template="tb", description=""))
    all_recipes = store.list_all()
    assert len(all_recipes) == 2
    assert all_recipes[0].name == "a"
    assert all_recipes[1].name == "b"


def test_list_all_empty(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    assert store.list_all() == []


def test_delete_existing(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    store.save(Recipe(name="r", template="t", description=""))
    assert store.delete("r") is True
    assert store.load("r") is None


def test_delete_missing_returns_false(tmp_path):
    store = RecipeStore(db_path=tmp_path / "r.db")
    assert store.delete("nope") is False


# ---------------------------------------------------------------------------
# SelfUpdater — check_for_update (mocked subprocess)
# ---------------------------------------------------------------------------

def _make_completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_self_updater_check_no_update(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "synth"\nversion = "1.0.0"\n'
    )
    updater = SelfUpdater(repo_path=repo)
    tag_lines = (
        "abc123\trefs/tags/v0.9.0\n"
        "def456\trefs/tags/v1.0.0\n"
    )
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_completed(stdout="git\n"),          # rev-parse --git-dir
            _make_completed(stdout=tag_lines),         # ls-remote --tags origin
        ]
        result = updater.check_for_update()
    assert result is not None
    assert result["current"] == "1.0.0"
    assert result["latest"] == "1.0.0"
    assert result["update_available"] is False


def test_self_updater_check_update_available(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "synth"\nversion = "1.0.0"\n'
    )
    updater = SelfUpdater(repo_path=repo)
    tag_lines = (
        "abc\trefs/tags/v1.0.0\n"
        "def\trefs/tags/v1.1.0\n"
        "ghi\trefs/tags/v2.0.0\n"
    )
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_completed(stdout="git\n"),
            _make_completed(stdout=tag_lines),
        ]
        result = updater.check_for_update()
    assert result is not None
    assert result["current"] == "1.0.0"
    assert result["latest"] == "2.0.0"
    assert result["update_available"] is True


def test_self_updater_check_not_in_repo(tmp_path):
    repo = tmp_path / "norepo"
    repo.mkdir()
    updater = SelfUpdater(repo_path=repo)
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.return_value = _make_completed(returncode=128, stderr="not a repo")
        result = updater.check_for_update()
    assert result is None


# ---------------------------------------------------------------------------
# SelfUpdater — update (mocked subprocess)
# ---------------------------------------------------------------------------

def test_self_updater_update_success(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    updater = SelfUpdater(repo_path=repo)
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_completed(stdout="git\n"),       # rev-parse
            _make_completed(stdout="Already up to date.\n"),  # pull
        ]
        assert updater.update() is True
    assert mock_run.call_count == 2


def test_self_updater_update_not_in_repo(tmp_path):
    repo = tmp_path / "norepo"
    repo.mkdir()
    updater = SelfUpdater(repo_path=repo)
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.return_value = _make_completed(returncode=128, stderr="not a repo")
        with pytest.raises(RuntimeError, match="not in a git repo"):
            updater.update()


def test_self_updater_update_pull_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    updater = SelfUpdater(repo_path=repo)
    with patch("synth.recipes.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _make_completed(stdout="git\n"),  # rev-parse succeeds
            _make_completed(returncode=1, stderr="merge conflict"),  # pull fails
        ]
        with pytest.raises(RuntimeError, match="failed"):
            updater.update()


# ---------------------------------------------------------------------------
# LiveMetrics — record + snapshot
# ---------------------------------------------------------------------------

def test_live_metrics_record_snapshot(tmp_path):
    metrics = LiveMetrics()
    metrics.record("gpt-4", input_tokens=100, output_tokens=50, latency_ms=200.0)
    metrics.record("gpt-4", input_tokens=200, output_tokens=100, latency_ms=400.0)
    snap = metrics.snapshot()
    assert snap["total_calls"] == 2
    assert snap["total_input_tokens"] == 300
    assert snap["total_output_tokens"] == 150
    assert snap["avg_latency_ms"] == 300.0
    assert "gpt-4" in snap["by_model"]
    assert snap["by_model"]["gpt-4"]["calls"] == 2


def test_live_metrics_multiple_models(tmp_path):
    metrics = LiveMetrics()
    metrics.record("gpt-4", 100, 50, 200.0)
    metrics.record("claude", 150, 75, 300.0)
    snap = metrics.snapshot()
    assert snap["total_calls"] == 2
    assert len(snap["by_model"]) == 2
    assert snap["by_model"]["gpt-4"]["calls"] == 1
    assert snap["by_model"]["claude"]["calls"] == 1


def test_live_metrics_format_table_non_empty(tmp_path):
    metrics = LiveMetrics()
    metrics.record("gpt-4", 100, 50, 200.0)
    table = metrics.format_table()
    assert "Synth Live Metrics" in table
    assert "gpt-4" in table
    assert "Total calls" in table


def test_live_metrics_format_table_empty(tmp_path):
    metrics = LiveMetrics()
    table = metrics.format_table()
    assert "Synth Live Metrics" in table
    assert "no model calls" in table


def test_live_metrics_avg_latency(tmp_path):
    metrics = LiveMetrics()
    metrics.record("m", 10, 5, 100.0)
    metrics.record("m", 20, 10, 300.0)
    snap = metrics.snapshot()
    assert snap["avg_latency_ms"] == 200.0
    assert snap["by_model"]["m"]["avg_latency_ms"] == 200.0


def test_live_metrics_window_cap(tmp_path):
    metrics = LiveMetrics(window=3)
    for i in range(10):
        metrics.record("m", i, i, float(i * 100))
    snap = metrics.snapshot()
    # total_calls counts all, but avg uses only the window
    assert snap["total_calls"] == 10
    # last 3 calls: i=7,8,9 -> latencies 700, 800, 900 -> avg 800
    assert snap["avg_latency_ms"] == 800.0


def test_live_metrics_format_table_has_headers(tmp_path):
    metrics = LiveMetrics()
    metrics.record("model-a", 10, 5, 50.0)
    table = metrics.format_table()
    assert "Model" in table
    assert "Calls" in table
    assert "Avg" in table
