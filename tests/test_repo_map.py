"""Tests for synth.repo_map.

One success and one error test per public function, using tmp_path for an
isolated repo root so nothing touches the real workspace. Pytest is not
executed here — this module is only syntax-verified via py_compile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth.repo_map import (
    MAX_PREVIEW_LINES,
    RepoMap,
    RepoMapError,
)


# --- helpers ---


def _write(path: Path, content: str | bytes) -> None:
    """Write text or bytes to a path, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _seed_repo(root: Path) -> None:
    """Build a small repo tree exercising every guard path."""
    _write(root / "src" / "main.py", "# main module\ndef hello():\n    pass\n")
    _write(root / "docs" / "README.md", "# Project\n\nA sample project.\n")
    _write(root / "src" / "excluded.py", "# should be excluded by pattern\n")
    _write(root / ".git" / "config", "git internal\n")  # excluded dir
    _write(root / "src" / "__pycache__" / "main.cpython-314.pyc", b"\x00binary")
    # A file larger than the max size limit; build_map should skip it.
    _write(root / "src" / "huge.py", "x\n" * 1000)
    # A binary file that passes the include pattern but should be skipped.
    _write(root / "src" / "blob.py", b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    # A file that doesn't match the default include patterns — ignored.
    _write(root / "notes.txt", "ignored by default include\n")


# --- __init__ ---


def test_repo_map_init_success(tmp_path: Path) -> None:
    """RepoMap accepts an existing directory."""
    rm = RepoMap(tmp_path)
    assert rm.root == tmp_path.resolve()


def test_repo_map_init_missing(tmp_path: Path) -> None:
    """RepoMap rejects a non-existent root with RepoMapError."""
    with pytest.raises(RepoMapError):
        RepoMap(tmp_path / "does-not-exist")


# --- build_map ---


def test_build_map_success(tmp_path: Path) -> None:
    """build_map includes/excludes correctly and honours guards."""
    _seed_repo(tmp_path)
    rm = RepoMap(tmp_path)
    result = rm.build_map(max_file_size=500)

    rel_paths = set(result.keys())
    # Included text files matching *.py / *.md appear.
    assert "src/main.py" in rel_paths
    assert "docs/README.md" in rel_paths
    assert "src/excluded.py" in rel_paths
    # .git directory is excluded.
    assert not any(p.startswith(".git") for p in rel_paths)
    # __pycache__ is excluded.
    assert not any("__pycache__" in p for p in rel_paths)
    # Binary and oversized files are skipped.
    assert "src/blob.py" not in rel_paths
    assert "src/huge.py" not in rel_paths
    # Non-matching extensions are skipped by default.
    assert "notes.txt" not in rel_paths
    # Preview is truncated to MAX_PREVIEW_LINES.
    assert result["src/main.py"].count("\n") < MAX_PREVIEW_LINES


def test_build_map_symlink_escape(tmp_path: Path) -> None:
    """Symlinks pointing outside the root are dropped, not followed."""
    outside = tmp_path / ".." / "outside.txt"
    _write(outside.resolve(), "secret")
    link = tmp_path / "leak.py"
    try:
        link.symlink_to(outside.resolve())
    except OSError:
        pytest.skip("symlinks not supported on this platform")
    rm = RepoMap(tmp_path)
    result = rm.build_map()
    assert "leak.py" not in result


# --- search_map ---


def test_search_map_success(tmp_path: Path) -> None:
    """search_map finds substring matches and sorts by score desc."""
    rm = RepoMap(tmp_path)
    data = {
        "a.py": "hello world hello",
        "b.py": "hello",
        "c.py": "nothing here",
    }
    hits = rm.search_map("hello", data)
    assert hits == [("a.py", 2.0), ("b.py", 1.0)]
    # Case-insensitive: 'HELLO' should also match.
    assert rm.search_map("HELLO", data)[0][0] == "a.py"


def test_search_map_empty_query(tmp_path: Path) -> None:
    """Empty query raises RepoMapError rather than returning every entry."""
    rm = RepoMap(tmp_path)
    with pytest.raises(RepoMapError):
        rm.search_map("", {"a.py": "x"})


# --- save_map / load_map ---


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """save_map followed by load_map returns the original data."""
    rm = RepoMap(tmp_path)
    data = {"src/main.py": "preview", "docs/README.md": "intro"}
    target = tmp_path / ".synth" / "repo_map.json"
    rm.save_map(data, target)
    assert target.exists()
    loaded = rm.load_map(target)
    assert loaded == data


def test_load_map_invalid(tmp_path: Path) -> None:
    """load_map raises RepoMapError on missing or malformed files."""
    rm = RepoMap(tmp_path)
    # Missing file.
    with pytest.raises(RepoMapError):
        rm.load_map(tmp_path / "missing.json")
    # Invalid JSON.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RepoMapError):
        rm.load_map(bad)
    # Wrong shape: list instead of dict.
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[]", encoding="utf-8")
    with pytest.raises(RepoMapError):
        rm.load_map(wrong)


def test_save_map_escape(tmp_path: Path) -> None:
    """save_map refuses to write outside the repo root."""
    rm = RepoMap(tmp_path)
    outside = tmp_path / ".." / "escape.json"
    with pytest.raises(RepoMapError):
        rm.save_map({"a": "b"}, outside.resolve())
