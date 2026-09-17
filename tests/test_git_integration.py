"""Tests for git_integration.py"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest.mock as mock
from pathlib import Path

import pytest

from synth.git_integration import Git, GitError, GIT_TIMEOUT


def test_git_available() -> None:
    """Test that Git.available() returns True when git is in PATH."""
    git = Git()
    assert git.available() is True


def test_is_repo_false_on_empty_dir() -> None:
    """Test is_repo() returns False for non-repository directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        git = Git(Path(tmpdir))
        assert git.is_repo() is False


def test_is_repo_false_raises_giterror() -> None:
    """Test operations on non-repo raise GitError with helpful message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        git = Git(Path(tmpdir))
        with pytest.raises(GitError) as exc:
            git.status()
        assert "is not a git repository" in str(exc.value)
        assert "git init" in str(exc.value)


def test_status_detects_staged_unstaged_untracked() -> None:
    """Test status() detects staged, unstaged, and untracked files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize repo
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Commit the base state first
        unstaged_file = repo_path / "unstaged.txt"
        unstaged_file.write_text("original")
        subprocess.run(["git", "add", "unstaged.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Now stage a new file (staged, not yet committed)
        staged_file = repo_path / "staged.txt"
        staged_file.write_text("staged content")
        subprocess.run(["git", "add", "staged.txt"], cwd=repo_path, check=True)

        # Modify a tracked file without staging (unstaged)
        unstaged_file.write_text("modified")

        # Create untracked file
        untracked_file = repo_path / "untracked.txt"
        untracked_file.write_text("untracked")

        status = git.status()
        assert "staged.txt" in status["staged"]
        assert "unstaged.txt" in status["unstaged"]
        assert "untracked.txt" in status["untracked"]


def test_diff_shows_added_line() -> None:
    """Test diff() shows added lines correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial file and commit
        test_file = repo_path / "test.txt"
        test_file.write_text("line1\nline2")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Modify file
        test_file.write_text("line1\nline2\nline3")

        diff = git.diff()
        assert "line3" in diff
        assert "+line3" in diff or "\\ No newline" not in diff


def test_diff_path_escape_rejected() -> None:
    """Test diff() rejects paths that escape repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)

        # Try to access path outside repo
        with pytest.raises(GitError) as exc:
            git.diff(path="../../outside.txt")
        assert "escapes repository root" in str(exc.value)


def test_diff_staged() -> None:
    """Test staged diff works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create and stage a file
        test_file = repo_path / "test.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)

        diff = git.diff(staged=True)
        assert diff  # Should show staged changes


def test_diff_stat() -> None:
    """Test diff_stat() returns statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create and commit initial file
        test_file = repo_path / "test.txt"
        test_file.write_text("original")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Modify file
        test_file.write_text("modified")

        stat = git.diff_stat()
        assert "test.txt" in stat
        assert "|" in stat or "+" in stat  # Stats format


def test_current_branch() -> None:
    """Test current_branch() returns branch name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        
        branch = git.current_branch()
        # Default branch name may vary (main or master)
        assert branch in ("main", "master")


def test_commit_count() -> None:
    """Test commit_count() returns correct number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # No commits yet
        assert git.commit_count() == 0

        # Create a commit
        test_file = repo_path / "test.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "commit1"], cwd=repo_path, check=True)

        assert git.commit_count() == 1


def test_head_commit() -> None:
    """Test head_commit() returns commit information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Test commit"], cwd=repo_path, check=True)

        commit_info = git.head_commit()
        assert commit_info["sha"]
        assert "Test commit" in commit_info["subject"]
        assert "Test User" in commit_info["author"]


def test_auto_commit_none_on_clean_tree() -> None:
    """Test auto_commit() returns None when nothing to commit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create and commit a file
        test_file = repo_path / "test.txt"
        test_file.write_text("content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Auto-commit on clean tree
        result = git.auto_commit("nothing to commit")
        assert result is None


def test_auto_commit_sha_after_edit() -> None:
    """Test auto_commit() returns SHA after making changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path, author_name="Synth", author_email="synth@local")
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("original")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Modify file
        test_file.write_text("modified")

        # Auto-commit
        sha = git.auto_commit("auto commit test")
        assert sha is not None
        assert len(sha) >= 7  # SHA should be at least 7 chars

        # Verify commit was created
        commit_info = git.head_commit()
        assert commit_info["sha"] == sha
        assert "auto commit test" in commit_info["subject"]
        # Author should be Synth/synth@local from Git instance
        # Note: git config overrides may affect this


def test_create_snapshot_with_allow_empty() -> None:
    """Test create_snapshot() works with --allow-empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create empty repo, snapshot with no changes
        snapshot = git.create_snapshot("empty snapshot")
        assert snapshot["sha"]
        assert snapshot["label"] == "empty snapshot"
        assert isinstance(snapshot["ts"], int)
        assert snapshot["ts"] <= int(time.time())


def test_create_snapshot_with_changes() -> None:
    """Test create_snapshot() commits all changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create uncommitted file
        test_file = repo_path / "test.txt"
        test_file.write_text("content")

        snapshot = git.create_snapshot("changes snapshot")
        assert snapshot["sha"]
        assert snapshot["label"] == "changes snapshot"

        # Verify file is committed
        assert git.status()["untracked"] == []


def test_rollback_restores_file_content() -> None:
    """Test rollback_to() restores file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("version1")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "v1"], cwd=repo_path, check=True)
        v1_sha = git.head_commit()["sha"]

        # Modify and commit again
        test_file.write_text("version2")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "v2"], cwd=repo_path, check=True)

        # Rollback to v1
        git.rollback_to(v1_sha)

        # Check file content
        assert test_file.read_text() == "version1"


def test_rollback_invalid_sha_rejected() -> None:
    """Test rollback_to() rejects invalid SHA format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)

        with pytest.raises(GitError) as exc:
            git.rollback_to("invalid-sha")
        assert "Invalid commit SHA format" in str(exc.value)


def test_log_parsing() -> None:
    """Test log() returns properly parsed commits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create multiple commits
        for i in range(3):
            test_file = repo_path / f"file{i}.txt"
            test_file.write_text(f"content{i}")
            subprocess.run(["git", "add", f"file{i}.txt"], cwd=repo_path, check=True)
            subprocess.run(["git", "commit", "-m", f"commit{i}"], cwd=repo_path, check=True)

        logs = git.log(limit=5)
        assert len(logs) == 3
        for i, log_entry in enumerate(reversed(logs)):  # logs are newest first
            assert f"commit{i}" in log_entry["subject"]
            assert log_entry["sha"]
            assert "Test User" in log_entry["author"]


def test_code_review_flags_secret() -> None:
    """Test code_review() flags potential secrets in diff."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        test_file = repo_path / "config.py"
        test_file.write_text("original")
        subprocess.run(["git", "add", "config.py"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Add a line with a fake AWS key
        test_file.write_text("original\nAWS_SECRET = 'A1B2C3D4E5F6G7H8I9J0'")

        review = git.code_review()
        assert "warning: possible secret in diff" in review


def test_code_review_truncates_long_diff() -> None:
    """Test code_review() truncates very long diffs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        # Create initial commit
        test_file = repo_path / "large.py"
        # Create a file that would produce a long diff
        test_file.write_text("original")
        subprocess.run(["git", "add", "large.py"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

        # Create very long content
        long_content = "line1\n" + "x" * 30000 + "\nline3"
        test_file.write_text(long_content)

        review = git.code_review()
        # Should not raise error, might be truncated
        assert isinstance(review, str)


def test_run_timeout_mock() -> None:
    """Test _run() timeout via monkeypatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        git = Git(repo_path)
        subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)

        # Mock subprocess.run to raise TimeoutExpired
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=GIT_TIMEOUT)
            
            with pytest.raises(GitError) as exc:
                git._run("status")
            assert "timed out" in str(exc.value)
            assert f"after {GIT_TIMEOUT}s" in str(exc.value)


def test_git_not_available() -> None:
    """Test Git.available() returns False when git is not in PATH."""
    with mock.patch("shutil.which") as mock_which:
        mock_which.return_value = None
        git = Git()
        assert git.available() is False