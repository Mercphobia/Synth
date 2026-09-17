"""Git integration wrapper for Synth (spec 6.8).

Provides defensive Git operations: status, diff, auto-commit, snapshot,
rollback, log, and code review scaffold. All methods return errors instead
of raising exceptions to match the ToolResult boundary style.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR


class GitError(Exception):
    """Raised when a Git operation fails."""


# Constants from spec 6.8
GIT_TIMEOUT = 30
MAX_DIFF_REVIEW_CHARS = 20_000

# Secret patterns for basic security scanning in diffs
SECRET_PATTERNS = (
    re.compile(r"(?i)aws.*[\'\"]([A-Z0-9]{20})[\'\"]"),  # AWS access key
    re.compile(r"(?i)(?:secret|token|password|key).*[\'\"][a-zA-Z0-9/+]{40,}[\'\"]"),  # Generic tokens
)


class Git:
    """Git repository wrapper with defensive operations."""

    def __init__(
        self,
        repo_root: Path | None = None,
        author_name: str = "Synth",
        author_email: str = "synth@local",
    ) -> None:
        """Initialize Git wrapper.

        Args:
            repo_root: Path to the repository root. Defaults to current working directory.
            author_name: Git author name for commits.
            author_email: Git author email for commits.
        """
        self.repo_root = repo_root.resolve() if repo_root else Path.cwd().resolve()
        self.author_name = author_name
        self.author_email = author_email

    def available(self) -> bool:
        """Check if Git is available on the system."""
        return shutil.which("git") is not None

    def _run(self, *args: str, timeout: int = GIT_TIMEOUT) -> str:
        """Run a Git command and return its stdout.

        Args:
            *args: Git command arguments.
            timeout: Command timeout in seconds.

        Returns:
            Command stdout as a string.

        Raises:
            GitError: If the command fails or times out.
        """
        try:
            proc = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                raise GitError(f"Git command failed: {' '.join(args)}\n{stderr}")
            return proc.stdout
        except subprocess.TimeoutExpired:
            raise GitError(f"Git command timed out after {timeout}s: {' '.join(args)}")

    def is_repo(self) -> bool:
        """Check if the current directory is a Git repository."""
        try:
            result = self._run("rev-parse", "--is-inside-work-tree")
            return result.strip() == "true"
        except GitError:
            return False

    def status(self) -> dict[str, list[str]]:
        """Get repository status.

        Returns:
            Dictionary with keys 'staged', 'unstaged', 'untracked' containing file paths.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        output = self._run("status", "--porcelain=v1", "-z")
        if not output:
            return {"staged": [], "unstaged": [], "untracked": []}

        staged = []
        unstaged = []
        untracked = []

        # Parse null-separated porcelain output
        entries = [entry for entry in output.split("\x00") if entry]
        for entry in entries:
            if len(entry) < 3:
                continue
            status = entry[:2]
            filepath = entry[3:]
            if not filepath:
                continue

            if status[0] in ("A", "M", "D", "R", "C"):
                # Staged changes
                staged.append(filepath)
            elif status[0] in ("?", "!"):
                # Untracked files
                untracked.append(filepath)
            elif status[1] in ("A", "M", "D", "R", "C"):
                # Unstaged changes
                unstaged.append(filepath)

        return {"staged": staged, "unstaged": unstaged, "untracked": untracked}

    def diff(self, staged: bool = False, path: str | None = None) -> str:
        """Get repository diff.

        Args:
            staged: If True, show staged changes only.
            path: Optional path to limit diff to. Must be inside repo_root.

        Returns:
            Unified diff as a string.

        Raises:
            GitError: If not in a Git repository, path escapes repo, or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        cmd_args = ["diff"]
        if staged:
            cmd_args.append("--staged")

        if path is not None:
            # Validate path doesn't escape repo_root
            try:
                resolved_path = (self.repo_root / path).resolve()
                resolved_path.relative_to(self.repo_root.resolve())
                cmd_args.append(path)
            except ValueError:
                raise GitError(f"Path '{path}' escapes repository root")

        return self._run(*cmd_args)

    def diff_stat(self) -> str:
        """Get diff statistics.

        Returns:
            Diff statistics as a string.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        return self._run("diff", "--stat")

    def current_branch(self) -> str:
        """Get current branch name.

        Returns:
            Current branch name.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        return self._run("branch", "--show-current").strip()

    def commit_count(self) -> int:
        """Get total commit count (0 for an unborn HEAD).

        Returns:
            Number of commits in the repository.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        try:
            count = self._run("rev-list", "--count", "HEAD").strip()
        except GitError as exc:
            # An unborn HEAD (no commits yet) is a valid zero, not an error.
            if "unknown revision" in str(exc) or "does not have any commits" in str(exc):
                return 0
            raise
        return int(count) if count.isdigit() else 0

    def head_commit(self) -> dict[str, str]:
        """Get HEAD commit information.

        Returns:
            Dictionary with keys 'sha', 'subject', 'author'.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        try:
            # Get commit info in format: sha|subject|author
            output = self._run(
                "log",
                "-1",
                "--format=%H|%s|%an <%ae>",
                "HEAD",
            ).strip()
            if not output:
                return {"sha": "", "subject": "", "author": ""}
            
            parts = output.split("|", 2)
            if len(parts) == 3:
                return {
                    "sha": parts[0],
                    "subject": parts[1],
                    "author": parts[2],
                }
            else:
                return {"sha": output, "subject": "", "author": ""}
        except GitError:
            return {"sha": "", "subject": "", "author": ""}

    def auto_commit(self, message: str, include_untracked: bool = True) -> str | None:
        """Automatically commit all changes.

        Args:
            message: Commit message.
            include_untracked: If True, include untracked files.

        Returns:
            New commit SHA if committed, None if nothing to commit.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        # Check if there are changes to commit
        status_info = self.status()
        has_changes = (
            bool(status_info["staged"]) 
            or bool(status_info["unstaged"]) 
            or (include_untracked and bool(status_info["untracked"]))
        )
        
        if not has_changes:
            return None

        # Stage changes
        if include_untracked:
            self._run("add", "-A")
        else:
            self._run("add", "-u")

        # Commit with author overrides
        commit_cmd = [
            "-c", f"user.name={self.author_name}",
            "-c", f"user.email={self.author_email}",
            "commit", "-m", message
        ]
        self._run(*commit_cmd)

        # Return new HEAD SHA
        return self.head_commit()["sha"]

    def create_snapshot(self, label: str) -> dict[str, Any]:
        """Create a snapshot commit of current changes.

        Args:
            label: Snapshot label.

        Returns:
            Dictionary with keys 'sha', 'label', 'ts'.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        message = f"synth-snapshot: {label}"
        timestamp = int(time.time())

        # Check if there are changes or if repo is empty
        status_info = self.status()
        has_changes = (
            bool(status_info["staged"]) 
            or bool(status_info["unstaged"]) 
            or bool(status_info["untracked"])
        )
        
        if not has_changes:
            # Create empty commit if no changes
            commit_cmd = [
                "-c", f"user.name={self.author_name}",
                "-c", f"user.email={self.author_email}",
                "commit", "--allow-empty", "-m", message
            ]
            self._run(*commit_cmd)
        else:
            # Commit all changes
            self.auto_commit(message, include_untracked=True)

        sha = self.head_commit()["sha"]
        return {"sha": sha, "label": label, "ts": timestamp}

    def rollback_to(self, sha: str) -> None:
        """Rollback working tree to a specific commit.

        Args:
            sha: Commit SHA to rollback to.

        Raises:
            GitError: If SHA is invalid, not in a Git repository, or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        # Validate SHA format (7-40 hex chars)
        if not re.match(r"^[0-9a-f]{7,40}$", sha):
            raise GitError(f"Invalid commit SHA format: {sha}")

        # Rollback working tree (keeps history intact)
        self._run("checkout", sha, "--", ".")

    def log(self, limit: int = 20, oneline: bool = True) -> list[dict[str, str]]:
        """Get commit log.

        Args:
            limit: Maximum number of commits to return.
            oneline: If True, use oneline format.

        Returns:
            List of commit dictionaries with keys 'sha', 'subject', 'author', 'date'.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        format_str = "%H|%s|%an <%ae>|%ad" if oneline else "%H|%s|%an <%ae>|%ad"
        cmd = [
            "log",
            f"--format={format_str}",
            f"-n{limit}",
            "--date=iso"
        ]
        
        output = self._run(*cmd)
        if not output:
            return []

        commits = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "sha": parts[0],
                    "subject": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })

        return commits

    def code_review(self) -> str:
        """Generate a code review scaffold from current changes.

        Returns:
            Unified diff of unstaged+staged changes, capped at MAX_DIFF_REVIEW_CHARS,
            with truncation note and secret warnings if applicable.

        Raises:
            GitError: If not in a Git repository or command fails.
        """
        if not self.is_repo():
            raise GitError(
                f"'{self.repo_root}' is not a git repository. "
                f"Fix: run 'git init' in this directory."
            )

        # Get combined diff of staged and unstaged changes
        staged_diff = self.diff(staged=True)
        unstaged_diff = self.diff(staged=False)
        
        combined_diff = ""
        if staged_diff:
            combined_diff += staged_diff
        if unstaged_diff:
            combined_diff += unstaged_diff
            
        if not combined_diff:
            return ""

        # Truncate if too long
        truncated = False
        if len(combined_diff) > MAX_DIFF_REVIEW_CHARS:
            combined_diff = combined_diff[:MAX_DIFF_REVIEW_CHARS]
            truncated = True

        # Check for secrets in added lines
        secret_warning = ""
        added_lines = []
        for line in combined_diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])
        
        for line in added_lines:
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    secret_warning = "\nwarning: possible secret in diff"
                    break
            if secret_warning:
                break

        result = combined_diff
        if truncated:
            result += f"\n...[truncated at {MAX_DIFF_REVIEW_CHARS} characters]"
        if secret_warning:
            result += secret_warning

        return result