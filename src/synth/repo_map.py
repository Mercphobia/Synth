"""Repo map builder and searcher.

Walks a repository tree, reads text files, and produces a map of relative
paths to content previews. The map can be searched for substrings (semantic
search is deferred to v2.0 — see spec 6.7) and persisted to disk as JSON so
a session can reload a previously built map without re-walking the tree.

Security posture mirrors the rest of Synth's tools: every path is resolved
under the root and rejected if it escapes, files larger than MAX_FILE_SIZE
are skipped, and binary files are detected by a NUL-byte sniff before any
text decoding is attempted.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from synth.constants import MAX_FILE_SIZE

# --- Module-local constants ---

# Number of leading lines to include in each file's preview. A 20-line
# window gives the agent enough of the module's docstring, imports, and
# top-level definitions to recognise what the file does without loading
# the whole source.
MAX_PREVIEW_LINES = 20

# Bytes of a file to sniff when deciding it is binary. Matches the value
# used by tools_ext._is_binary so both modules agree on what "binary"
# means; a NUL byte in the first 2 KB is a reliable binary signal.
BINARY_SNIFF_BYTES = 2048


class RepoMapError(Exception):
    """Raised when repo-map construction or lookup fails."""


def _is_binary(path: Path, sniff_bytes: int = BINARY_SNIFF_BYTES) -> bool:
    """Return True if the file looks binary (NUL byte in the first chunk)."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(sniff_bytes)
    except OSError:
        # Unreadable files are treated as binary so callers skip them
        # rather than crashing on a permission or IO error.
        return True


def _safe_under(candidate: Path, root: Path) -> Path:
    """Resolve `candidate` and ensure it stays inside `root`.

    Raises RepoMapError if the resolved path escapes the root — this is
    the same path-traversal guard used elsewhere in Synth, applied here
    so symlinked entries or caller-supplied save paths cannot leak out.
    """
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise RepoMapError(
            f"path escapes repo root: {candidate} (root={resolved_root})"
        ) from exc
    return resolved_candidate


class RepoMap:
    """Build, search, save, and load a repo map for a given root directory."""

    def __init__(self, root: Path) -> None:
        """Initialise the RepoMap with a root directory.

        The root must exist and be a directory at construction time — a
        missing root is a configuration error that should surface early
        rather than after a long walk.
        """
        if not isinstance(root, Path):
            raise RepoMapError(f"root must be a Path, got {type(root).__name__}")
        if not root.exists():
            raise RepoMapError(f"repo root does not exist: {root}")
        if not root.is_dir():
            raise RepoMapError(f"repo root is not a directory: {root}")
        self.root: Path = root.resolve()

    # --- building ---

    def build_map(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size: int = MAX_FILE_SIZE,
        max_preview_lines: int = MAX_PREVIEW_LINES,
    ) -> dict[str, str]:
        """Walk the repo and return {relative_path: preview} for text files.

        include_patterns defaults to ['*.py', '*.md']; exclude_patterns
        defaults to ['.git', '__pycache__']. Excluded names are matched
        against every path component, so '.git' skips a .git/ directory
        and anything beneath it. Binary files and files larger than
        max_file_size are silently skipped (the agent simply doesn't see
        them, which matches the grep/glob tools' behaviour).
        """
        if include_patterns is None:
            include_patterns = ["*.py", "*.md"]
        if exclude_patterns is None:
            exclude_patterns = [".git", "__pycache__"]

        result: dict[str, str] = {}

        # rglob('*') walks recursively; sorting gives deterministic output
        # so serialised maps are stable across builds.
        try:
            entries = sorted(self.root.rglob("*"))
        except OSError as exc:
            raise RepoMapError(f"cannot walk repo root {self.root}: {exc}") from exc

        for entry in entries:
            if not entry.is_file():
                continue

            # Exclude by component name: a '.git' match anywhere in the
            # path drops the entry, mirroring how developers think of
            # "skip .git directories".
            parts = entry.relative_to(self.root).parts
            if any(
                fnmatch.fnmatch(part, pattern) for part in parts
                for pattern in exclude_patterns
            ):
                continue

            # Include by filename only (the common case). Patterns like
            # '*.py' match against the basename, not the full relative
            # path, which keeps 'src/foo.py' matched by '*.py'.
            if not any(
                fnmatch.fnmatch(entry.name, pattern)
                for pattern in include_patterns
            ):
                continue

            # Path-traversal guard: symlinked entries may resolve outside
            # the root, which would let a crafted repo leak unrelated
            # files into the map.
            try:
                _safe_under(entry, self.root)
            except RepoMapError:
                continue

            # Size and binary guards: silently skip rather than raise,
            # because the map is best-effort.
            try:
                if entry.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue

            if _is_binary(entry):
                continue

            try:
                text = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            preview_lines = text.splitlines()[:max_preview_lines]
            preview = "\n".join(preview_lines)
            result[str(entry.relative_to(self.root))] = preview

        return result

    # --- searching ---

    def search_map(
        self,
        query: str,
        map_data: dict[str, str],
    ) -> list[tuple[str, float]]:
        """Return (path, score) pairs whose preview contains the query.

        Score is the number of (possibly overlapping) substring matches —
        a file mentioning the query twice ranks above one mentioning it
        once. Results are sorted by score descending, then path ascending
        for stability. A semantic/embedding-based search is planned for
        v2.0 (spec 6.7) but not implemented here.
        """
        if not isinstance(query, str) or not query:
            raise RepoMapError("query must be a non-empty string")

        lowered = query.lower()
        scored: list[tuple[str, float]] = []
        for path, preview in map_data.items():
            haystack = preview.lower()
            count = 0
            start = 0
            while True:
                idx = haystack.find(lowered, start)
                if idx == -1:
                    break
                count += 1
                start = idx + 1
            if count > 0:
                scored.append((path, float(count)))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    # --- persistence ---

    def save_map(self, map_data: dict[str, str], path: Path) -> None:
        """Write the map to disk as JSON.

        The destination must stay under the repo root — saving to an
        arbitrary path would let the agent write outside the sandbox via
        a caller-supplied location.
        """
        if not isinstance(map_data, dict):
            raise RepoMapError("map_data must be a dict")
        target = _safe_under(path, self.root)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(map_data, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RepoMapError(f"cannot write map to {target}: {exc}") from exc

    def load_map(self, path: Path) -> dict[str, str]:
        """Read a previously saved map from disk.

        Raises RepoMapError if the file is missing, not valid JSON, or
        not a string-keyed dict — the agent should rebuild in that case
        rather than operate on a corrupted cache.
        """
        target = _safe_under(path, self.root)
        if not target.exists():
            raise RepoMapError(f"map file does not exist: {target}")
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise RepoMapError(f"cannot read map from {target}: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RepoMapError(f"invalid JSON in {target}: {exc}") from exc

        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise RepoMapError(f"map file {target} is not a {{str: str}} dict")

        return data
