"""Context management for the Synth agent (spec #84, #86, #87, #159).

Three responsibilities:

* Semantic search (#84) — a lightweight TF-style ranking that counts query
  word matches across the repo's text files and returns scored snippets.
  No embeddings; pure stdlib so it runs anywhere Python 3.14 does.
* Auto-context (#86) — given a user query, surface the most relevant files so
  the agent starts each turn with ground truth instead of guessing.
* Context offloading + 1M+ token window (#87, #159) — when a conversation
  grows past ``max_context_tokens``, the oldest messages are spilled to
  ``~/.synth/contexts/<sha256>.json`` and reloaded on demand, so a long
  session is never truncated silently.

Security posture mirrors repo_map.py and memory.py: offload files live under
a directory we control, and the offload id is a content hash so callers cannot
arbitrarily name a path to read or write.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR

# --- Module-local constants (spec #84/#86/#87/#159) ---

# Default token ceiling. 200k covers the large-context models targeted by
# spec #87 while leaving headroom for the system prompt and tool output.
MAX_CONTEXT_TOKENS_DEFAULT = 200_000

# Subdirectory of ~/.synth/ (CONFIG_DIR) holding offloaded conversation JSON.
OFFLOAD_DIR_NAME = "contexts"

# Number of most-recent messages kept in-window when offloading. Keeping 10
# (system + a few turns) preserves continuity so the model doesn't lose the
# thread the instant old context is spilled.
OFFLOAD_KEEP_RECENT = 10

# Cap on auto_context result count. Five files is enough signal without
# flooding the prompt; the agent can read more via the file tool if needed.
AUTO_CONTEXT_MAX_RESULTS = 5

# Characters per token, the standard 4-char proxy for English text. Cheap
# and stable; a real tokenizer would be more accurate but adds a dependency
# the stdlib-only constraint (spec 7) forbids.
TOKEN_RATIO = 4

# Window-indicator thresholds as a fraction of max_context_tokens. Green is
# comfortable, yellow is getting tight, red means offload should trigger.
WINDOW_GREEN = 0.50
WINDOW_YELLOW = 0.80

# Bytes of a file to sniff for binary detection (matches repo_map so the two
# modules agree on what "binary" means).
BINARY_SNIFF_BYTES = 2048

# Max characters of a matching line included in a snippet. Long lines are
# truncated so one verbose file can't dominate the context block.
SNIPPET_MAX_CHARS = 200


class ContextError(Exception):
    """Raised when context build, offload, or load operations fail."""


def _is_binary(path: Path, sniff_bytes: int = BINARY_SNIFF_BYTES) -> bool:
    """Return True if the file looks binary (NUL byte in the first chunk)."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(sniff_bytes)
    except OSError:
        # Unreadable files are treated as binary so callers skip them.
        return True


def _word_tokenise(query: str) -> list[str]:
    """Split a query into lowered, de-duplicated search words.

    Words shorter than 3 characters are dropped: "is", "a", "to" add noise
    without discriminating between files.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in query.split():
        word = raw.strip(".,;:!?()[]{}\"'`/\\").lower()
        if len(word) >= 3 and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def _make_snippet(text_lowered: str, words: list[str], max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Return a one-line context snippet around the first query-word hit.

    The snippet is collapsed to a single line so it renders cleanly in the
    prompt; if no word is found (shouldn't happen for a scored file) we fall
    back to the start of the file.
    """
    earliest = -1
    for w in words:
        idx = text_lowered.find(w)
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
    if earliest == -1:
        earliest = 0
    start = max(0, earliest - 40)
    end = min(len(text_lowered), start + max_chars)
    # Read from the original-cased text for a readable snippet. We approximate
    # by slicing the lowered string; callers only need a hint, not fidelity.
    raw = text_lowered[start:end]
    raw = " ".join(raw.split())  # collapse internal whitespace/newlines
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text_lowered) else ""
    return f"{prefix}{raw}{suffix}".strip()


class ContextManager:
    """Build, size, offload, and reload the agent's context window.

    Args:
        repo_root: Repository root used for the repo map and auto-context
            file walks. Must exist and be a directory.
        max_context_tokens: Soft ceiling on in-window tokens. Defaults to
            MAX_CONTEXT_TOKENS_DEFAULT (spec #87).
        offload_dir: Directory for spilled messages. Defaults to
            ~/.synth/contexts/ (CONFIG_DIR / OFFLOAD_DIR_NAME). Override in
            tests with a tmp_path to avoid touching the real home dir.

    Raises:
        ContextError: If repo_root is missing/not a dir, or the offload dir
            cannot be created.
    """

    def __init__(
        self,
        repo_root: Path,
        max_context_tokens: int = MAX_CONTEXT_TOKENS_DEFAULT,
        offload_dir: Path | None = None,
    ) -> None:
        if not isinstance(repo_root, Path):
            raise ContextError(f"repo_root must be a Path, got {type(repo_root).__name__}")
        if not repo_root.exists():
            raise ContextError(f"repo root does not exist: {repo_root}")
        if not repo_root.is_dir():
            raise ContextError(f"repo root is not a directory: {repo_root}")
        if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
            raise ContextError("max_context_tokens must be a positive int")

        self.repo_root: Path = repo_root.resolve()
        self.max_context_tokens: int = max_context_tokens

        if offload_dir is None:
            self.offload_dir: Path = Path.home() / CONFIG_DIR / OFFLOAD_DIR_NAME
        else:
            self.offload_dir = Path(offload_dir)

        # Most-recent offload id, if any. build_context surfaces this so the
        # agent knows older turns are recoverable via load_offload().
        self._offload_id: str | None = None

    # --- token estimation (spec #87) ---

    def estimate_tokens(self, text: str) -> int:
        """Return a cheap token estimate: len(text) // TOKEN_RATIO.

        The 4-char proxy is the standard quick estimate for English text.
        It over-counts code slightly but is good enough for window sizing
        without pulling in a tokenizer dependency.
        """
        if not isinstance(text, str):
            raise ContextError("text must be a string")
        return len(text) // TOKEN_RATIO

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        """Flatten messages into a single string for token estimation."""
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif content is None:
                parts.append("")
            else:
                parts.append(json.dumps(content, ensure_ascii=False))
        return "\n".join(parts)

    def context_fit(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if the messages fit within max_context_tokens."""
        return self.estimate_tokens(self._messages_to_text(messages)) <= self.max_context_tokens

    # --- auto-context / semantic search (spec #84, #86) ---

    def auto_context(self, query: str) -> list[dict[str, Any]]:
        """Find files relevant to ``query`` via simple TF-style word matching.

        Splits the query into words, walks the repo's text files, counts how
        many times each word appears (case-insensitive), and returns the top
        AUTO_CONTEXT_MAX_RESULTS as dicts with keys path/score/snippet.

        Returns:
            A list of {'path': str, 'score': int, 'snippet': str}, sorted by
            score descending then path ascending. Empty list for an empty
            query or a repo with no text matches.
        """
        if not isinstance(query, str):
            raise ContextError("query must be a string")
        words = _word_tokenise(query)
        if not words:
            return []

        scored: list[dict[str, Any]] = []
        for path in self._iter_text_files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = text.lower()
            count = sum(lowered.count(w) for w in words)
            if count > 0:
                rel = path.relative_to(self.repo_root)
                scored.append(
                    {
                        "path": str(rel),
                        "score": count,
                        "snippet": _make_snippet(lowered, words),
                    }
                )

        scored.sort(key=lambda r: (-r["score"], r["path"]))
        return scored[:AUTO_CONTEXT_MAX_RESULTS]

    def _iter_text_files(self):
        """Yield text files under repo_root, skipping common junk dirs.

        Mirrors repo_map.py's include/exclude logic but is generator-based so
        a huge repo doesn't load every preview into memory at once.
        """
        exclude = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}
        try:
            entries = sorted(self.repo_root.rglob("*"))
        except OSError as exc:
            raise ContextError(f"cannot walk repo root {self.repo_root}: {exc}") from exc

        for entry in entries:
            if not entry.is_file():
                continue
            parts = entry.relative_to(self.repo_root).parts
            if any(part in exclude for part in parts):
                continue
            # Only consider source-ish text extensions to avoid scanning
            # binaries, images, and lockfiles that pollute match counts.
            if entry.suffix not in (".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".rst", ".js", ".ts"):
                continue
            try:
                if entry.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            if _is_binary(entry):
                continue
            yield entry

    # --- context building (spec #86, #159) ---

    def build_context(self, query: str, messages: list[dict[str, Any]]) -> str:
        """Build a context string from repo map + semantic search + offload.

        Three sections, each included only when it has content:

        1. Repository summary — delegates to RepoMap.build_map when available,
           falling back to a simple file listing. Keeps the agent oriented.
        2. Relevant files — top auto_context results for ``query`` with
           scores and snippets so the model has ground truth immediately.
        3. Offloaded context — a pointer to any spilled conversation so the
           agent knows older turns exist on disk and can reload them.

        The returned string is never empty when the repo has files: at
        minimum the repository summary is present.
        """
        sections: list[str] = []

        summary = self._repo_summary()
        if summary:
            sections.append(summary)

        if isinstance(query, str) and query.strip():
            results = self.auto_context(query)
            if results:
                lines = ["### Relevant files"]
                for r in results:
                    lines.append(f"- {r['path']} (score: {r['score']})")
                    snippet = r.get("snippet", "")
                    if snippet:
                        lines.append(f"    {snippet}")
                sections.append("\n".join(lines))

        if self._offload_id:
            target = self.offload_dir / f"{self._offload_id}.json"
            if target.exists():
                sections.append(
                    f"### Offloaded context\nOlder turns stored at "
                    f"~/.synth/{OFFLOAD_DIR_NAME}/{self._offload_id}.json "
                    f"(reload via context manager)."
                )

        return "\n\n".join(sections)

    def _repo_summary(self) -> str:
        """Return a '### Repository' block listing repo files.

        Delegates to RepoMap when importable; otherwise walks the tree with
        the same include/exclude heuristics. Capped at 50 entries so a large
        repo doesn't blow the prompt.
        """
        try:
            from synth.repo_map import RepoMap, RepoMapError  # local import avoids cycle

            rm = RepoMap(self.repo_root)
            mapping = rm.build_map()
        except Exception:
            # Any failure (import, walk, binary) falls back to a plain list.
            mapping = self._simple_listing()

        if not mapping:
            return "### Repository\n(no source files found)"

        lines = ["### Repository"]
        for rel in sorted(mapping.keys())[:50]:
            lines.append(f"- {rel}")
        return "\n".join(lines)

    def _simple_listing(self) -> dict[str, str]:
        """Fallback file listing when RepoMap is unavailable."""
        result: dict[str, str] = {}
        for path in self._iter_text_files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            preview = "\n".join(text.splitlines()[:5])
            result[str(path.relative_to(self.repo_root))] = preview
        return result

    # --- context offloading (spec #159) ---

    def offload(
        self,
        messages: list[dict[str, Any]],
        keep_recent: int = OFFLOAD_KEEP_RECENT,
    ) -> tuple[list[dict[str, Any]], str]:
        """Spill oldest messages to disk when the window is over budget.

        If the messages still fit, returns them unchanged with an empty id
        (no work to do). Otherwise writes every message *before* the last
        ``keep_recent`` to ``<offload_dir>/<sha256>.json`` and returns the
        trimmed tail plus the offload id.

        Args:
            messages: The full conversation as a list of message dicts.
            keep_recent: How many trailing messages to keep in-window.

        Returns:
            (trimmed_messages, offload_id). offload_id is '' when nothing
            was offloaded (window still fits, or nothing to spill).

        Raises:
            ContextError: If the offload directory cannot be created or the
                payload cannot be written.
        """
        if not isinstance(messages, list):
            raise ContextError("messages must be a list")
        if not isinstance(keep_recent, int) or keep_recent < 0:
            raise ContextError("keep_recent must be a non-negative int")

        total = self.estimate_tokens(self._messages_to_text(messages))
        if total <= self.max_context_tokens:
            return list(messages), ""

        if len(messages) <= keep_recent:
            # Nothing safe to spill without emptying the window entirely.
            return list(messages), ""

        to_offload = messages[:-keep_recent] if keep_recent > 0 else list(messages)
        trimmed = messages[-keep_recent:] if keep_recent > 0 else []

        if not to_offload:
            return list(messages), ""

        payload = json.dumps(to_offload, ensure_ascii=False)
        offload_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        try:
            self.offload_dir.mkdir(parents=True, exist_ok=True)
            target = self.offload_dir / f"{offload_id}.json"
            target.write_text(payload, encoding="utf-8")
        except OSError as exc:
            raise ContextError(f"cannot write offload {offload_id}: {exc}") from exc

        self._offload_id = offload_id
        return list(trimmed), offload_id

    def load_offload(self, offload_id: str) -> list[dict[str, Any]]:
        """Reload messages previously spilled by :meth:`offload`.

        Args:
            offload_id: The sha256 hex id returned by offload().

        Returns:
            The list of message dicts that were spilled.

        Raises:
            ContextError: If the id is malformed, the file is missing, or
                the stored JSON is not a list.
        """
        if not isinstance(offload_id, str) or not offload_id:
            raise ContextError("offload_id must be a non-empty string")
        # A sha256 hex is 64 chars; reject anything else so a caller can't
        # smuggle a path like '../../etc/passwd' into the filename.
        if len(offload_id) != 64 or any(c not in "0123456789abcdef" for c in offload_id):
            raise ContextError(f"offload_id is not a valid sha256 hex: {offload_id!r}")

        target = self.offload_dir / f"{offload_id}.json"
        if not target.exists():
            raise ContextError(f"offload not found: {offload_id}")
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContextError(f"cannot read offload {offload_id}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContextError(f"invalid JSON in offload {offload_id}: {exc}") from exc
        if not isinstance(data, list):
            raise ContextError(f"offload {offload_id} is not a list")
        return data

    # --- window indicator (spec #87) ---

    def window_indicator(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a token-usage snapshot for UI display.

        Returns a dict with:
            used_tokens: estimated tokens currently in-window.
            max_tokens: the configured ceiling.
            percentage: used / max * 100 (0.0 when max is 0).
            status: 'green' (<50%), 'yellow' (50-80%), 'red' (>80%).
        """
        used = self.estimate_tokens(self._messages_to_text(messages))
        pct = (used / self.max_context_tokens * 100) if self.max_context_tokens else 0.0
        if pct >= WINDOW_YELLOW * 100:
            status = "red"
        elif pct >= WINDOW_GREEN * 100:
            status = "yellow"
        else:
            status = "green"
        return {
            "used_tokens": used,
            "max_tokens": self.max_context_tokens,
            "percentage": round(pct, 2),
            "status": status,
        }
