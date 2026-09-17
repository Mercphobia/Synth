"""Time travel replay, watch mode, and fuzzy session picker for Synth.

Spec #73 (time travel replay), #80 (watch mode), #127 (session picker).
"""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any, Callable

from synth.checkpoints import CheckpointStore
from synth.constants import (
    FUZZY_LIMIT_DEFAULT,
    TIMELINE_PREVIEW_CHARS,
    WATCH_INTERVAL_DEFAULT,
)
from synth.session import SessionStore


class TimeTravel:
    """Replay a session from a checkpoint, message by message.

    Wraps a CheckpointStore + SessionStore pair so callers can rewind to any
    saved snapshot and walk the conversation forward, invoking a callback at
    each step (spec #73).
    """

    def __init__(self, checkpoints: CheckpointStore, sessions: SessionStore) -> None:
        self._checkpoints = checkpoints
        self._sessions = sessions
        self._pause_index: int | None = None

    def replay(
        self,
        session_id: str,
        checkpoint_id: str,
        callback: Callable[[str, str | None], None],
    ) -> None:
        """Restore ``checkpoint_id`` into ``session_id`` then replay messages.

        The checkpoint's messages replace the live session, then each message
        is fed to ``callback(role, content)`` in order. If
        :meth:`pause_at` was called, replay stops after the Nth message
        (0-indexed).
        """
        # Restore the snapshot into the live session DB.
        self._checkpoints.restore_checkpoint(checkpoint_id)
        messages = self._sessions.load_session(session_id)
        limit = self._pause_index
        for i, msg in enumerate(messages):
            if limit is not None and i > limit:
                break
            callback(msg.role, msg.content)

    def pause_at(self, message_index: int) -> None:
        """Cap the next :meth:`replay` so it stops after ``message_index``.

        ``message_index`` is 0-based; replay will emit messages 0..index
        inclusive and then stop. Pass a negative value to clear the cap.
        """
        if message_index < 0:
            self._pause_index = None
        else:
            self._pause_index = message_index

    def get_timeline(self, session_id: str) -> list[dict]:
        """Return one dict per message in the live session.

        Each dict: ``{'index', 'role', 'content_preview', 'has_checkpoint'}``.
        ``has_checkpoint`` is True when a checkpoint exists whose message list
        would end at that index (i.e. the snapshot length minus one equals
        the message index).
        """
        messages = self._sessions.load_session(session_id)
        try:
            checkpoints = self._checkpoints.list_checkpoints(session_id)
        except Exception:
            checkpoints = []

        # Snapshot lengths: for each checkpoint we'd need its message count.
        # restore_checkpoint returns the message list; rather than deserialize
        # every checkpoint (expensive), we cache lengths lazily.
        snap_lengths: set[int] = set()
        for cp in checkpoints:
            try:
                cp_messages = self._checkpoints.restore_checkpoint(cp["id"])
                snap_lengths.add(len(cp_messages) - 1)
            except Exception:
                continue

        timeline: list[dict] = []
        for i, msg in enumerate(messages):
            preview = ""
            if msg.content:
                preview = msg.content[:TIMELINE_PREVIEW_CHARS]
            timeline.append(
                {
                    "index": i,
                    "role": msg.role,
                    "content_preview": preview,
                    "has_checkpoint": i in snap_lengths,
                }
            )
        return timeline


class WatchMode:
    """Poll a directory for file changes and fire a callback per change.

    Single-threaded mtime-based watcher (spec #80). The caller drives the
    loop via :meth:`watch` (blocking) or calls :meth:`poll_once` from its own
    loop.
    """

    def __init__(self, callback: Callable[[Path], None]) -> None:
        self._callback = callback
        self._stopped = False
        self._mtimes: dict[Path, float] = {}

    def stop(self) -> None:
        """Signal the watch loop to exit after the next iteration."""
        self._stopped = True

    def _scan(self, path: Path, pattern: str) -> list[Path]:
        """Return paths under ``path`` matching ``pattern`` with current mtimes."""
        changed: list[Path] = []
        if not path.exists():
            return changed
        if path.is_file():
            files = [path]
        else:
            files = [p for p in path.rglob("*") if p.is_file()]
        for fp in files:
            if not fnmatch.fnmatch(fp.name, pattern):
                continue
            try:
                st = os.stat(fp)
            except OSError:
                continue
            mtime = st.st_mtime
            prev = self._mtimes.get(fp)
            if prev is None:
                self._mtimes[fp] = mtime
                # First sighting: treat as a change so the initial state fires.
                changed.append(fp)
            elif mtime > prev:
                self._mtimes[fp] = mtime
                changed.append(fp)
            # No change: leave stored mtime as-is.
        return changed

    def poll_once(self, path: Path, pattern: str = "*.py") -> list[Path]:
        """Run one scan and fire the callback for each changed file."""
        changed = self._scan(path, pattern)
        for fp in changed:
            self._callback(fp)
        return changed

    def watch(
        self,
        path: Path,
        pattern: str = "*.py",
        interval: float = WATCH_INTERVAL_DEFAULT,
    ) -> None:
        """Block, polling ``path`` every ``interval`` seconds until :meth:`stop``.

        On the first scan every file is reported as "changed" so the caller
        learns the initial state. Subsequent scans only report files whose
        mtime increased.
        """
        self._stopped = False
        while not self._stopped:
            self.poll_once(path, pattern)
            if self._stopped:
                break
            # Sleep in small slices so stop() is responsive.
            elapsed = 0.0
            while elapsed < interval and not self._stopped:
                time.sleep(min(0.1, interval - elapsed))
                elapsed += 0.1


class SessionPicker:
    """Fuzzy-match session metadata against a free-text query (spec #127)."""

    # Scoring constants.
    SCORE_EXACT = 100
    SCORE_PREFIX = 80
    SCORE_SUBSTRING = 60
    SCORE_WORD_BOUNDARY = 40
    SCORE_NONE = 0

    @staticmethod
    def _score_one(haystack: str, needle: str) -> int:
        """Return the best score for ``needle`` against ``haystack``."""
        if not needle:
            return SessionPicker.SCORE_NONE
        low = haystack.lower()
        n = needle.lower()
        if low == n:
            return SessionPicker.SCORE_EXACT
        if low.startswith(n):
            return SessionPicker.SCORE_PREFIX
        if n in low:
            return SessionPicker.SCORE_SUBSTRING
        # Word-boundary: needle matches the start of some word.
        words = low.replace("_", " ").replace("-", " ").split()
        for w in words:
            if w.startswith(n):
                return SessionPicker.SCORE_WORD_BOUNDARY
        return SessionPicker.SCORE_NONE

    @classmethod
    def _score_session(cls, session: dict, query: str) -> int:
        """Best score across the session's id and title fields."""
        sid = str(session.get("id", ""))
        title = str(session.get("title", "") or "")
        return max(
            cls._score_one(sid, query),
            cls._score_one(title, query),
        )

    @classmethod
    def fuzzy_search(
        cls,
        sessions: list[dict],
        query: str,
        limit: int = FUZZY_LIMIT_DEFAULT,
    ) -> list[dict]:
        """Return sessions ranked by fuzzy match score, capped at ``limit``."""
        scored: list[tuple[int, dict]] = []
        for s in sessions:
            score = cls._score_session(s, query)
            if score > cls.SCORE_NONE:
                scored.append((score, s))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[:limit]]

    @classmethod
    def pick_interactive(cls, sessions: list[dict], query: str) -> dict | None:
        """Return the single best fuzzy match, or None if nothing matches."""
        results = cls.fuzzy_search(sessions, query, limit=1)
        return results[0] if results else None
