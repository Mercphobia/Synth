"""SQLite-backed session storage for Synth.

Stores sessions and messages per spec section 10. Uses parameterized queries
throughout — user input never touches SQL as text.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from synth.constants import CONFIG_DIR, SESSION_DB_FILENAME, SESSION_LIST_LIMIT

DEFAULT_SESSION_DIR = Path.home() / CONFIG_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    ts INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, ts);
"""


@dataclass(frozen=True)
class Message:
    """One row in the messages table."""

    role: str
    content: str | None = None
    tool_calls: str | None = None  # JSON-encoded
    tool_call_id: str | None = None
    ts: int = 0


class SessionError(Exception):
    """Raised when session storage fails unexpectedly."""


class SessionStore:
    """CRUD interface for sessions and their messages."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open (or create) the session database.

        Args:
            db_path: Path to sessions.db. None uses the default location.

        Raises:
            SessionError: If the database cannot be opened or migrated.
        """
        if db_path is None:
            self.db_path = DEFAULT_SESSION_DIR / SESSION_DB_FILENAME
        else:
            self.db_path = Path(str(db_path)).expanduser()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # Enforce foreign keys: appending to a nonexistent session fails
            # loudly instead of silently orphaning the message.
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise SessionError(f"Cannot open session DB {self.db_path}: {exc}") from exc

    def _migrate(self) -> None:
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SessionError(f"Schema migration failed: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            # Closing is best-effort; the process is likely shutting down.
            pass

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- sessions ---

    def create_session(self, title: str | None = None) -> str:
        """Create a new session and return its generated ID."""
        session_id = str(uuid.uuid4())
        now = int(time.time())
        try:
            self._conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SessionError(f"Failed to create session: {exc}") from exc
        return session_id

    def list_sessions(self, limit: int = SESSION_LIST_LIMIT) -> list[sqlite3.Row]:
        """Return recent sessions, newest first.

        Args:
            limit: Max rows to return (default 20 per spec 10.2).

        Raises:
            SessionError: If the query fails.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        try:
            cursor = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return list(cursor.fetchall())
        except sqlite3.Error as exc:
            raise SessionError(f"Failed to list sessions: {exc}") from exc

    def update_session_title(self, session_id: str, title: str) -> None:
        """Set the title of a session.

        Raises:
            SessionError: If the session does not exist or the write fails.
        """
        now = int(time.time())
        try:
            cursor = self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            if cursor.rowcount == 0:
                raise SessionError(f"Session not found: {session_id}")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise SessionError(f"Failed to update session {session_id}: {exc}") from exc

    # --- messages ---

    def append_message(self, session_id: str, message: Message) -> int:
        """Append one message to a session; return its new row ID.

        Raises:
            SessionError: If the session doesn't exist or the insert fails.
        """
        now = message.ts or int(time.time())
        try:
            cursor = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    message.role,
                    message.content,
                    message.tool_calls,
                    message.tool_call_id,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise SessionError(f"Insert did not return a row ID for session {session_id}")
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as exc:
            raise SessionError(f"Failed to append message to {session_id}: {exc}") from exc

    def load_session(self, session_id: str) -> list[Message]:
        """Load all messages of one session, oldest first.

        Raises:
            SessionError: If the query fails.
        """
        try:
            cursor = self._conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, ts FROM messages "
                "WHERE session_id = ? ORDER BY ts ASC, id ASC",
                (session_id,),
            )
            return [
                Message(
                    role=row["role"],
                    content=row["content"],
                    tool_calls=row["tool_calls"],
                    tool_call_id=row["tool_call_id"],
                    ts=row["ts"],
                )
                for row in cursor.fetchall()
            ]
        except sqlite3.Error as exc:
            raise SessionError(f"Failed to load session {session_id}: {exc}") from exc
