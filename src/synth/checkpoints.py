"""Session checkpoints and undo for Synth (spec 6.6).

Snapshots the message history of a session so it can be restored later,
and provides an undo-last-step operation that rewinds to the most recent
user message.

Storage uses pickle of trusted Message dataclass instances only — no
arbitrary deserialization path.  All SQL is parameterized.
"""

from __future__ import annotations

import pickle
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from synth.constants import CONFIG_DIR

# Default checkpoint database location.  Lives next to sessions.db in the
# Synth config directory (~/.synth/).  Defined here rather than in
# constants.py so this module can be added without touching existing files.
DEFAULT_CHECKPOINT_DB_PATH: Path = Path.home() / CONFIG_DIR / "checkpoints.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL,
    messages BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session
    ON checkpoints(session_id, created_at);
"""


class CheckpointError(Exception):
    """Raised when checkpoint storage fails unexpectedly."""


# Lightweight Message mirror — identical shape to synth.session.Message so
# that pickle round-trips are compatible.  Importing session.Message directly
# would couple this module to the session DB schema at import time.
@dataclass(frozen=True)
class _Msg:
    role: str
    content: str | None = None
    tool_calls: str | None = None
    tool_call_id: str | None = None
    ts: int = 0


class CheckpointStore:
    """CRUD for session checkpoints and undo operations.

    Extends the SessionStore pattern: opens a dedicated SQLite file, runs
    migrations on startup, and supports ``with CheckpointStore() as cs:``.
    """

    def __init__(self, db_path: str | Path = DEFAULT_CHECKPOINT_DB_PATH) -> None:
        """Open or create the checkpoint database.

        Args:
            db_path: Path to checkpoints.db.  Defaults to
                ``~/.synth/checkpoints.db``.

        Raises:
            CheckpointError: If the database cannot be opened or migrated.
        """
        self.db_path = Path(str(db_path)).expanduser()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise CheckpointError(
                f"Cannot open checkpoint DB {self.db_path}: {exc}"
            ) from exc

    # --- lifecycle -------------------------------------------------------

    def _migrate(self) -> None:
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Schema migration failed: {exc}") from exc

    def close(self) -> None:
        """Close the database connection (best-effort)."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _serialize(messages: list) -> bytes:
        """Pickle a list of Message-like objects.

        Only Message dataclasses are ever pickled; the module does not
        expose any path that would deserialize untrusted data.
        """
        return pickle.dumps(messages, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize(blob: bytes) -> list:
        """Inverse of ``_serialize``.  Caller guarantees trusted input."""
        return pickle.loads(blob)  # noqa: S301 — trusted, internal use only

    # --- public API ------------------------------------------------------

    def create_checkpoint(self, session_id: str, tag: str = "") -> str:
        """Snapshot the current session messages as a checkpoint.

        Args:
            session_id: The session whose messages should be snapshotted.
                Messages are read via a fresh query so this store can be
                used alongside an open ``SessionStore`` on the same file,
                but the simplest flow is to pass the already-loaded
                message list — see ``create_checkpoint_from_messages``.
            tag: Optional human-readable label (e.g. ``"before refactor"``).

        Returns:
            The new checkpoint ID.

        Raises:
            CheckpointError: If the read or write fails.
        """
        checkpoint_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Read messages from the shared sessions DB.  We open a read-only
        # connection to the sessions DB in the same directory so callers
        # do not have to hand us a SessionStore instance.
        sessions_db = self.db_path.parent / "sessions.db"
        messages: list[_Msg] = []
        if sessions_db.exists():
            try:
                read_conn = sqlite3.connect(
                    f"file:{sessions_db}?mode=ro", uri=True
                )
                read_conn.row_factory = sqlite3.Row
                cur = read_conn.execute(
                    "SELECT role, content, tool_calls, tool_call_id, ts "
                    "FROM messages WHERE session_id = ? ORDER BY ts ASC, id ASC",
                    (session_id,),
                )
                messages = [
                    _Msg(
                        role=r["role"],
                        content=r["content"],
                        tool_calls=r["tool_calls"],
                        tool_call_id=r["tool_call_id"],
                        ts=r["ts"],
                    )
                    for r in cur.fetchall()
                ]
                read_conn.close()
            except sqlite3.Error as exc:
                raise CheckpointError(
                    f"Failed to read messages for session {session_id}: {exc}"
                ) from exc

        try:
            self._conn.execute(
                "INSERT INTO checkpoints (id, session_id, tag, created_at, messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (checkpoint_id, session_id, tag, now, sqlite3.Binary(self._serialize(messages))),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to create checkpoint: {exc}") from exc
        return checkpoint_id

    def create_checkpoint_from_messages(
        self, session_id: str, messages: list, tag: str = ""
    ) -> str:
        """Create a checkpoint from an explicit message list.

        Useful when the caller already has messages in memory (e.g. from
        a ``SessionStore.load_session`` call) and does not want a second
        read round-trip.

        Returns:
            The new checkpoint ID.
        """
        checkpoint_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            self._conn.execute(
                "INSERT INTO checkpoints (id, session_id, tag, created_at, messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    checkpoint_id,
                    session_id,
                    tag,
                    now,
                    sqlite3.Binary(self._serialize(messages)),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Failed to create checkpoint: {exc}") from exc
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> list:
        """Load the messages stored in a checkpoint.

        Returns:
            List of Message-like dataclass instances (same shape as
            ``synth.session.Message``).

        Raises:
            CheckpointError: If the checkpoint does not exist or the read
                fails.
        """
        try:
            cur = self._conn.execute(
                "SELECT messages FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise CheckpointError(f"Checkpoint not found: {checkpoint_id}")
            return self._deserialize(bytes(row["messages"]))
        except sqlite3.Error as exc:
            raise CheckpointError(
                f"Failed to restore checkpoint {checkpoint_id}: {exc}"
            ) from exc

    def list_checkpoints(self, session_id: str) -> list[dict]:
        """Return metadata for every checkpoint of a session, newest first.

        Each entry is ``{'id', 'tag', 'created_at'}``.
        """
        try:
            cur = self._conn.execute(
                "SELECT id, tag, created_at FROM checkpoints "
                "WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            )
            return [
                {"id": row["id"], "tag": row["tag"], "created_at": row["created_at"]}
                for row in cur.fetchall()
            ]
        except sqlite3.Error as exc:
            raise CheckpointError(
                f"Failed to list checkpoints for {session_id}: {exc}"
            ) from exc

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.  Returns True if a row was removed."""
        try:
            cur = self._conn.execute(
                "DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            raise CheckpointError(
                f"Failed to delete checkpoint {checkpoint_id}: {exc}"
            ) from exc

    def undo_last_step(self, session_id: str) -> bool:
        """Delete assistant/tool messages back to (but not including) the
        last user message.

        This implements the spec 6.6 "undo" semantics: one invocation
        removes the most recent assistant turn plus any tool calls/results
        that preceded it, stopping at the user message that triggered them.

        Args:
            session_id: The session whose messages table should be rewound.
                Operates on the sessions DB next to this checkpoint DB.

        Returns:
            True if any rows were deleted, False if nothing to undo.

        Raises:
            CheckpointError: If the sessions DB cannot be opened or the
                write fails.
        """
        sessions_db = self.db_path.parent / "sessions.db"
        if not sessions_db.exists():
            raise CheckpointError(f"Sessions DB not found: {sessions_db}")
        try:
            conn = sqlite3.connect(str(sessions_db))
            try:
                # Find the id of the most recent user message.
                cur = conn.execute(
                    "SELECT id FROM messages "
                    "WHERE session_id = ? AND role = 'user' "
                    "ORDER BY ts DESC, id DESC LIMIT 1",
                    (session_id,),
                )
                user_row = cur.fetchone()
                if user_row is None:
                    return False
                user_id = user_row[0]

                # Delete everything after that user message.
                del_cur = conn.execute(
                    "DELETE FROM messages WHERE session_id = ? AND id > ?",
                    (session_id, user_id),
                )
                conn.commit()
                return del_cur.rowcount > 0
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise CheckpointError(f"Undo failed for {session_id}: {exc}") from exc
