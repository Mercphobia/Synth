"""Memory for the Synth agent (spec section 6.3).

Four layers, only the first two implemented:

* L1 Prompt memory — MEMORY.md and USER.md files whose contents are injected
  into the system prompt so the model carries them across turns.
* L2 Session archive — lessons learned, stored in SQLite. FTS5 full-text
  search is a later addition; for now search_lessons uses LIKE.
* L3 Skills memory — stub (v1.0 backlog item — see Spec 6.3).
* L4 External memory — stub (v1.0 backlog item — see Spec 6.3).

Security: every SQL statement is parameterized; user text never touches SQL
as a literal. Path arguments are validated to reject '..' traversal. File
contents are never logged.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from synth.constants import CONFIG_DIR

# --- L1 prompt memory ---
# Memory files live directly under the memory dir and are read into the
# system prompt verbatim (spec 6.3 L1).
MEMORY_DIR_NAME = "memory"  # subdir of ~/.synth/ (CONFIG_DIR)
MEMORY_NOTES_FILENAME = "MEMORY.md"  # free-form notes the agent keeps
USER_FACTS_FILENAME = "USER.md"  # durable facts about the user

# Human-readable timestamp prefix on each appended note line.
# ISO 8601 sorts lexicographically, which keeps MEMORY.md scannable.
NOTE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- L2 lessons store ---
# Same DB engine as sessions.db (spec 10); a separate file keeps memory
# schema migrations independent of session migrations.
MEMORY_DB_FILENAME = "memory.db"

# Default number of recent lessons surfaced by load_recent_lessons.
# Matches SESSION_LIST_LIMIT in spirit: one terminal page of context.
LESSON_DEFAULT_LIMIT = 20

# How many recent lessons build_memory_prompt injects. Kept small — every
# lesson in the prompt costs tokens on every turn.
LESSON_PROMPT_LIMIT = 5

# Guard against unbounded note/lesson text (mirrors MAX_OUTPUT_SIZE intent:
# don't let one write eat the whole context window).
MAX_MEMORY_TEXT_LENGTH = 10_000

# Heading used by build_memory_prompt; the agent learns to expect it.
MEMORY_PROMPT_HEADING = "## Memory"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_lessons_created_at ON lessons(created_at);
"""


@dataclass(frozen=True)
class Lesson:
    """One row of the lessons table (spec 6.3 L2)."""

    id: int
    text: str
    created_at: int
    occurrence_count: int


class MemoryError(Exception):
    """Raised when a memory store cannot be opened, read, or written."""


def _validate_path(path: Path) -> Path:
    """Return the path after checking it cannot escape via '..'.

    Raises:
        MemoryError: If any component of the path is '..'.
    """
    if ".." in path.parts:
        raise MemoryError(f"Rejected path containing '..': {path}")
    return path


def _validate_text(text: str, field: str) -> str:
    """Return stripped text after checking it is non-empty and in range.

    Raises:
        ValueError: If the text is empty or longer than MAX_MEMORY_TEXT_LENGTH.
    """
    if not isinstance(text, str):
        raise ValueError(f"{field} must be a string")
    trimmed = text.strip()
    if not trimmed:
        raise ValueError(f"{field} must not be empty")
    if len(trimmed) > MAX_MEMORY_TEXT_LENGTH:
        raise ValueError(
            f"{field} exceeds {MAX_MEMORY_TEXT_LENGTH} characters"
        )
    return trimmed


class MemoryStore:
    """L1 prompt-memory files plus the L2 lessons database.

    Args:
        base_dir: Directory holding the memory files and memory.db.
            None defaults to ~/.synth/memory/.

    Raises:
        MemoryError: If the directory cannot be created or the database
            cannot be opened, or if base_dir escapes via '..'.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            self.base_dir = Path.home() / CONFIG_DIR / MEMORY_DIR_NAME
        else:
            self.base_dir = _validate_path(Path(base_dir))
        self.memory_notes_path = self.base_dir / MEMORY_NOTES_FILENAME
        self.user_facts_path = self.base_dir / USER_FACTS_FILENAME
        self.db_path = self.base_dir / MEMORY_DB_FILENAME

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MemoryError(f"Cannot create memory dir {self.base_dir}: {exc}") from exc

        try:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryError(f"Cannot open memory DB {self.db_path}: {exc}") from exc

    def close(self) -> None:
        """Close the database connection. Best-effort; idempotent."""
        try:
            self._conn.close()
        except sqlite3.Error:
            # Closing is best-effort; the process is likely shutting down.
            pass

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- L1: prompt memory ---

    def load_prompt_memory(self) -> str:
        """Read MEMORY.md and USER.md for injection into the system prompt.

        Returns:
            The concatenated file contents, or '' if neither file exists.
            Missing files are not an error — a fresh install has no memory.

        Raises:
            MemoryError: If an existing file cannot be read.
        """
        return "\n".join(
            part for part in (self._read_file(self.memory_notes_path),
                              self._read_file(self.user_facts_path))
            if part
        )

    def save_user_fact(self, fact: str) -> None:
        """Append one durable fact about the user to USER.md.

        Args:
            fact: The fact to remember. Whitespace is trimmed.

        Raises:
            ValueError: If the fact is empty.
            MemoryError: If the file cannot be written.
        """
        self._append_line(self.user_facts_path, _validate_text(fact, "fact"))

    def save_memory_note(self, text: str) -> None:
        """Append a timestamped note to MEMORY.md.

        Args:
            text: The note body.

        Raises:
            ValueError: If the text is empty.
            MemoryError: If the file cannot be written.
        """
        note = _validate_text(text, "note")
        stamp = datetime.now().strftime(NOTE_TIMESTAMP_FORMAT)
        self._append_line(self.memory_notes_path, f"- [{stamp}] {note}")

    def _read_file(self, path: Path) -> str:
        """Return file contents, or '' if the file does not exist.

        Raises:
            MemoryError: If an existing file cannot be read.
        """
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise MemoryError(f"Cannot read {path}: {exc}") from exc

    def _append_line(self, path: Path, line: str) -> None:
        """Append one line to a memory file, creating it if needed.

        Raises:
            MemoryError: If the write fails.
        """
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line.rstrip("\n") + "\n")
        except OSError as exc:
            raise MemoryError(f"Cannot write {path}: {exc}") from exc

    # --- L2: lessons store ---

    def save_lesson(self, text: str) -> int:
        """Record a lesson, incrementing occurrence_count if it is known.

        Args:
            text: The lesson text. Whitespace is trimmed.

        Returns:
            The row id of the stored lesson.

        Raises:
            ValueError: If the text is empty.
            MemoryError: If the write fails.
        """
        lesson = _validate_text(text, "lesson")
        now = int(time.time())
        try:
            existing = self._conn.execute(
                "SELECT id FROM lessons WHERE text = ?", (lesson,)
            ).fetchone()
            if existing is None:
                cursor = self._conn.execute(
                    "INSERT INTO lessons (text, created_at, occurrence_count) "
                    "VALUES (?, ?, 1)",
                    (lesson, now),
                )
                self._conn.commit()
                if cursor.lastrowid is None:
                    raise MemoryError("Lesson insert did not return a row id")
                return cursor.lastrowid
            # Known lesson: bump the count, keep the original id and created_at.
            self._conn.execute(
                "UPDATE lessons SET occurrence_count = occurrence_count + 1 "
                "WHERE id = ?",
                (existing["id"],),
            )
            self._conn.commit()
            return existing["id"]
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to save lesson: {exc}") from exc

    def load_recent_lessons(self, limit: int = LESSON_DEFAULT_LIMIT) -> list[Lesson]:
        """Return the most recently recorded lessons.

        Args:
            limit: Max lessons to return.

        Raises:
            ValueError: If limit is not positive.
            MemoryError: If the query fails.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        try:
            cursor = self._conn.execute(
                "SELECT id, text, created_at, occurrence_count FROM lessons "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
            return [_row_to_lesson(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to load lessons: {exc}") from exc

    def search_lessons(self, query: str) -> list[Lesson]:
        """Return lessons whose text contains the query (case-insensitive).

        Uses LIKE; FTS5 full-text search is a v1.0 backlog item (spec 6.3 L2).

        Args:
            query: Substring to search for. LIKE wildcards are escaped so the
                query is matched literally.

        Raises:
            ValueError: If the query is empty.
            MemoryError: If the query fails.
        """
        pattern = "%" + _validate_text(query, "query").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        try:
            cursor = self._conn.execute(
                "SELECT id, text, created_at, occurrence_count FROM lessons "
                "WHERE text LIKE ? ESCAPE '\\' ORDER BY created_at DESC, id DESC",
                (pattern,),
            )
            return [_row_to_lesson(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to search lessons: {exc}") from exc

    # --- L3: skills memory (stub) ---

    def save_skill(self, name: str, content: str) -> str:
        """Persist a reusable skill (spec 6.3 L3).

        v1.0 backlog item — see Spec 6.3.
        """
        raise NotImplementedError("L3 skills memory is not implemented (v1.0 backlog item — see Spec 6.3)")

    def load_skill(self, name: str) -> str:
        """Read back a stored skill (spec 6.3 L3).

        v1.0 backlog item — see Spec 6.3.
        """
        raise NotImplementedError("L3 skills memory is not implemented (v1.0 backlog item — see Spec 6.3)")

    # --- L4: external memory (stub) ---

    def search_external(self, query: str) -> list[str]:
        """Query an external memory source (spec 6.3 L4).

        v1.0 backlog item — see Spec 6.3.
        """
        raise NotImplementedError("L4 external memory is not implemented (v1.0 backlog item — see Spec 6.3)")


def _row_to_lesson(row: sqlite3.Row) -> Lesson:
    """Convert one DB row into a Lesson dataclass."""
    return Lesson(
        id=row["id"],
        text=row["text"],
        created_at=row["created_at"],
        occurrence_count=row["occurrence_count"],
    )


def build_memory_prompt(store: MemoryStore, lesson_limit: int = LESSON_PROMPT_LIMIT) -> str:
    """Format a memory block for injection into the system prompt.

    Includes the user facts from USER.md and the most recent lessons.

    Args:
        store: The memory store to read from.
        lesson_limit: Max lessons to include (default LESSON_PROMPT_LIMIT).

    Returns:
        A '\\n## Memory\\n...' block, or '' if the store holds nothing to
        inject. Callers should skip the empty string rather than appending
        a heading with no content.

    Raises:
        MemoryError: If the store cannot be read.
    """
    sections: list[str] = []

    user_facts = store._read_file(store.user_facts_path)
    if user_facts:
        sections.append(f"### About the user\n{user_facts}")

    notes = store._read_file(store.memory_notes_path)
    if notes:
        sections.append(f"### Notes\n{notes}")

    lessons = store.load_recent_lessons(limit=lesson_limit)
    if lessons:
        body = "\n".join(f"- {lesson.text}" for lesson in lessons)
        sections.append(f"### Recent lessons\n{body}")

    if not sections:
        return ""
    return "\n".join([MEMORY_PROMPT_HEADING, *sections])
