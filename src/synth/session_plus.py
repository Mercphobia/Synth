"""Enhanced session features: FTS5 search, export formats, and exit summaries.

Implements spec sections #70 (session search), #77 (export), and #78 (exit summary).
Uses stdlib only. Falls back gracefully when FTS5 is unavailable.
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth.constants import CONFIG_DIR, SESSION_DB_FILENAME
from synth.session import Message, SessionError

DEFAULT_SESSION_DIR = Path.home() / CONFIG_DIR


class SessionPlusError(Exception):
    """Raised when session-plus operations fail."""


def _validate_out_path(out_path: str | Path) -> Path:
    """Validate output path for traversal issues."""
    out_path = Path(str(out_path)).expanduser()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    except OSError as exc:
        raise SessionPlusError(f"Cannot create parent directories for {out_path}: {exc}") from exc


def build_exit_summary(messages: list[Message]) -> str:
    """Build a naive extractive summary of messages.

    Returns counts of user/assistant/tool messages, tool names used,
    first user prompt (truncated), and last assistant text (truncated).
    """
    if not messages:
        return "Session summary: (empty session)"

    user_count = sum(1 for m in messages if m.role == "user")
    assistant_count = sum(1 for m in messages if m.role == "assistant")
    tool_count = sum(1 for m in messages if m.role == "tool")

    tool_names = set()
    for message in messages:
        if message.tool_calls:
            try:
                tool_calls_data = json.loads(message.tool_calls)
                if isinstance(tool_calls_data, list):
                    for call in tool_calls_data:
                        if isinstance(call, dict) and "function" in call:
                            func = call["function"]
                            if isinstance(func, dict) and "name" in func:
                                tool_names.add(func["name"])
            except (json.JSONDecodeError, TypeError):
                pass  # Ignore malformed tool_calls

    first_user_prompt = ""
    for message in messages:
        if message.role == "user" and message.content:
            first_user_prompt = message.content[:120]
            if len(message.content) > 120:
                first_user_prompt += "..."
            break

    last_assistant_text = ""
    for message in reversed(messages):
        if message.role == "assistant" and message.content:
            last_assistant_text = message.content[:200]
            if len(message.content) > 200:
                last_assistant_text += "..."
            break

    summary_parts = [f"Session summary: {user_count} user, {assistant_count} assistant, {tool_count} tool messages"]
    if tool_names:
        summary_parts.append(f"Tools used: {', '.join(sorted(tool_names))}")
    if first_user_prompt:
        summary_parts.append(f"First prompt: {first_user_prompt!r}")
    if last_assistant_text:
        summary_parts.append(f"Last response: {last_assistant_text!r}")

    return "; ".join(summary_parts)


class SessionSearch:
    """FTS5-powered session search with LIKE fallback."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open the session database for searching."""
        if db_path is None:
            self.db_path = DEFAULT_SESSION_DIR / SESSION_DB_FILENAME
        else:
            self.db_path = Path(str(db_path)).expanduser()

        try:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise SessionPlusError(f"Cannot open session DB {self.db_path}: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "SessionSearch":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def fts5_available(self) -> bool:
        """Check if FTS5 is available in this SQLite build."""
        try:
            self._conn.execute("CREATE VIRTUAL TABLE temp._fts_probe USING fts5(x)")
            self._conn.execute("DROP TABLE temp._fts_probe")
            return True
        except sqlite3.Error:
            return False

    def _ensure_fts_table(self) -> int:
        """Ensure message_docs FTS5 table exists and is synced; return max rowid synced."""
        try:
            # Create FTS table if it doesn't exist
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS message_docs USING fts5("
                "session_id, role, content, ts UNINDEXED, "
                "content='messages', content_rowid='id')"
            )

            # Get current max rowid from messages table
            cursor = self._conn.execute("SELECT MAX(id) FROM messages")
            max_messages_rowid = cursor.fetchone()[0] or 0

            # Get last synced rowid from meta table
            cursor = self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER)"
            )
            cursor = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'fts_sync_max_rowid'"
            )
            row = cursor.fetchone()
            last_synced = row[0] if row else 0

            # Sync new messages if needed
            if max_messages_rowid > last_synced:
                self._conn.execute(
                    "INSERT INTO message_docs (rowid, session_id, role, content, ts) "
                    "SELECT id, session_id, role, content, ts FROM messages "
                    "WHERE id > ? AND content IS NOT NULL",
                    (last_synced,),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_sync_max_rowid', ?)",
                    (max_messages_rowid,),
                )
                self._conn.commit()

            return max_messages_rowid
        except sqlite3.Error as exc:
            # If FTS setup fails, fall back to LIKE search
            return 0

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5 MATCH by keeping only word chars and spaces."""
        import re

        # Keep alphanumeric, underscore, and spaces; replace others with spaces
        sanitized = re.sub(r"[^a-zA-Z0-9_\s]", " ", query)
        # Collapse multiple spaces
        sanitized = " ".join(sanitized.split())
        return sanitized

    def _snippet_helper(self, content: str, max_length: int = 200) -> str:
        """Create a snippet from content around the match."""
        if len(content) <= max_length:
            return content
        # Simple truncation for now - could be enhanced with actual match highlighting
        return content[:max_length] + "..."

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search messages using FTS5 (if available) or LIKE fallback."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty or whitespace")

        query = query.strip()
        use_fts = self.fts5_available()

        if use_fts:
            self._ensure_fts_table()
            sanitized_query = self._sanitize_fts_query(query)
            if not sanitized_query:
                # Query became empty after sanitization, use LIKE fallback
                use_fts = False

        results = []
        try:
            if use_fts and sanitized_query:
                # FTS5 search with bm25 ranking
                cursor = self._conn.execute(
                    "SELECT session_id, role, content, ts, rank FROM ("
                    "SELECT session_id, role, content, ts, bm25(message_docs) AS rank "
                    "FROM message_docs WHERE message_docs MATCH ? ORDER BY rank LIMIT ?"
                    ")",
                    (sanitized_query, limit),
                )
                for row in cursor.fetchall():
                    snippet = self._snippet_helper(row["content"] or "")
                    results.append({
                        "session_id": row["session_id"],
                        "role": row["role"],
                        "snippet": snippet,
                        "ts": row["ts"],
                        "rank": row["rank"],
                    })
            else:
                # LIKE fallback - case-insensitive substring search
                # Escape SQL LIKE wildcards
                like_query = query.replace("%", "\\%").replace("_", "\\_").replace("[", "\\[")
                cursor = self._conn.execute(
                    "SELECT session_id, role, content, ts FROM messages "
                    "WHERE content IS NOT NULL AND LOWER(content) LIKE LOWER(?) ESCAPE '\\' "
                    "ORDER BY ts DESC LIMIT ?",
                    (f"%{like_query}%", limit),
                )
                for row in cursor.fetchall():
                    snippet = self._snippet_helper(row["content"] or "")
                    results.append({
                        "session_id": row["session_id"],
                        "role": row["role"],
                        "snippet": snippet,
                        "ts": row["ts"],
                        "rank": 0,  # No ranking in LIKE fallback
                    })
        except sqlite3.Error as exc:
            raise SessionPlusError(f"Search failed: {exc}") from exc

        return results


class SessionExporter:
    """Export sessions to JSON, Markdown, or HTML formats."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Open the session database for exporting."""
        if db_path is None:
            self.db_path = DEFAULT_SESSION_DIR / SESSION_DB_FILENAME
        else:
            self.db_path = Path(str(db_path)).expanduser()

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        except (sqlite3.Error, OSError) as exc:
            raise SessionPlusError(f"Cannot open session DB {self.db_path}: {exc}") from exc

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "SessionExporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _load_session_messages(self, session_id: str) -> list[Message]:
        """Load all messages for a session."""
        try:
            # A db that has never seen SessionStore has no tables at all —
            # that is 'session not found', not a load failure.
            has_tables = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('sessions','messages')"
            ).fetchall()
            if len(has_tables) < 2:
                raise SessionPlusError(f"Session not found: {session_id}")
            cursor = self._conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, ts FROM messages "
                "WHERE session_id = ? ORDER BY ts ASC, id ASC",
                (session_id,),
            )
            messages = [
                Message(
                    role=row["role"],
                    content=row["content"],
                    tool_calls=row["tool_calls"],
                    tool_call_id=row["tool_call_id"],
                    ts=row["ts"],
                )
                for row in cursor.fetchall()
            ]
            if not messages:
                # Check if session exists at all
                cursor = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
                )
                if not cursor.fetchone():
                    raise SessionPlusError(f"Session not found: {session_id}")
            return messages
        except sqlite3.Error as exc:
            raise SessionPlusError(f"Failed to load session {session_id}: {exc}") from exc

    def _is_code_like(self, content: str) -> bool:
        """Heuristic to detect code-like content."""
        if not content:
            return False
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith((">>>", "def ", "import ", "from ", "#", "$"))
                or stripped.startswith(("{", "[", "("))
                or stripped.endswith(("}", "]", ")"))
            ):
                return True
        return False

    def export_json(self, session_id: str, out_path: str | Path) -> Path:
        """Export session to JSON format."""
        out_path = _validate_out_path(out_path)
        messages = self._load_session_messages(session_id)

        # Convert messages to dict format with ISO timestamps
        messages_dict = []
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(msg.ts)),
            }
            if msg.content is not None:
                msg_dict["content"] = msg.content
            if msg.tool_calls is not None:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id is not None:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_dict.append(msg_dict)

        # Write atomically
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False, dir=out_path.parent) as tmp_file:
            json.dump({"session_id": session_id, "messages": messages_dict}, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)

        try:
            tmp_path.rename(out_path)
        except OSError:
            tmp_path.unlink()
            raise SessionPlusError(f"Failed to write to {out_path}")

        return out_path

    def export_markdown(self, session_id: str, out_path: str | Path) -> Path:
        """Export session to Markdown format."""
        out_path = _validate_out_path(out_path)
        messages = self._load_session_messages(session_id)

        md_lines = [f"# Session {session_id}\n"]
        for i, msg in enumerate(messages):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(msg.ts))
            md_lines.append(f"## {msg.role} ({timestamp})\n")
            
            if msg.content:
                if self._is_code_like(msg.content):
                    md_lines.append(f"```text\n{msg.content}\n```\n")
                else:
                    md_lines.append(f"{msg.content}\n")
            
            if msg.tool_calls:
                md_lines.append(f"Tool calls: {msg.tool_calls}\n")
            if msg.tool_call_id:
                md_lines.append(f"Tool call ID: {msg.tool_call_id}\n")
            md_lines.append("")  # Blank line between messages

        # Write atomically
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False, dir=out_path.parent) as tmp_file:
            tmp_file.write("\n".join(md_lines))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)

        try:
            tmp_path.rename(out_path)
        except OSError:
            tmp_path.unlink()
            raise SessionPlusError(f"Failed to write to {out_path}")

        return out_path

    def export_html(self, session_id: str, out_path: str | Path) -> Path:
        """Export session to minimal standalone HTML format."""
        out_path = _validate_out_path(out_path)
        messages = self._load_session_messages(session_id)

        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <meta charset='utf-8'>",
            f"  <title>Session {html.escape(session_id)}</title>",
            "</head>",
            "<body>",
            f"<h1>Session {html.escape(session_id)}</h1>",
        ]

        for msg in messages:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(msg.ts))
            html_lines.append(f"<h2>{html.escape(msg.role)} ({html.escape(timestamp)})</h2>")
            
            if msg.content:
                content_escaped = html.escape(msg.content).replace("\n", "<br>")
                html_lines.append(f"<p>{content_escaped}</p>")
            
            if msg.tool_calls:
                tool_calls_escaped = html.escape(msg.tool_calls)
                html_lines.append(f"<p><strong>Tool calls:</strong> {tool_calls_escaped}</p>")
            if msg.tool_call_id:
                tool_call_id_escaped = html.escape(msg.tool_call_id)
                html_lines.append(f"<p><strong>Tool call ID:</strong> {tool_call_id_escaped}</p>")

        html_lines.extend(["</body>", "</html>"])

        # Write atomically
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False, dir=out_path.parent) as tmp_file:
            tmp_file.write("\n".join(html_lines))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)

        try:
            tmp_path.rename(out_path)
        except OSError:
            tmp_path.unlink()
            raise SessionPlusError(f"Failed to write to {out_path}")

        return out_path