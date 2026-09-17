"""Tests for session_plus.py features.

These are integration tests that create actual session databases.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synth.session import Message, SessionStore
from synth.session_plus import (
    SessionExporter,
    SessionPlusError,
    SessionSearch,
    build_exit_summary,
)


def test_build_exit_summary_empty() -> None:
    """Test summary with empty messages list."""
    assert build_exit_summary([]) == "Session summary: (empty session)"


def test_build_exit_summary_counts() -> None:
    """Test message counting in summary."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?"),
        Message(role="assistant", content="I'm fine"),
        Message(role="tool", content="Result", tool_calls='[{"function": {"name": "search"}}]'),
    ]
    summary = build_exit_summary(messages)
    assert "2 user, 2 assistant, 1 tool messages" in summary
    assert "Tools used: search" in summary


def test_build_exit_summary_tool_names() -> None:
    """Test extraction of tool names."""
    messages = [
        Message(role="assistant", content="Let me search", tool_calls='[{"function": {"name": "search"}}]'),
        Message(role="tool", content="Found results", tool_call_id="1"),
        Message(role="assistant", content="Now read", tool_calls='[{"function": {"name": "read_file"}}]'),
    ]
    summary = build_exit_summary(messages)
    assert "read_file" in summary
    assert "search" in summary
    # Order shouldn't matter, just presence


def test_build_exit_summary_truncation() -> None:
    """Test truncation of first and last messages."""
    long_text = "A" * 200
    messages = [
        Message(role="user", content=long_text),
        Message(role="assistant", content="Response 1"),
        Message(role="assistant", content=long_text * 2),
    ]
    summary = build_exit_summary(messages)
    assert "First prompt:" in summary
    assert "Last response:" in summary
    assert len(summary) < 1000  # Shouldn't be huge


def test_session_search_fts5_available() -> None:
    """Test FTS5 availability detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        search = SessionSearch(db_path)
        try:
            available = search.fts5_available()
            # Should return a boolean
            assert isinstance(available, bool)
        finally:
            search.close()


def test_session_search_empty_query() -> None:
    """Test that empty query raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        search = SessionSearch(db_path)
        try:
            with pytest.raises(ValueError, match="must not be empty"):
                search.search("")
            with pytest.raises(ValueError, match="must not be empty"):
                search.search("   ")
        finally:
            search.close()


def test_session_search_context_manager() -> None:
    """Test SessionSearch works as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with SessionSearch(db_path) as search:
            # Should be able to call methods
            _ = search.fts5_available()
        # Should be closed after context exit


def test_session_search_like_fallback() -> None:
    """Test LIKE fallback when FTS5 is unavailable or query sanitized empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create a session with messages
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="Hello world"))
            store.append_message(session_id, Message(role="assistant", content="Hi there"))
            store.append_message(session_id, Message(role="user", content="Search for something"))
        
        with SessionSearch(db_path) as search:
            # Test LIKE fallback (should work regardless of FTS5)
            results = search.search("world", limit=5)
            assert len(results) == 1
            assert results[0]["session_id"] == session_id
            assert "world" in results[0]["snippet"].lower()


def test_session_search_special_chars() -> None:
    """Test that queries with special characters don't crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="Test [brackets]"))
        
        with SessionSearch(db_path) as search:
            # These queries should not crash
            search.search("[test]", limit=5)
            search.search("test*", limit=5)
            search.search("test?", limit=5)
            search.search("te[]st", limit=5)


def test_session_exporter_load_session() -> None:
    """Test SessionExporter._load_session_messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="Hello"))
            store.append_message(session_id, Message(role="assistant", content="Hi"))
        
        with SessionExporter(db_path) as exporter:
            messages = exporter._load_session_messages(session_id)
            assert len(messages) == 2
            assert messages[0].role == "user"
            assert messages[1].role == "assistant"


def test_session_exporter_missing_session() -> None:
    """Test that missing session raises SessionPlusError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionExporter(db_path) as exporter:
            with pytest.raises(SessionPlusError, match="Session not found"):
                exporter.export_json("nonexistent", Path(tmpdir) / "out.json")


def test_session_exporter_json_roundtrip() -> None:
    """Test JSON export and round-trip loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="Hello", ts=1234567890))
            store.append_message(session_id, Message(role="assistant", content="Hi", tool_calls='[{"function": {"name": "test"}}]', ts=1234567891))
        
        with SessionExporter(db_path) as exporter:
            out_path = Path(tmpdir) / "export.json"
            exporter.export_json(session_id, out_path)
            
            # Verify file exists
            assert out_path.exists()
            
            # Load and verify
            with open(out_path) as f:
                data = json.load(f)
                assert data["session_id"] == session_id
                assert len(data["messages"]) == 2
                assert data["messages"][0]["role"] == "user"
                assert data["messages"][0]["content"] == "Hello"
                assert "ts" in data["messages"][0]
                assert data["messages"][1]["tool_calls"] == '[{"function": {"name": "test"}}]'


def test_session_exporter_markdown_code_fence() -> None:
    """Test Markdown export with code fence detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="def hello():\n    return 'world'", ts=1234567890))
            store.append_message(session_id, Message(role="assistant", content="Regular text", ts=1234567891))
        
        with SessionExporter(db_path) as exporter:
            out_path = Path(tmpdir) / "export.md"
            exporter.export_markdown(session_id, out_path)
            
            assert out_path.exists()
            content = out_path.read_text()
            assert "# Session" in content
            assert "def hello():" in content
            # Should have code fences for the Python-like content
            assert "```text" in content or "```" in content


def test_session_exporter_html_escaping() -> None:
    """Test HTML export escapes special characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="<script>alert('xss')</script>", ts=1234567890))
        
        with SessionExporter(db_path) as exporter:
            out_path = Path(tmpdir) / "export.html"
            exporter.export_html(session_id, out_path)
            
            assert out_path.exists()
            content = out_path.read_text()
            assert "&lt;script&gt;" in content
            assert "<script>" not in content  # Should be escaped


def test_session_exporter_atomic_write() -> None:
    """Test that export writes atomically (tmp + rename)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="Hello"))
        
        with SessionExporter(db_path) as exporter:
            out_path = Path(tmpdir) / "export.md"
            # Create a file at the target path
            out_path.write_text("old content")
            
            # Export should overwrite atomically
            exporter.export_markdown(session_id, out_path)
            
            # Verify new content
            content = out_path.read_text()
            assert "old content" not in content
            assert "Hello" in content


def test_session_exporter_context_manager() -> None:
    """Test SessionExporter works as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with SessionExporter(db_path) as exporter:
            # Should be usable
            assert exporter.db_path == Path(db_path)
        # Should be closed after context exit


def test_validate_out_path_traversal() -> None:
    """Test that output path validation creates parent directories."""
    from synth.session_plus import _validate_out_path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a nested path
        out_path = Path(tmpdir) / "deep" / "nested" / "file.txt"
        validated = _validate_out_path(out_path)
        assert validated == out_path
        assert out_path.parent.exists()
        
        # Test with string path
        out_path_str = str(Path(tmpdir) / "another" / "file.txt")
        validated_str = _validate_out_path(out_path_str)
        assert validated_str == Path(out_path_str)
        assert Path(out_path_str).parent.exists()


def test_session_search_fts5_sync() -> None:
    """Test FTS5 table sync mechanism."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Skip if FTS5 not available
        with SessionSearch(db_path) as search:
            if not search.fts5_available():
                pytest.skip("FTS5 not available")
        
        # Create messages
        with SessionStore(db_path) as store:
            session_id = store.create_session("test")
            store.append_message(session_id, Message(role="user", content="First message"))
        
        # Search should sync and find it
        with SessionSearch(db_path) as search:
            if search.fts5_available():
                results = search.search("First", limit=5)
                assert len(results) >= 1
                
                # Add another message
                with SessionStore(db_path) as store:
                    store.append_message(session_id, Message(role="assistant", content="Second message"))
                
                # Should find the new message too
                results2 = search.search("Second", limit=5)
                assert len(results2) >= 1


def test_is_code_like_heuristic() -> None:
    """Test the code detection heuristic."""
    from synth.session_plus import SessionExporter
    
    exporter = SessionExporter()
    
    # Test code-like patterns
    assert exporter._is_code_like("def hello():")
    assert exporter._is_code_like("import os")
    assert exporter._is_code_like("from pathlib import Path")
    assert exporter._is_code_like("# This is a comment")
    assert exporter._is_code_like("$ ls -la")
    assert exporter._is_code_like(">>> print('hello')")
    assert exporter._is_code_like('{"key": "value"}')
    assert exporter._is_code_like("[1, 2, 3]")
    
    # Test non-code
    assert not exporter._is_code_like("Hello world")
    assert not exporter._is_code_like("This is regular text.")
    assert not exporter._is_code_like("")
    assert not exporter._is_code_like(None)


def test_session_exporter_all_formats_missing_session() -> None:
    """Test that all export methods raise for missing session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with SessionExporter(db_path) as exporter:
            out_path = Path(tmpdir) / "out"
            
            with pytest.raises(SessionPlusError):
                exporter.export_json("nonexistent", out_path.with_suffix(".json"))
            
            with pytest.raises(SessionPlusError):
                exporter.export_markdown("nonexistent", out_path.with_suffix(".md"))
            
            with pytest.raises(SessionPlusError):
                exporter.export_html("nonexistent", out_path.with_suffix(".html"))


def test_build_exit_summary_malformed_tool_calls() -> None:
    """Test summary handles malformed tool_calls JSON gracefully."""
    messages = [
        Message(role="assistant", content="Test", tool_calls="not valid json"),
        Message(role="assistant", content="Test2", tool_calls='{"not": "a list"}'),
    ]
    # Should not crash
    summary = build_exit_summary(messages)
    assert "Session summary" in summary