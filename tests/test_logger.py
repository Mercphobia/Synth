"""Comprehensive tests for synth.logger module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth.logger import (
    LogLevel,
    Logger,
    LogContext,
    get_logger,
    log_agent_iteration,
    log_tool_call,
    log_tool_result,
    log_llm_request,
    log_llm_response,
)


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file path."""
    return tmp_path / "test.log"


@pytest.fixture
def logger(temp_log_file):
    """Create a logger instance with temporary file."""
    return Logger(name="test", log_file=temp_log_file, level=LogLevel.DEBUG)


class TestLoggerCreation:
    """Test logger initialization and configuration."""
    
    def test_create_logger_with_file(self, temp_log_file):
        """Test creating logger with file handler."""
        logger = Logger(name="test", log_file=temp_log_file)
        assert logger.name == "test"
        assert logger.level == LogLevel.INFO
        assert len(logger.handlers) == 1
        
    def test_create_logger_with_console(self):
        """Test creating logger with console handler."""
        logger = Logger(name="test", enable_console=True)
        assert len(logger.handlers) == 1
        
    def test_create_logger_with_both_handlers(self, temp_log_file):
        """Test creating logger with both file and console handlers."""
        logger = Logger(
            name="test",
            log_file=temp_log_file,
            enable_console=True
        )
        assert len(logger.handlers) == 2
    
    def test_create_logger_without_handlers(self):
        """Test creating logger without any handlers."""
        logger = Logger(name="test")
        assert len(logger.handlers) == 0


class TestLogLevels:
    """Test log level filtering."""
    
    def test_debug_level_filtering(self, logger, temp_log_file):
        """Test that DEBUG messages are logged when level is DEBUG."""
        logger.debug("Debug message")
        
        content = temp_log_file.read_text()
        assert "Debug message" in content
        assert "DEBUG" in content
    
    def test_info_level_filtering(self, logger, temp_log_file):
        """Test that INFO messages are logged."""
        logger.info("Info message")
        
        content = temp_log_file.read_text()
        assert "Info message" in content
        assert "INFO" in content
    
    def test_warning_level_filtering(self, logger, temp_log_file):
        """Test that WARNING messages are logged."""
        logger.warning("Warning message")
        
        content = temp_log_file.read_text()
        assert "Warning message" in content
        assert "WARNING" in content
    
    def test_error_level_filtering(self, logger, temp_log_file):
        """Test that ERROR messages are logged."""
        logger.error("Error message")
        
        content = temp_log_file.read_text()
        assert "Error message" in content
        assert "ERROR" in content
    
    def test_level_filtering_blocks_lower_levels(self, temp_log_file):
        """Test that lower level messages are filtered out."""
        logger = Logger(name="test", log_file=temp_log_file, level=LogLevel.WARNING)
        
        logger.debug("Debug should not appear")
        logger.info("Info should not appear")
        logger.warning("Warning should appear")
        logger.error("Error should appear")
        
        content = temp_log_file.read_text()
        assert "Debug should not appear" not in content
        assert "Info should not appear" not in content
        assert "Warning should appear" in content
        assert "Error should appear" in content
    
    def test_change_log_level(self, logger, temp_log_file):
        """Test changing log level dynamically."""
        logger.set_level(LogLevel.ERROR)
        
        logger.info("Info should not appear")
        logger.error("Error should appear")
        
        content = temp_log_file.read_text()
        assert "Info should not appear" not in content
        assert "Error should appear" in content


class TestJSONFormat:
    """Test JSON output format validation."""
    
    def test_json_format_valid(self, logger, temp_log_file):
        """Test that log entries are valid JSON."""
        logger.info("Test message")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert isinstance(entry, dict)
        assert "timestamp" in entry
        assert "level" in entry
        assert "message" in entry
        assert "context" in entry
        assert "thread" in entry
    
    def test_json_format_timestamp(self, logger, temp_log_file):
        """Test timestamp format in JSON."""
        logger.info("Test message")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        timestamp = entry["timestamp"]
        assert timestamp.endswith("Z")
        assert "T" in timestamp  # ISO format
    
    def test_json_format_level(self, logger, temp_log_file):
        """Test level field in JSON."""
        logger.warning("Test message")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["level"] == "WARNING"
    
    def test_json_format_context(self, logger, temp_log_file):
        """Test context field in JSON."""
        logger.info("Test message", context={"key": "value", "number": 42})
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["context"]["key"] == "value"
        assert entry["context"]["number"] == 42
    
    def test_json_format_extra(self, logger, temp_log_file):
        """Test extra field in JSON."""
        logger.info("Test message", extra={"extra_key": "extra_value"})
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["extra"]["extra_key"] == "extra_value"
    
    def test_json_format_thread(self, logger, temp_log_file):
        """Test thread field in JSON."""
        logger.info("Test message")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "thread" in entry
        assert isinstance(entry["thread"], str)


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_basic(self, logger, temp_log_file):
        """Test basic context manager usage."""
        with LogContext(logger, session_id="abc123"):
            logger.info("Inside context")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["context"]["session_id"] == "abc123"
    
    def test_context_manager_nested(self, logger, temp_log_file):
        """Test nested context managers."""
        with LogContext(logger, session_id="abc123"):
            with LogContext(logger, user="testuser"):
                logger.info("Inside nested context")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["context"]["session_id"] == "abc123"
        assert entry["context"]["user"] == "testuser"
    
    def test_context_manager_cleanup(self, logger, temp_log_file):
        """Test that context is cleaned up after exit."""
        with LogContext(logger, session_id="abc123"):
            logger.info("Inside context")
        
        logger.info("Outside context")
        
        lines = temp_log_file.read_text().strip().split("\n")
        inside = json.loads(lines[0])
        outside = json.loads(lines[1])
        
        assert inside["context"]["session_id"] == "abc123"
        assert "session_id" not in outside["context"]
    
    def test_push_context(self, logger, temp_log_file):
        """Test push_context method."""
        with logger.push_context(session_id="abc123"):
            logger.info("Inside push_context")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["context"]["session_id"] == "abc123"
    
    def test_context_manager_with_exception(self, logger, temp_log_file):
        """Test context manager handles exceptions."""
        with pytest.raises(ValueError):
            with LogContext(logger, session_id="abc123"):
                logger.info("Before exception")
                raise ValueError("Test exception")
        
        # Should have 2 log entries: info + error
        lines = temp_log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        
        error_entry = json.loads(lines[1])
        assert error_entry["level"] == "ERROR"
        assert "Test exception" in error_entry["message"]


class TestFileRotation:
    """Test file rotation functionality."""
    
    def test_rotation_creates_backup(self, temp_log_file):
        """Test that rotation creates backup files."""
        # Create logger with small max size for testing
        logger = Logger(name="test", log_file=temp_log_file, level=LogLevel.DEBUG)
        
        # Override the handler's max size for testing
        if logger.handlers:
            handler = logger.handlers[0]
            handler.maxBytes = 1000  # 1KB for testing
        
        # Write enough data to trigger rotation
        for i in range(100):
            logger.info(f"Log message {i}" * 20)
        
        # Check that backup files were created
        assert temp_log_file.exists()
        
        # Look for backup files
        backup_files = list(temp_log_file.parent.glob("test.log.*"))
        assert len(backup_files) > 0 or temp_log_file.stat().st_size < 1000
    
    def test_rotation_keeps_backup_count(self, temp_log_file):
        """Test that rotation keeps only specified number of backups."""
        logger = Logger(name="test", log_file=temp_log_file, level=LogLevel.DEBUG)
        
        if logger.handlers:
            handler = logger.handlers[0]
            handler.maxBytes = 500  # 500 bytes for testing
            handler.backupCount = 3
        
        # Write lots of data
        for i in range(200):
            logger.info(f"Log message {i}" * 30)
        
        # Count backup files
        backup_files = list(temp_log_file.parent.glob("test.log.*"))
        assert len(backup_files) <= 3


class TestIntegrationHelpers:
    """Test integration helper functions."""
    
    def test_log_agent_iteration(self, logger, temp_log_file):
        """Test log_agent_iteration helper."""
        log_agent_iteration(logger, iteration=5, max_iterations=20)
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "Agent iteration 5/20" in entry["message"]
        assert entry["context"]["iteration"] == 5
        assert entry["context"]["max_iterations"] == 20
    
    def test_log_tool_call(self, logger, temp_log_file):
        """Test log_tool_call helper."""
        log_tool_call(logger, "read_file", {"path": "test.py"})
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "Tool call: read_file" in entry["message"]
        assert entry["context"]["tool"] == "read_file"
        assert entry["context"]["args"]["path"] == "test.py"
    
    def test_log_tool_result(self, logger, temp_log_file):
        """Test log_tool_result helper."""
        log_tool_result(logger, "read_file", "File content here")
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "Tool result: read_file" in entry["message"]
        assert entry["context"]["tool"] == "read_file"
        assert entry["context"]["result_length"] == 17
    
    def test_log_tool_result_truncation(self, logger, temp_log_file):
        """Test that long tool results are truncated."""
        long_result = "x" * 2000
        log_tool_result(logger, "read_file", long_result)
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["extra"]["result_length"] == 2000
        assert entry["extra"]["result_preview"].endswith("...")
    
    def test_log_llm_request(self, logger, temp_log_file):
        """Test log_llm_request helper."""
        log_llm_request(logger, "chat", context={"model": "gpt-4"})
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "LLM request: chat" in entry["message"]
        assert entry["context"]["request_type"] == "chat"
        assert entry["context"]["model"] == "gpt-4"
    
    def test_log_llm_response(self, logger, temp_log_file):
        """Test log_llm_response helper."""
        log_llm_response(
            logger,
            "chat",
            has_tool_calls=True,
            response_length=500
        )
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert "LLM response: chat" in entry["message"]
        assert entry["context"]["has_tool_calls"] is True
        assert entry["context"]["response_length"] == 500


class TestGetLogger:
    """Test get_logger factory function."""
    
    def test_get_logger_creates_instance(self, temp_log_file):
        """Test that get_logger creates a logger instance."""
        logger = get_logger("test_factory", temp_log_file)
        
        assert isinstance(logger, Logger)
        assert logger.name == "test_factory"
    
    def test_get_logger_reuses_instance(self, temp_log_file):
        """Test that get_logger reuses existing instances."""
        logger1 = get_logger("test_reuse", temp_log_file)
        logger2 = get_logger("test_reuse", temp_log_file)
        
        assert logger1 is logger2
    
    def test_get_logger_different_names(self, temp_log_file):
        """Test that different names create different instances."""
        logger1 = get_logger("test1", temp_log_file)
        logger2 = get_logger("test2", temp_log_file)
        
        assert logger1 is not logger2
    
    def test_get_logger_default_log_file(self):
        """Test that get_logger uses default log file when none specified."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock the default log file path
            from synth.logger import _DEFAULT_LOG_FILE
            original_default = _DEFAULT_LOG_FILE
            try:
                default_log_file = Path(tmp_dir) / "default.log"
                # Patch the module-level variable
                import synth.logger
                synth.logger._DEFAULT_LOG_FILE = default_log_file
                
                logger = get_logger("test_default")
                
                assert isinstance(logger, Logger)
                assert logger.name == "test_default"
                # Verify it's using the default file
                assert len(logger.handlers) == 1
                handler = logger.handlers[0]
                assert str(default_log_file) in str(handler.baseFilename)
            finally:
                synth.logger._DEFAULT_LOG_FILE = original_default


class TestErrorLogging:
    """Test error logging with exception info."""
    
    def test_error_with_exception(self, logger, temp_log_file):
        """Test logging error with exception info."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Something went wrong", exc_info=e)
        
        content = temp_log_file.read_text().strip()
        entry = json.loads(content)
        
        assert entry["level"] == "ERROR"
        assert "Something went wrong" in entry["message"]
        assert entry["extra"]["exception"] == "Test error"
        assert entry["extra"]["exception_type"] == "ValueError"


class TestMultipleLoggers:
    """Test multiple logger instances."""
    
    def test_multiple_loggers_separate_files(self, tmp_path):
        """Test that multiple loggers write to separate files."""
        log_file1 = tmp_path / "log1.log"
        log_file2 = tmp_path / "log2.log"
        
        logger1 = Logger(name="logger1", log_file=log_file1)
        logger2 = Logger(name="logger2", log_file=log_file2)
        
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")
        
        content1 = log_file1.read_text()
        content2 = log_file2.read_text()
        
        assert "logger1" in content1
        assert "Message from logger2" not in content1
        assert "logger2" in content2
        assert "Message from logger1" not in content2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])