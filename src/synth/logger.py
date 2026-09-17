"""Structured logging system for Synth.

Features:
- Log levels: DEBUG, INFO, WARNING, ERROR
- JSON output format with timestamp, level, message, context
- File rotation (max 10MB, keep 5 files)
- Context manager for session-scoped logs
- Integration points: agent loop, tool calls, LLM requests

Usage:
    from synth.logger import Logger, LogContext
    
    # Basic usage
    logger = Logger(name="agent", log_file="agent.log")
    logger.info("Agent started", {"session_id": "abc123"})
    
    # Context manager for session-scoped logging
    with LogContext(logger, session_id="abc123"):
        logger.debug("Processing tool call", {"tool": "read_file", "path": "test.py"})
"""

from __future__ import annotations

import json
import logging.handlers
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Log levels matching Python's logging module
class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class Logger:
    """Structured JSON logger with file rotation."""
    
    # Rotation settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Union[str, Path]] = None,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = False,
    ) -> None:
        """Initialize logger.
        
        Args:
            name: Logger name for context tracking
            log_file: Path to log file (optional)
            level: Minimum log level
            enable_console: Whether to also log to console
        """
        self.name = name
        self.level = level
        self._context_stack: list[Dict[str, Any]] = []
        
        # Configure handlers
        self.handlers = []
        
        # File handler with rotation
        if log_file:
            self._setup_file_handler(log_file)
        
        # Optional console handler
        if enable_console:
            self._setup_console_handler()
    
    def _setup_file_handler(self, log_file: Union[str, Path]) -> None:
        """Set up rotating file handler."""
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use logging.RotatingFileHandler for rotation
        handler = logging.handlers.RotatingFileHandler(
            str(log_path),
            maxBytes=self.MAX_FILE_SIZE,
            backupCount=self.BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON
        self.handlers.append(handler)
    
    def _setup_console_handler(self) -> None:
        """Set up console handler for development/debugging."""
        import sys
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.handlers.append(handler)
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message at given level should be logged."""
        return level >= self.level
    
    def _format_message(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format log message as JSON."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Merge contexts from bottom of stack up
        merged_context: Dict[str, Any] = {}
        for ctx in self._context_stack:
            merged_context.update(ctx)
        
        # Add current context
        if context:
            merged_context.update(context)
        
        # Build log entry
        entry = {
            "timestamp": timestamp,
            "level": level.name,
            "logger": self.name,
            "message": message,
            "context": merged_context,
        }
        
        # Add extra fields
        if extra:
            entry["extra"] = extra
        
        # Add thread info for debugging
        entry["thread"] = threading.current_thread().name
        
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    
    def _write_to_handlers(self, formatted_message: str) -> None:
        """Write formatted message to all handlers."""
        for handler in self.handlers:
            # Create a log record and handle it
            record = logging.LogRecord(
                name=self.name,
                level=logging.INFO,  # Level doesn't matter for formatting
                pathname="",
                lineno=0,
                msg=formatted_message,
                args=(),
                exc_info=None,
            )
            handler.handle(record)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message at specified level."""
        if not self._should_log(level):
            return
        
        formatted = self._format_message(level, message, context, extra)
        self._write_to_handlers(formatted)
    
    def debug(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log at DEBUG level."""
        self.log(LogLevel.DEBUG, message, context, extra)
    
    def info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log at INFO level."""
        self.log(LogLevel.INFO, message, context, extra)
    
    def warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log at WARNING level."""
        self.log(LogLevel.WARNING, message, context, extra)
    
    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None,
    ) -> None:
        """Log at ERROR level."""
        extra_dict = extra or {}
        if exc_info:
            extra_dict["exception"] = str(exc_info)
            extra_dict["exception_type"] = type(exc_info).__name__
        self.log(LogLevel.ERROR, message, context, extra_dict)
    
    @contextmanager
    def push_context(self, **context: Any):
        """Add context to the stack for duration of block."""
        self._context_stack.append(context)
        try:
            yield
        finally:
            self._context_stack.pop()
    
    def set_level(self, level: LogLevel) -> None:
        """Change the minimum log level."""
        self.level = level


class LogContext:
    """Context manager for session-scoped logging."""
    
    def __init__(self, logger: Logger, **context: Any):
        """Initialize with logger and context."""
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        """Enter context - push context onto stack."""
        self.logger._context_stack.append(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - pop context from stack."""
        self.logger._context_stack.pop()
        if exc_val:
            self.logger.error(
                f"Context exited with exception: {exc_val}",
                context=self.context,
                exc_info=exc_val,
            )


# Global logger instances
_LOGGERS: Dict[str, Logger] = {}
_DEFAULT_LOG_FILE = Path.home() / ".synth" / "logs" / "synth.log"


def get_logger(
    name: str = "synth",
    log_file: Optional[Union[str, Path]] = None,
    level: LogLevel = LogLevel.INFO,
) -> Logger:
    """Get or create a logger instance."""
    if name not in _LOGGERS:
        if log_file is None:
            log_file = _DEFAULT_LOG_FILE
        _LOGGERS[name] = Logger(name, log_file, level)
    return _LOGGERS[name]


# Integration helpers
def log_agent_iteration(
    logger: Logger,
    iteration: int,
    max_iterations: int,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log agent loop iteration."""
    logger.debug(
        f"Agent iteration {iteration}/{max_iterations}",
        context={"iteration": iteration, "max_iterations": max_iterations, **(context or {})},
    )


def log_tool_call(
    logger: Logger,
    tool_name: str,
    tool_args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log tool call execution."""
    logger.info(
        f"Tool call: {tool_name}",
        context={
            "tool": tool_name,
            "args": tool_args,
            **(context or {}),
        },
    )


def log_tool_result(
    logger: Logger,
    tool_name: str,
    result: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log tool result."""
    # Truncate very long results
    if len(result) > 1000:
        result_preview = result[:1000] + "..."
        extra = {"result_preview": result_preview, "result_length": len(result)}
    else:
        extra = None
    
    logger.debug(
        f"Tool result: {tool_name}",
        context={
            "tool": tool_name,
            "result_length": len(result),
            **(context or {}),
        },
        extra=extra,
    )


def log_llm_request(
    logger: Logger,
    request_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log LLM request."""
    logger.info(
        f"LLM request: {request_type}",
        context={"request_type": request_type, **(context or {})},
    )


def log_llm_response(
    logger: Logger,
    request_type: str,
    has_tool_calls: bool,
    response_length: int,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log LLM response."""
    logger.debug(
        f"LLM response: {request_type}",
        context={
            "request_type": request_type,
            "has_tool_calls": has_tool_calls,
            "response_length": response_length,
            **(context or {}),
        },
    )