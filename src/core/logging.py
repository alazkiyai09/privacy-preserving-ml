"""
FedPhish Structured Logging

Provides consistent JSON-structured logging across all projects.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Optional, Dict
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "pathname", "filename", "module", "lineno", "funcName",
            }:
                log_data[key] = value

        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper around logging.Logger with convenient methods."""

    def __init__(self, name: str, logger: logging.Logger):
        self._logger = logger
        self.name = name

    def _log_with_context(
        self,
        level: int,
        msg: str,
        context: Optional[dict[str, Any]] = None,
        exc_info: Optional[bool] = None,
    ):
        """Log a message with optional context dictionary."""
        if context:
            extra = {**context}
            self._logger.log(level, msg, extra=extra, exc_info=exc_info)
        else:
            self._logger.log(level, msg, exc_info=exc_info)

    def debug(self, msg: str, **context):
        self._log_with_context(logging.DEBUG, msg, context)

    def info(self, msg: str, **context):
        self._log_with_context(logging.INFO, msg, context)

    def warning(self, msg: str, **context):
        self._log_with_context(logging.WARNING, msg, context)

    def error(self, msg: str, exc_info: bool = False, **context):
        self._log_with_context(logging.ERROR, msg, context, exc_info=exc_info)

    def critical(self, msg: str, exc_info: bool = False, **context):
        self._log_with_context(logging.CRITICAL, msg, context, exc_info=exc_info)

    def exception(self, msg: str, **context):
        self._log_with_context(logging.ERROR, msg, context, exc_info=True)


_loggers: dict[str, StructuredLogger] = {}
_configured = False


def configure_logging(
    level: int | str = logging.INFO,
    log_file: Optional[str | Path] = None,
    json_output: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for all FedPhish loggers.

    Args:
        level: Logging level (e.g., logging.INFO, "INFO")
        log_file: Optional path to log file
        json_output: If True, use JSON formatting; otherwise use text format
        format_string: Optional custom format string for text mode
    """
    global _configured

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(level)

    # Set formatter
    if json_output:
        handler.setFormatter(StructuredFormatter())
    elif format_string:
        handler.setFormatter(logging.Formatter(format_string))
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.addHandler(handler)
    _configured = True


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger with the given name.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        StructuredLogger instance
    """
    if name in _loggers:
        return _loggers[name]

    if not _configured:
        configure_logging()

    base_logger = logging.getLogger(name)
    structured_logger = StructuredLogger(name, base_logger)
    _loggers[name] = structured_logger
    return structured_logger
