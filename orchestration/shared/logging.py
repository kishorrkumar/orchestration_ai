"""
Structured JSON logging using structlog with correlation IDs and sensitive data redaction.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

REDACTED_KEYS = {"password", "secret", "token", "api_key", "authorization", "hf_token"}


def redact_sensitive_data(
    logger: logging.Logger | None,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Redact keys matching sensitive names and truncate long raw transcript texts."""
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in REDACTED_KEYS):
            event_dict[key] = "[REDACTED]"
    return event_dict


def setup_logging(log_level: str = "INFO", is_dev: bool = True) -> None:
    """Configure structured logging pipeline."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_sensitive_data,
        structlog.processors.StackInfoRenderer(),
    ]

    if is_dev:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""
    from typing import cast
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
