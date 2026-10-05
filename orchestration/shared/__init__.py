"""Shared foundational utilities: errors, logging, settings, and identifiers."""

from .errors import (
    AuthenticationError,
    DomainError,
    NotFoundError,
    ProblemDetail,
    PromptTooLongError,
    RateLimitError,
    UpstreamUnavailableError,
    ValidationError,
    WorkerBusyError,
)
from .ids import new_agent_id, new_error_id, new_session_id, new_turn_id
from .logging import get_logger, setup_logging
from .settings import Settings, settings

__all__ = [
    "AuthenticationError",
    "DomainError",
    "NotFoundError",
    "ProblemDetail",
    "PromptTooLongError",
    "RateLimitError",
    "Settings",
    "UpstreamUnavailableError",
    "ValidationError",
    "WorkerBusyError",
    "get_logger",
    "new_agent_id",
    "new_error_id",
    "new_session_id",
    "new_turn_id",
    "settings",
    "setup_logging",
]
