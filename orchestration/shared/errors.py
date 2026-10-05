"""
Typed domain exception hierarchy and RFC 9457 Problem Details data model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .ids import new_error_id


class ProblemDetail(BaseModel):
    """
    RFC 9457 / RFC 7807 compliant problem details document.
    Never exposes internal stack traces to HTTP API clients.
    """
    type: str = Field(..., description="URI reference identifying problem type")
    title: str = Field(..., description="Short, human-readable summary of problem")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Human-readable explanation specific to this occurrence")
    code: str = Field(..., description="Stable programmatic application error code")
    error_id: str = Field(..., description="Correlation ID for server-side log lookup")
    invalid_params: list[dict[str, Any]] | None = Field(
        default=None, description="Validation error specifics if applicable"
    )


class DomainError(Exception):
    """Base class for all domain-level business rule violations."""
    code: str = "DOMAIN_ERROR"
    status_code: int = 400
    title: str = "Domain Rule Violation"

    def __init__(self, detail: str, *, error_id: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_id = error_id or new_error_id()

    def to_problem(self, base_url: str = "https://errors.personaplex.ai") -> ProblemDetail:
        return ProblemDetail(
            type=f"{base_url}/{self.code.lower().replace('_', '-')}",
            title=self.title,
            status=self.status_code,
            detail=self.detail,
            code=self.code,
            error_id=self.error_id,
        )


class NotFoundError(DomainError):
    code = "NOT_FOUND"
    status_code = 404
    title = "Resource Not Found"


class ValidationError(DomainError):
    code = "VALIDATION_ERROR"
    status_code = 422
    title = "Validation Failed"

    def __init__(
        self,
        detail: str,
        *,
        error_id: str | None = None,
        invalid_params: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail, error_id=error_id)
        self.invalid_params = invalid_params

    def to_problem(self, base_url: str = "https://errors.personaplex.ai") -> ProblemDetail:
        problem = super().to_problem(base_url)
        problem.invalid_params = self.invalid_params
        return problem


class PromptTooLongError(ValidationError):
    code = "PROMPT_TOO_LONG"
    status_code = 422
    title = "Prompt Exceeds Maximum Token Limit"


class WorkerBusyError(DomainError):
    code = "WORKER_BUSY"
    status_code = 503
    title = "Voice Workers Busy"


class UpstreamUnavailableError(DomainError):
    code = "UPSTREAM_UNAVAILABLE"
    status_code = 502
    title = "Upstream Model Engine Unavailable"


class RateLimitError(DomainError):
    code = "RATE_LIMIT_EXCEEDED"
    status_code = 429
    title = "Rate Limit Exceeded"


class AuthenticationError(DomainError):
    code = "UNAUTHORIZED"
    status_code = 401
    title = "Authentication Required"
