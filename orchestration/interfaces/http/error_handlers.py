"""
RFC 9457 Problem Details exception handlers for FastAPI.
Formats all domain, validation, and unexpected 500 errors into application/problem+json.
Prevents internal stack traces from leaking to clients.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from orchestration.shared.errors import DomainError, ProblemDetail
from orchestration.shared.ids import new_error_id
from orchestration.shared.logging import get_logger

logger = get_logger("orchestration.interfaces.http")


def register_error_handlers(app: FastAPI) -> None:
    """Register RFC 9457 exception handlers on the FastAPI application."""

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        problem = exc.to_problem()
        logger.warning(
            "Domain error occurred",
            error_id=exc.error_id,
            code=exc.code,
            status=exc.status_code,
            detail=exc.detail,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=problem.status,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        error_id = new_error_id()
        invalid_params = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            invalid_params.append({
                "name": loc,
                "reason": err.get("msg", "Invalid parameter value"),
                "type": err.get("type", "value_error"),
            })

        reasons = [f"{p['name']}: {p['reason']}" for p in invalid_params]
        detail_msg = f"Validation failed: {'; '.join(reasons)}" if reasons else "The request body or query parameters failed schema validation."

        problem = ProblemDetail(
            type="https://errors.personaplex.ai/validation-error",
            title="Validation Failed",
            status=422,
            detail=detail_msg,
            code="VALIDATION_ERROR",
            error_id=error_id,
            invalid_params=invalid_params,
        )

        logger.info(
            "Request validation failed",
            error_id=error_id,
            invalid_params=invalid_params,
            path=str(request.url),
        )

        return JSONResponse(
            status_code=problem.status,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_id = new_error_id()
        problem = ProblemDetail(
            type=f"https://errors.personaplex.ai/http-{exc.status_code}",
            title=exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            status=exc.status_code,
            detail=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
            error_id=error_id,
        )

        return JSONResponse(
            status_code=problem.status,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = new_error_id()
        logger.error(
            "Unhandled server exception",
            error_id=error_id,
            exc_info=exc,
            path=str(request.url),
        )

        problem = ProblemDetail(
            type="https://errors.personaplex.ai/internal-server-error",
            title="Internal Server Error",
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred. Please reference error ID '{error_id}' when contacting support.",
            code="INTERNAL_SERVER_ERROR",
            error_id=error_id,
        )

        return JSONResponse(
            status_code=problem.status,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
        )
