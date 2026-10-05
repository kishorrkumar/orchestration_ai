# ADR 0004: RFC 9457 Problem Details Standard for HTTP APIs

## Context
Default FastAPI error responses (`{"detail": ...}`) lack standardized error codes, classification URLs, parameter location lists, and correlation IDs for observability.

## Decision
We adopted **RFC 9457 Problem Details for HTTP APIs** (`application/problem+json`) across all v2 REST endpoints:
- Every error includes a machine-readable `code`, user-readable `title` and `detail`, status code, and prefixed `error_id` (`err_...`).
- Validation errors include an `invalid_params` array with exact parameter path and message.
- Internal exceptions (500) never leak internal python tracebacks to clients; full details are logged securely with correlation tracking.

## Consequences
- **Positive:** Predictable client error handling and enterprise-grade API compliance.
