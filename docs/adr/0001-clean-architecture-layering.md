# ADR 0001: Clean Layered Architecture & Domain Decoupling

## Context
The legacy implementation mixed database ORM entities, FastAPI routing, and voice session loops into monolithic files. Business invariants (such as prompt delimiters, timezone calculations, and token limit checks) were scattered across endpoint handlers.

## Decision
We refactored the entire system into five decoupled layers:
1. `domain/`: Pure python entities and value objects with zero framework dependencies.
2. `application/`: Explicit use cases and orchestration services.
3. `infrastructure/`: SQLAlchemy async repositories, SentencePiece tokenizers, and system clocks.
4. `interfaces/`: FastAPI REST controllers and WebSocket endpoints.
5. `shared/`: Cross-cutting errors, logging, configuration, and ID generators.

## Consequences
- **Positive:** Domain logic can be unit-tested without databases, websockets, or GPU hardware.
- **Positive:** Strict typing (`mypy --strict`) enforced cleanly on domain and application layers.
- **Trade-off:** Requires explicit mapping between ORM models and domain aggregates.
