"""
Domain protocols defining abstract interfaces for repositories, clock, and tokenizer.
Enforces Dependency Inversion: domain depends on zero external libraries or databases.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .agent import Agent, AgentVersion
    from .session import CallSession, CallTurn


@runtime_checkable
class Clock(Protocol):
    """Protocol for abstracting time generation for deterministic testing."""

    def now_utc(self) -> datetime:
        """Return current UTC datetime."""
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for counting discrete BPE tokens."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in given text."""
        ...


@runtime_checkable
class AgentRepository(Protocol):
    """Protocol for Agent persistence operations."""

    async def get_by_id(self, agent_id: str) -> Agent | None:
        ...

    async def list_agents(self, limit: int = 50, offset: int = 0) -> list[Agent]:
        ...

    async def save(self, agent: Agent) -> Agent:
        ...

    async def delete(self, agent_id: str) -> bool:
        ...

    async def get_version(self, agent_id: str, version_number: int) -> AgentVersion | None:
        ...

    async def list_versions(self, agent_id: str) -> list[AgentVersion]:
        ...

    async def save_version(self, version: AgentVersion) -> AgentVersion:
        ...


@runtime_checkable
class CallSessionRepository(Protocol):
    """Protocol for CallSession and turn logging persistence."""

    async def get_by_id(self, session_id: str) -> CallSession | None:
        ...

    async def list_sessions(self, agent_id: str | None = None, limit: int = 50, offset: int = 0) -> list[CallSession]:
        ...

    async def save_session(self, session: CallSession) -> CallSession:
        ...

    async def save_turn(self, turn: CallTurn) -> CallTurn:
        ...

    async def list_turns(self, session_id: str) -> list[CallTurn]:
        ...


@runtime_checkable
class SpeechToSpeechEngine(Protocol):
    """Protocol for upstream speech-to-speech inference worker."""

    async def is_ready(self) -> bool:
        ...

    async def acquire_stream(self) -> str:
        """Acquire a dedicated stream on a worker, returning the session token."""
        ...

    async def release_stream(self, stream_token: str) -> None:
        """Release dedicated stream back to the worker pool."""
        ...
