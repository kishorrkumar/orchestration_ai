"""
Unified Voice Engine Protocol and Shared Session Contract.
Both Engine A (PersonaPlex S2S) and Engine B (Cascaded Cloud Stack) implement VoiceEngine.
Ensures 100% wire parity over /v2/voice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from fastapi import WebSocket

from orchestration.db.models import Agent, AgentVersion, CallSession
from orchestration.domain.engines.spec import EngineType


@dataclass
class SessionContext:
    websocket: WebSocket
    agent: Agent
    version: AgentVersion | None
    session_record: CallSession
    session_factory: Any
    client_sample_rate: int = 16000
    codec: str = "pcm16"
    template_vars: Dict[str, str] = field(default_factory=dict)
    workspace_id: str = "wks_default"


@runtime_checkable
class VoiceEngine(Protocol):
    """Protocol satisfied by both PersonaPlexVoiceEngine and CascadedVoiceEngine."""

    @property
    def engine_type(self) -> EngineType:
        """Returns the engine type tag."""
        ...

    async def run_session(self, context: SessionContext) -> None:
        """Runs the real-time full-duplex session loop until disconnect or call end."""
        ...
