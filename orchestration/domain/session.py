"""
Domain entities for real-time speech-to-speech call sessions and turn tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class CallStatus(StrEnum):
    INITIATING = "initiating"
    CONNECTED = "connected"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"


class TurnSpeaker(StrEnum):
    CALLER = "caller"
    AGENT = "agent"
    SYSTEM = "system"


class EndReason(StrEnum):
    AGENT_CLOSED = "agent_closed"
    CALLER_HUNG_UP = "caller_hung_up"
    SILENCE_TIMEOUT = "silence_timeout"
    MAX_DURATION = "max_duration"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class CallTurn:
    """A single chronological turn of spoken dialogue in a call session."""
    id: str
    session_id: str
    turn_index: int
    speaker: TurnSpeaker
    text: str
    started_at_sec: float
    ended_at_sec: float
    latency_ms: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CallSession:
    """Core domain entity tracking a live or historical phone call."""
    id: str
    agent_id: str
    agent_version: int
    status: CallStatus = CallStatus.INITIATING
    client_type: str = "web"  # "web" (16kHz) or "telephony" (8kHz G.711)
    duration_sec: float = 0.0
    end_reason: EndReason | None = None
    total_turns: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
