"""Pydantic v2 schemas for CallSession and turn logging."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StartCallRequest(BaseModel):
    agent_id: str = Field(..., json_schema_extra={"example": "agt_0123456789abcdef"})
    client_type: str = Field(default="web", json_schema_extra={"example": "web"})


class CallTurnResponse(BaseModel):
    id: str
    turn_index: int
    speaker: str
    text: str
    started_at_sec: float
    ended_at_sec: float
    latency_ms: float | None = None
    created_at: datetime


class CallSessionResponse(BaseModel):
    id: str
    agent_id: str
    agent_version: int
    status: str
    client_type: str
    duration_sec: float
    end_reason: str | None
    total_turns: int
    created_at: datetime
    ended_at: datetime | None
