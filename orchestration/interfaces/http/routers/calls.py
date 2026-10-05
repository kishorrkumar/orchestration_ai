"""FastAPI REST controller for Call history and turn transcripts (/v2/calls)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from orchestration.application.calls.service import CallApplicationService
from orchestration.interfaces.http.dependencies import get_call_service
from orchestration.interfaces.http.schemas.call_schemas import (
    CallSessionResponse,
    CallTurnResponse,
)

router = APIRouter(prefix="/v2/calls", tags=["Calls"])


@router.get("", response_model=list[CallSessionResponse])
async def list_calls(
    agent_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    svc: CallApplicationService = Depends(get_call_service),
) -> list[CallSessionResponse]:
    """List recent call sessions, optionally filtered by agent_id."""
    sessions = await svc.list_sessions(agent_id=agent_id, limit=limit, offset=offset)
    return [
        CallSessionResponse(
            id=s.id,
            agent_id=s.agent_id,
            agent_version=s.agent_version,
            status=s.status.value,
            client_type=s.client_type,
            duration_sec=s.duration_sec,
            end_reason=s.end_reason.value if s.end_reason else None,
            total_turns=s.total_turns,
            created_at=s.created_at,
            ended_at=s.ended_at,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=CallSessionResponse)
async def get_call(
    session_id: str,
    svc: CallApplicationService = Depends(get_call_service),
) -> CallSessionResponse:
    """Fetch call session summary by ID."""
    s = await svc.get_session(session_id)
    return CallSessionResponse(
        id=s.id,
        agent_id=s.agent_id,
        agent_version=s.agent_version,
        status=s.status.value,
        client_type=s.client_type,
        duration_sec=s.duration_sec,
        end_reason=s.end_reason.value if s.end_reason else None,
        total_turns=s.total_turns,
        created_at=s.created_at,
        ended_at=s.ended_at,
    )


@router.get("/{session_id}/turns", response_model=list[CallTurnResponse])
async def list_call_turns(
    session_id: str,
    svc: CallApplicationService = Depends(get_call_service),
) -> list[CallTurnResponse]:
    """Fetch chronological turn-by-turn dialogue transcript for a call."""
    turns = await svc.list_turns(session_id)
    return [
        CallTurnResponse(
            id=t.id,
            turn_index=t.turn_index,
            speaker=t.speaker.value,
            text=t.text,
            started_at_sec=t.started_at_sec,
            ended_at_sec=t.ended_at_sec,
            latency_ms=t.latency_ms,
            created_at=t.created_at,
        )
        for t in turns
    ]
