"""
Agent Management and Versioning Router.
Dual-compatible with both Lean S2S Platform and legacy PersonaPlex client contracts.
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Agent, AgentVersion
from ..db.service import AgentService
from ..db.session import get_db
from ..persona.registry import OFFICIAL_VOICE_PRESETS
from .schemas import (
    AgentCreateRequest,
    AgentPublishRequest,
    AgentResponse,
    AgentUpdateRequest,
    VersionResponse,
)

router = APIRouter(prefix="/v1/agents", tags=["Agents"])


def _agent_to_dict(agent: Agent) -> dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "status": agent.status,
        "published_version_id": agent.published_version_id,
        "current_version_no": agent.current_version_no,
        "draft_voice_id": agent.draft_voice_id,
        "draft_greeting_text": agent.draft_greeting_text,
        "draft_greeting_mode": agent.draft_greeting_mode,
        "draft_system_prompt": agent.draft_system_prompt,
        "draft_ending_text": agent.draft_ending_text,
        "draft_end_silence_sec": agent.draft_end_silence_sec,
        "draft_max_duration_sec": agent.draft_max_duration_sec,
        "draft_timezone": agent.draft_timezone,
        # Legacy compatibility aliases
        "voice_prompt": agent.draft_voice_id,
        "text_prompt": agent.draft_system_prompt,
        "system_prompt": agent.draft_system_prompt,
        "created_at": agent.created_at.isoformat() if agent.created_at else None,
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
        "versions_count": len(agent.__dict__["versions"]) if "versions" in agent.__dict__ and agent.__dict__["versions"] else 0,
    }


def _version_to_dict(ver: AgentVersion) -> dict[str, Any]:
    return {
        "id": ver.id,
        "agent_id": ver.agent_id,
        "version_no": ver.version_no,
        "voice_id": ver.voice_id,
        "greeting_text": ver.greeting_text,
        "greeting_mode": ver.greeting_mode,
        "system_prompt": ver.system_prompt,
        "ending_text": ver.ending_text,
        "end_silence_sec": ver.end_silence_sec,
        "max_duration_sec": ver.max_duration_sec,
        "timezone": ver.timezone,
        "compiled_token_count": ver.compiled_token_count,
        "change_note": ver.change_note,
        "created_at": ver.created_at.isoformat() if ver.created_at else None,
    }


@router.get("")
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by status (draft/published/archived)"),
    search: Optional[str] = Query(None, description="Search by agent name"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List voice agents with pagination and search."""
    service = AgentService(db)
    agents, total = await service.list_agents(status=status, search=search, limit=limit, offset=offset)
    serialized = [_agent_to_dict(a) for a in agents]
    return {
        "items": serialized,
        "agents": serialized,
        "personas": serialized,
        "voice_presets": OFFICIAL_VOICE_PRESETS,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """Create a new voice agent supporting both modern and legacy payload shapes."""
    service = AgentService(db)

    name = payload.get("name") or "New Agent"
    voice_id = payload.get("voice_id") or payload.get("voice_prompt") or "NATF0.pt"
    greeting_text = payload.get("greeting_text") or ""
    greeting_mode = payload.get("greeting_mode") or "agent_first"
    system_prompt = payload.get("system_prompt") or payload.get("text_prompt") or ""
    ending_text = payload.get("ending_text") or ""
    end_silence_sec = int(payload.get("end_silence_sec") or 20)
    max_duration_sec = int(payload.get("max_duration_sec") or 600)
    timezone = payload.get("timezone") or "Asia/Kolkata"
    auto_publish = bool(payload.get("auto_publish", True))

    agent = await service.create_agent(
        name=name,
        id=payload.get("id"),
        voice_id=voice_id,
        greeting_text=greeting_text,
        greeting_mode=greeting_mode,
        system_prompt=system_prompt,
        ending_text=ending_text,
        end_silence_sec=end_silence_sec,
        max_duration_sec=max_duration_sec,
        timezone=timezone,
        auto_publish=auto_publish,
    )
    res_dict = _agent_to_dict(agent)
    return {
        **res_dict,
        "status": "created",
        "agent_status": agent.status,
        "agent": res_dict,
        "persona": res_dict,
    }


@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full agent details and draft configuration."""
    service = AgentService(db)
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_dict(agent)


@router.patch("/{agent_id}")
@router.put("/{agent_id}")
async def update_agent(
    agent_id: str,
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """Autosave or update mutable working draft fields (supports both PATCH and PUT)."""
    service = AgentService(db)

    name = payload.get("name")
    voice_id = payload.get("voice_id") or payload.get("voice_prompt")
    greeting_text = payload.get("greeting_text")
    greeting_mode = payload.get("greeting_mode")
    system_prompt = payload.get("system_prompt") or payload.get("text_prompt")
    ending_text = payload.get("ending_text")
    end_silence_sec = int(payload["end_silence_sec"]) if "end_silence_sec" in payload and payload["end_silence_sec"] is not None else None
    max_duration_sec = int(payload["max_duration_sec"]) if "max_duration_sec" in payload and payload["max_duration_sec"] is not None else None
    timezone = payload.get("timezone")

    agent = await service.update_agent_draft(
        agent_id=agent_id,
        name=name,
        voice_id=voice_id,
        greeting_text=greeting_text,
        greeting_mode=greeting_mode,
        system_prompt=system_prompt,
        ending_text=ending_text,
        end_silence_sec=end_silence_sec,
        max_duration_sec=max_duration_sec,
        timezone=timezone,
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    res_dict = _agent_to_dict(agent)
    return {
        **res_dict,
        "status": "updated",
        "agent_status": agent.status,
        "agent": res_dict,
        "persona": res_dict,
    }


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete an agent and its version history."""
    service = AgentService(db)
    deleted = await service.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "agent_id": agent_id, "persona_id": agent_id}


@router.post("/{agent_id}/publish")
async def publish_agent(
    agent_id: str,
    req: AgentPublishRequest,
    db: AsyncSession = Depends(get_db),
):
    """Publish the current draft as an immutable version."""
    service = AgentService(db)
    try:
        ver = await service.publish_agent(agent_id=agent_id, change_note=req.change_note)
        if not ver:
            raise HTTPException(status_code=404, detail="Agent not found")
        return _version_to_dict(ver)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{agent_id}/versions")
async def list_versions(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all immutable version records for an agent."""
    service = AgentService(db)
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return [_version_to_dict(v) for v in agent.versions]


@router.post("/{agent_id}/restore/{version_no}")
async def restore_version(
    agent_id: str,
    version_no: int,
    db: AsyncSession = Depends(get_db),
):
    """Restore working draft configuration to a specific version number."""
    service = AgentService(db)
    agent = await service.restore_agent(agent_id=agent_id, version_no=version_no)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent or target version not found")
    return _agent_to_dict(agent)


@router.post("/{agent_id}/duplicate")
async def duplicate_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Duplicate an existing agent and its draft settings."""
    service = AgentService(db)
    new_agent = await service.duplicate_agent(agent_id=agent_id)
    if not new_agent:
        raise HTTPException(status_code=404, detail="Original agent not found")
    return _agent_to_dict(new_agent)
