"""
FastAPI REST controller for Voice Agents and Immutable Versions (/v2/agents).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from orchestration.application.agents.commands import (
    CreateAgentCommand,
    PublishVersionCommand,
    UpdateAgentCommand,
)
from orchestration.application.agents.service import AgentApplicationService
from orchestration.domain.agent import OFFICIAL_PRESETS, Agent
from orchestration.interfaces.http.dependencies import get_agent_service
from orchestration.interfaces.http.schemas.agent_schemas import (
    AgentResponse,
    AgentVersionResponse,
    CreateAgentRequest,
    PublishVersionRequest,
    UpdateAgentRequest,
    VoicePresetResponse,
)

router = APIRouter(prefix="/v2/agents", tags=["Agents"])


def _to_agent_response(a: Agent) -> AgentResponse:
    return AgentResponse(
        id=a.id,
        name=a.name,
        voice_id=a.voice_id,
        greeting=a.greeting,
        agent_speaks_first=a.agent_speaks_first,
        system_prompt=a.system_prompt,
        ending=a.ending,
        end_call_timeout_sec=a.end_call_timeout_sec,
        silence_timeout_sec=a.silence_timeout_sec,
        max_duration_sec=a.max_duration_sec,
        timezone_str=a.timezone_str,
        status=a.status.value,
        current_version=a.current_version,
        published_version=a.published_version,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )


@router.get("/voices", response_model=list[VoicePresetResponse])
async def list_voice_presets() -> list[VoicePresetResponse]:
    """Return all 18 official PersonaPlex voice conditioning presets."""
    return [
        VoicePresetResponse(
            id=p.id,
            name=p.name,
            gender=p.gender,
            speaking_style=p.speaking_style,
            accent=p.accent,
            recommended_for=p.recommended_for,
        )
        for p in OFFICIAL_PRESETS.values()
    ]


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    req: CreateAgentRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Create a new Voice Agent in draft status with Version 1."""
    cmd = CreateAgentCommand(
        name=req.name,
        voice_id=req.voice_id,
        greeting=req.greeting,
        agent_speaks_first=req.agent_speaks_first,
        system_prompt=req.system_prompt,
        ending=req.ending,
        end_call_timeout_sec=req.end_call_timeout_sec,
        silence_timeout_sec=req.silence_timeout_sec,
        max_duration_sec=req.max_duration_sec,
        timezone_str=req.timezone_str,
    )
    agent = await svc.create_agent(cmd)
    return _to_agent_response(agent)


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    limit: int = 50,
    offset: int = 0,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> list[AgentResponse]:
    """List all voice agents ordered by last updated timestamp."""
    agents = await svc.list_agents(limit=limit, offset=offset)
    return [_to_agent_response(a) for a in agents]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Fetch an agent by ID."""
    agent = await svc.get_agent(agent_id)
    return _to_agent_response(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    req: UpdateAgentRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Autosave agent edits, bumping the version and creating an immutable snapshot."""
    cmd = UpdateAgentCommand(
        name=req.name,
        voice_id=req.voice_id,
        greeting=req.greeting,
        agent_speaks_first=req.agent_speaks_first,
        system_prompt=req.system_prompt,
        ending=req.ending,
        end_call_timeout_sec=req.end_call_timeout_sec,
        silence_timeout_sec=req.silence_timeout_sec,
        max_duration_sec=req.max_duration_sec,
        timezone_str=req.timezone_str,
        change_note=req.change_note,
    )
    agent = await svc.update_agent(agent_id, cmd)
    return _to_agent_response(agent)


@router.post("/{agent_id}/publish", response_model=AgentResponse)
async def publish_version(
    agent_id: str,
    req: PublishVersionRequest,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Publish current draft version. Enforces strict <=350 token limit."""
    cmd = PublishVersionCommand(change_note=req.change_note)
    agent = await svc.publish_version(agent_id, cmd)
    return _to_agent_response(agent)


@router.post("/{agent_id}/revert/{version_number}", response_model=AgentResponse)
async def revert_version(
    agent_id: str,
    version_number: int,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> AgentResponse:
    """Restore agent draft configuration from an earlier immutable version."""
    agent = await svc.revert_version(agent_id, version_number)
    return _to_agent_response(agent)


@router.get("/{agent_id}/versions", response_model=list[AgentVersionResponse])
async def list_versions(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> list[AgentVersionResponse]:
    """List all immutable version snapshots of an agent."""
    versions = await svc.list_versions(agent_id)
    return [
        AgentVersionResponse(
            version_id=v.version_id,
            agent_id=v.agent_id,
            version_number=v.version_number,
            name=v.name,
            voice_id=v.voice_id,
            greeting=v.greeting,
            agent_speaks_first=v.agent_speaks_first,
            system_prompt=v.system_prompt,
            ending=v.ending,
            end_call_timeout_sec=v.end_call_timeout_sec,
            silence_timeout_sec=v.silence_timeout_sec,
            max_duration_sec=v.max_duration_sec,
            timezone_str=v.timezone_str,
            compiled_prompt=v.compiled_prompt,
            token_count=v.token_count,
            created_at=v.created_at,
            change_note=v.change_note,
        )
        for v in versions
    ]


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    svc: AgentApplicationService = Depends(get_agent_service),
) -> None:
    """Delete an agent and all associated versions."""
    await svc.delete_agent(agent_id)
