"""
Database Seeder for Lean S2S Voice Agent Platform.
Enforces the SINGLE-AGENT invariant loaded from agent.yaml.
Removes legacy multi-agent starter records.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..settings import app_settings
from .models import Agent, AgentVersion
from .service import AgentService

logger = logging.getLogger("orchestration.db.seed")


LEGACY_STARTER_NAMES = {
    "Friendly Caller (Blank Template)",
    "Clinic Appointment Assistant",
    "Support Agent (Alex)",
    "Indian Tech Support (Arjun)",
    "Wise Teacher (Dr. Elena)",
    "Marcus (Sales Specialist)",
    "Sam (Casual Friend)",
}


async def seed_database(db: AsyncSession) -> None:
    """Seeds default workspace and synchronizes the single agent defined in agent.yaml."""
    service = AgentService(db)
    workspace = await service.get_or_create_default_workspace()

    agent_cfg = app_settings.agent
    pipeline_data = {
        "audio_temperature": agent_cfg.audio_temperature,
        "text_temperature": agent_cfg.text_temperature,
        "audio_topk": agent_cfg.audio_topk,
        "text_topk": agent_cfg.text_topk,
        "seed": agent_cfg.seed,
        **agent_cfg.variables,
    }

    # Fetch existing agents in workspace
    stmt = select(Agent).where(Agent.workspace_id == workspace.id)
    res = await db.execute(stmt)
    existing_agents = list(res.scalars().all())

    # Prune legacy starter agents to enforce ONE default agent
    for ag in existing_agents:
        if ag.name in LEGACY_STARTER_NAMES:
            logger.info("Pruning legacy multi-agent starter record: %s (%s)", ag.name, ag.id)
            await db.execute(delete(AgentVersion).where(AgentVersion.agent_id == ag.id))
            await db.execute(delete(Agent).where(Agent.id == ag.id))

    # Find or create the primary single agent defined in agent.yaml
    stmt = select(Agent).where(Agent.workspace_id == workspace.id, Agent.name == agent_cfg.name)
    res = await db.execute(stmt)
    target_agent = res.scalar_one_or_none()

    if target_agent is None:
        target_agent = await service.create_agent(
            name=agent_cfg.name,
            voice_id=agent_cfg.voice_prompt,
            greeting_text=agent_cfg.greeting_text,
            greeting_mode=agent_cfg.greeting_mode,
            system_prompt=agent_cfg.system_prompt,
            ending_text=agent_cfg.ending_text,
            end_silence_sec=agent_cfg.end_silence_sec,
            max_duration_sec=agent_cfg.max_duration_sec,
            timezone=agent_cfg.timezone,
            pipeline_json=json.dumps(pipeline_data),
            auto_publish=True,
            change_note="Initial single agent from agent.yaml",
        )
        logger.info("Created single voice agent from agent.yaml: %s (%s)", target_agent.name, target_agent.id)
    else:
        # Update existing agent to match agent.yaml
        target_agent.name = agent_cfg.name
        target_agent.draft_voice_id = agent_cfg.voice_prompt
        target_agent.draft_greeting_text = agent_cfg.greeting_text
        target_agent.draft_greeting_mode = agent_cfg.greeting_mode
        target_agent.draft_system_prompt = agent_cfg.system_prompt
        target_agent.draft_ending_text = agent_cfg.ending_text
        target_agent.draft_end_silence_sec = agent_cfg.end_silence_sec
        target_agent.draft_max_duration_sec = agent_cfg.max_duration_sec
        target_agent.draft_timezone = agent_cfg.timezone
        target_agent.draft_pipeline_json = json.dumps(pipeline_data)
        await service.publish_agent(
            agent_id=target_agent.id,
            change_note="Synchronized with agent.yaml",
        )
        logger.info("Synchronized active voice agent with agent.yaml: %s (%s)", target_agent.name, target_agent.id)

    await db.commit()
    logger.info("Single-agent database synchronization complete.")


async def main() -> None:
    from .session import async_session_factory, init_db
    await init_db()
    async with async_session_factory() as db:
        await seed_database(db)
    print("Database seeding completed successfully.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
