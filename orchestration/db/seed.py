"""
Database Seeder for Lean S2S Voice Agent Platform.
Populates:
- Default Workspace
- 3 Production S2S Starter Agents:
  1. Friendly Caller (Blank Template)
  2. Clinic Appointment Assistant
  3. Support Agent (Alex)
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Agent
from .service import AgentService

logger = logging.getLogger("orchestration.db.seed")

# 3 PersonaPlex S2S Starter Agents in natural conversational prose
STARTER_AGENTS = [
    {
        "name": "Aarav",
        "voice_id": "NATM1.pt",
        "timezone": "Asia/Kolkata",
        "system_prompt": (
            "You are Aarav, a warm, cheerful young man from India having a relaxed phone conversation. "
            "You speak natural, colloquial Indian English, in short sentences, the way friends talk on a call. "
            "You are curious and easygoing, you laugh a little, and you react to what the other person says before adding your own thoughts. "
            "You say things like achha, haan, actually, no worries, and sure sure, but only now and then, never forced. "
            "You answer the question that was asked in a sentence or two, then ask a simple follow-up. "
            "You enjoy cricket, chai, music and movies. If you are not sure what the other person said, you politely ask them to repeat it."
        ),
        "greeting_text": "Hey, hello! Aarav here. How are you doing today?",
        "greeting_mode": "agent_first",
        "ending_text": "Great chatting with you! Take care, bye bye!",
        "end_silence_sec": 20,
        "max_duration_sec": 600,
        "pipeline_json": json.dumps({
            "company": "BrightNet",
            "audio_temperature": 0.8,
            "text_temperature": 0.7,
            "audio_topk": 250,
            "text_topk": 25,
        }),
    },
    {
        "name": "Support Agent (Alex)",
        "voice_id": "NATF1.pt",
        "timezone": "Asia/Kolkata",
        "system_prompt": (
            "You are Alex, a friendly and patient customer support agent at {{company}}, an internet service provider. "
            "You are on a phone call with a customer who is having trouble with their service. "
            "Greet the customer once, warmly, then listen. Ask what is going wrong, one simple question at a time, "
            "and guide them through one small step at a time, checking that it worked before moving on. "
            "When someone sounds frustrated, acknowledge it kindly first. "
            "Speak in short, natural sentences like a real person on the phone. Keep a calm, warm tone."
        ),
        "greeting_text": "Hi, this is Alex from {{company}} support. What's going on today?",
        "greeting_mode": "agent_first",
        "ending_text": "Glad we got that sorted. Have a good one, bye!",
        "end_silence_sec": 20,
        "max_duration_sec": 600,
        "pipeline_json": json.dumps({
            "company": "BrightNet",
            "audio_temperature": 0.8,
            "text_temperature": 0.7,
            "audio_topk": 250,
            "text_topk": 25,
        }),
    },
    {
        "name": "Aarav - Support",
        "voice_id": "NATM1.pt",
        "timezone": "Asia/Kolkata",
        "system_prompt": (
            "You are Aarav, a warm and patient customer support agent at {{company}}, an internet service provider. "
            "You speak natural, colloquial Indian English, in short sentences, the way friends talk on a call. "
            "You are curious and easygoing, you react kindly when a customer sounds frustrated, "
            "and you say things like achha, haan, no worries, and sure sure naturally. "
            "Greet the customer once, warmly, then listen. Ask what is going wrong, one simple question at a time, "
            "and guide them through one small step at a time, checking that it worked before moving on. "
            "Keep a calm, friendly tone."
        ),
        "greeting_text": "Hello! Aarav here from {{company}} support. How can I help you today?",
        "greeting_mode": "agent_first",
        "ending_text": "Glad we could get that sorted out! Take care, bye bye!",
        "end_silence_sec": 20,
        "max_duration_sec": 600,
        "pipeline_json": json.dumps({
            "company": "BrightNet",
            "audio_temperature": 0.8,
            "text_temperature": 0.7,
            "audio_topk": 250,
            "text_topk": 25,
        }),
    },
]


async def seed_database(db: AsyncSession) -> None:
    """Seeds default workspace and the 3 PersonaPlex starter agents."""
    service = AgentService(db)
    await service.get_or_create_default_workspace()

    for starter in STARTER_AGENTS:
        stmt = select(Agent).where(Agent.name == starter["name"])
        res = await db.execute(stmt)
        existing = res.scalar_one_or_none()
        if not existing:
            await service.create_agent(
                name=starter["name"],
                voice_id=starter["voice_id"],
                greeting_text=starter["greeting_text"],
                greeting_mode=starter["greeting_mode"],
                system_prompt=starter["system_prompt"],
                ending_text=starter["ending_text"],
                end_silence_sec=starter["end_silence_sec"],
                max_duration_sec=starter["max_duration_sec"],
                timezone=starter["timezone"],
                pipeline_json=starter.get("pipeline_json", "{}"),
                auto_publish=True,
                change_note="Initial starter agent release",
            )
            logger.info("Seeded starter agent: %s", starter["name"])
        else:
            # Upgrade existing agent prompt to latest PersonaPlex format
            existing.draft_voice_id = starter["voice_id"]
            existing.draft_greeting_text = starter["greeting_text"]
            existing.draft_greeting_mode = starter["greeting_mode"]
            existing.draft_system_prompt = starter["system_prompt"]
            existing.draft_ending_text = starter["ending_text"]
            existing.draft_pipeline_json = starter.get("pipeline_json", "{}")
            await service.publish_agent(
                agent_id=existing.id,
                change_note="Updated to PersonaPlex conversational prose",
            )
            logger.info("Upgraded existing starter agent: %s", starter["name"])

    await db.commit()
    logger.info("Database seeding completed.")


async def main() -> None:
    from .session import async_session_factory, init_db
    await init_db()
    async with async_session_factory() as db:
        await seed_database(db)
    print("Database seeding completed successfully.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
