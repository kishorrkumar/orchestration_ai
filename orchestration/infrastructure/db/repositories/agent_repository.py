"""
SQLAlchemy 2.0 repository implementing domain AgentRepository protocol.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestration.db import models as db_models
from orchestration.domain.agent import Agent, AgentStatus, AgentVersion
from orchestration.domain.protocols import AgentRepository


class SqlAlchemyAgentRepository(AgentRepository):
    """Concrete repository mapping between domain Agent aggregates and SQLAlchemy models."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _to_domain_agent(rec: db_models.Agent) -> Agent:
        pub_ver: int | None = None
        if rec.published_version_id and "_v" in rec.published_version_id:
            try:
                pub_ver = int(rec.published_version_id.split("_v")[-1])
            except ValueError:
                pub_ver = None

        return Agent(
            id=rec.id,
            name=rec.name,
            voice_id=rec.draft_voice_id,
            greeting=rec.draft_greeting_text,
            agent_speaks_first=(rec.draft_greeting_mode == "agent_first"),
            system_prompt=rec.draft_system_prompt,
            ending=rec.draft_ending_text,
            end_call_timeout_sec=2.0,
            silence_timeout_sec=float(rec.draft_end_silence_sec),
            max_duration_sec=float(rec.draft_max_duration_sec),
            timezone_str=rec.draft_timezone,
            engine=getattr(rec, "draft_engine", "personaplex_s2s") or "personaplex_s2s",
            language=getattr(rec, "draft_language", "en") or "en",
            pipeline_json=getattr(rec, "draft_pipeline_json", "{}") or "{}",
            status=AgentStatus(rec.status),
            current_version=rec.current_version_no,
            published_version=pub_ver,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
        )

    @staticmethod
    def _to_domain_version(rec: db_models.AgentVersion) -> AgentVersion:
        return AgentVersion(
            version_id=rec.id,
            agent_id=rec.agent_id,
            version_number=rec.version_no,
            name=rec.name or "",
            voice_id=rec.voice_id,
            greeting=rec.greeting_text,
            agent_speaks_first=(rec.greeting_mode == "agent_first"),
            system_prompt=rec.system_prompt,
            ending=rec.ending_text,
            end_call_timeout_sec=2.0,
            silence_timeout_sec=float(rec.end_silence_sec),
            max_duration_sec=float(rec.max_duration_sec),
            timezone_str=rec.timezone,
            compiled_prompt=rec.compiled_prompt or "",
            token_count=rec.compiled_token_count,
            engine=getattr(rec, "engine", "personaplex_s2s") or "personaplex_s2s",
            language=getattr(rec, "language", "en") or "en",
            pipeline_json=getattr(rec, "pipeline_json", "{}") or "{}",
            created_at=rec.created_at,
            change_note=rec.change_note,
        )

    async def get_by_id(self, agent_id: str) -> Agent | None:
        stmt = select(db_models.Agent).where(db_models.Agent.id == agent_id)
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()
        return self._to_domain_agent(record) if record else None

    async def list_agents(self, limit: int = 50, offset: int = 0) -> list[Agent]:
        stmt = (
            select(db_models.Agent)
            .order_by(db_models.Agent.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        records = result.scalars().all()
        return [self._to_domain_agent(r) for r in records]

    async def save(self, agent: Agent) -> Agent:
        stmt = select(db_models.Agent).where(db_models.Agent.id == agent.id)
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            record = db_models.Agent(
                id=agent.id,
                workspace_id="wks_default",
                name=agent.name,
                status=agent.status.value,
                current_version_no=agent.current_version,
                draft_voice_id=agent.voice_id,
                draft_greeting_text=agent.greeting,
                draft_greeting_mode="agent_first" if agent.agent_speaks_first else "user_first",
                draft_system_prompt=agent.system_prompt,
                draft_ending_text=agent.ending,
                draft_end_silence_sec=int(agent.silence_timeout_sec),
                draft_max_duration_sec=int(agent.max_duration_sec),
                draft_timezone=agent.timezone_str,
                draft_engine=agent.engine,
                draft_language=agent.language,
                draft_pipeline_json=agent.pipeline_json,
                created_at=agent.created_at,
                updated_at=agent.updated_at,
            )
            self.session.add(record)
        else:
            record.name = agent.name
            record.status = agent.status.value
            record.current_version_no = agent.current_version
            if agent.published_version:
                record.published_version_id = f"{agent.id}_v{agent.published_version}"
            record.draft_voice_id = agent.voice_id
            record.draft_greeting_text = agent.greeting
            record.draft_greeting_mode = "agent_first" if agent.agent_speaks_first else "user_first"
            record.draft_system_prompt = agent.system_prompt
            record.draft_ending_text = agent.ending
            record.draft_end_silence_sec = int(agent.silence_timeout_sec)
            record.draft_max_duration_sec = int(agent.max_duration_sec)
            record.draft_timezone = agent.timezone_str
            record.draft_engine = agent.engine
            record.draft_language = agent.language
            record.draft_pipeline_json = agent.pipeline_json
            record.updated_at = agent.updated_at

        await self.session.commit()
        await self.session.refresh(record)
        return self._to_domain_agent(record)

    async def delete(self, agent_id: str) -> bool:
        stmt = delete(db_models.Agent).where(db_models.Agent.id == agent_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        rowcount = getattr(result, "rowcount", 0)
        return bool(rowcount > 0)

    async def get_version(self, agent_id: str, version_number: int) -> AgentVersion | None:
        stmt = select(db_models.AgentVersion).where(
            db_models.AgentVersion.agent_id == agent_id,
            db_models.AgentVersion.version_no == version_number,
        )
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()
        return self._to_domain_version(record) if record else None

    async def list_versions(self, agent_id: str) -> list[AgentVersion]:
        stmt = (
            select(db_models.AgentVersion)
            .where(db_models.AgentVersion.agent_id == agent_id)
            .order_by(db_models.AgentVersion.version_no.desc())
        )
        result = await self.session.execute(stmt)
        records = result.scalars().all()
        return [self._to_domain_version(r) for r in records]

    async def save_version(self, version: AgentVersion) -> AgentVersion:
        record = db_models.AgentVersion(
            id=version.version_id,
            agent_id=version.agent_id,
            version_no=version.version_number,
            name=version.name,
            voice_id=version.voice_id,
            greeting_text=version.greeting,
            greeting_mode="agent_first" if version.agent_speaks_first else "user_first",
            system_prompt=version.system_prompt,
            ending_text=version.ending,
            end_silence_sec=int(version.silence_timeout_sec),
            max_duration_sec=int(version.max_duration_sec),
            timezone=version.timezone_str,
            engine=version.engine,
            language=version.language,
            pipeline_json=version.pipeline_json,
            compiled_prompt=version.compiled_prompt,
            compiled_token_count=version.token_count,
            change_note=version.change_note or "",
            created_at=version.created_at,
        )
        self.session.add(record)
        await self.session.commit()
        return version
