"""
Agent and Immutable Versioning Service for Lean S2S Voice Agent Platform.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..prompts.compiler import compile_prompt
from .models import Agent, AgentVersion, Workspace, generate_prefixed_id


class AgentService:
    """Service managing lean S2S agent lifecycle, drafting, and version snapshots."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_or_create_default_workspace(self) -> Workspace:
        """Returns the default workspace or creates one if none exists."""
        stmt = select(Workspace).limit(1)
        res = await self.db.execute(stmt)
        wks = res.scalar_one_or_none()
        if not wks:
            wks = Workspace(id="wks_default", name="Default Workspace", slug="default")
            self.db.add(wks)
            await self.db.flush()
        return wks

    async def create_agent(
        self,
        name: str,
        id: Optional[str] = None,
        voice_id: str = "NATF0.pt",
        greeting_text: str = "",
        greeting_mode: str = "agent_first",
        system_prompt: str = "",
        ending_text: str = "",
        end_silence_sec: int = 20,
        max_duration_sec: int = 600,
        timezone: str = "Asia/Kolkata",
        auto_publish: bool = True,
        change_note: str = "Initial release",
    ) -> Agent:
        """Creates a new agent in draft state and optionally creates initial version 1."""
        wks = await self.get_or_create_default_workspace()

        agent_id = id or generate_prefixed_id("agt")
        if id:
            existing = await self.get_agent(id)
            if existing:
                existing.name = name
                existing.draft_voice_id = voice_id
                existing.draft_greeting_text = greeting_text
                existing.draft_greeting_mode = greeting_mode
                existing.draft_system_prompt = system_prompt
                existing.draft_ending_text = ending_text
                existing.draft_end_silence_sec = end_silence_sec
                existing.draft_max_duration_sec = max_duration_sec
                existing.draft_timezone = timezone
                if auto_publish:
                    await self.publish_agent(existing.id, change_note=change_note)
                else:
                    existing.status = "draft"
                    await self.db.flush()
                await self.db.refresh(existing)
                return existing
            else:
                # Ensure no orphaned versions exist from prior runs
                await self.db.execute(delete(AgentVersion).where(AgentVersion.agent_id == id))
                await self.db.flush()

        agent = Agent(
            id=agent_id,
            workspace_id=wks.id,
            name=name,
            status="draft",
            current_version_no=1,
            draft_voice_id=voice_id,
            draft_greeting_text=greeting_text,
            draft_greeting_mode=greeting_mode,
            draft_system_prompt=system_prompt,
            draft_ending_text=ending_text,
            draft_end_silence_sec=end_silence_sec,
            draft_max_duration_sec=max_duration_sec,
            draft_timezone=timezone,
        )
        self.db.add(agent)
        await self.db.flush()

        if auto_publish:
            compiled = compile_prompt(
                system_prompt=system_prompt,
                greeting_text=greeting_text,
                greeting_mode=greeting_mode,
                ending_text=ending_text,
                agent_name=name,
                timezone=timezone,
            )
            v1 = AgentVersion(
                id=generate_prefixed_id("ver"),
                agent_id=agent.id,
                version_no=1,
                voice_id=voice_id,
                greeting_text=greeting_text,
                greeting_mode=greeting_mode,
                system_prompt=system_prompt,
                ending_text=ending_text,
                end_silence_sec=end_silence_sec,
                max_duration_sec=max_duration_sec,
                timezone=timezone,
                compiled_token_count=compiled.token_count,
                change_note=change_note,
            )
            self.db.add(v1)
            await self.db.flush()

            agent.published_version_id = v1.id
            agent.status = "published"
            await self.db.flush()

        await self.db.refresh(agent)
        return agent

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Fetches an agent with versions loaded."""
        stmt = (
            select(Agent)
            .options(selectinload(Agent.versions))
            .where(Agent.id == agent_id)
        )
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def list_agents(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Agent], int]:
        """Lists agents with optional filters and total count."""
        stmt = select(Agent).options(selectinload(Agent.versions))
        count_stmt = select(func.count(Agent.id))

        if status:
            stmt = stmt.where(Agent.status == status)
            count_stmt = count_stmt.where(Agent.status == status)
        if search:
            search_pat = f"%{search}%"
            stmt = stmt.where(Agent.name.ilike(search_pat))
            count_stmt = count_stmt.where(Agent.name.ilike(search_pat))

        total_res = await self.db.execute(count_stmt)
        total = total_res.scalar_one()

        stmt = stmt.order_by(desc(Agent.updated_at)).limit(limit).offset(offset)
        res = await self.db.execute(stmt)
        agents = list(res.scalars().all())
        return agents, total

    async def update_agent_draft(
        self,
        agent_id: str,
        name: Optional[str] = None,
        voice_id: Optional[str] = None,
        greeting_text: Optional[str] = None,
        greeting_mode: Optional[str] = None,
        system_prompt: Optional[str] = None,
        ending_text: Optional[str] = None,
        end_silence_sec: Optional[int] = None,
        max_duration_sec: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> Optional[Agent]:
        """Updates draft fields in place."""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None

        if name is not None:
            agent.name = name
        if voice_id is not None:
            agent.draft_voice_id = voice_id
        if greeting_text is not None:
            agent.draft_greeting_text = greeting_text
        if greeting_mode is not None:
            agent.draft_greeting_mode = greeting_mode
        if system_prompt is not None:
            agent.draft_system_prompt = system_prompt
        if ending_text is not None:
            agent.draft_ending_text = ending_text
        if end_silence_sec is not None:
            agent.draft_end_silence_sec = end_silence_sec
        if max_duration_sec is not None:
            agent.draft_max_duration_sec = max_duration_sec
        if timezone is not None:
            agent.draft_timezone = timezone

        # Modifying draft marks agent as draft if previously published
        if agent.status == "published":
            agent.status = "draft"

        await self.db.flush()
        await self.db.refresh(agent)
        return agent

    async def publish_agent(
        self,
        agent_id: str,
        change_note: str = "",
    ) -> Optional[AgentVersion]:
        """
        Creates a new immutable AgentVersion from the current draft state,
        validates SentencePiece token limits, bumps version_no, and sets published_version_id.
        """
        agent = await self.get_agent(agent_id)
        if not agent:
            return None

        compiled = compile_prompt(
            system_prompt=agent.draft_system_prompt,
            greeting_text=agent.draft_greeting_text,
            greeting_mode=agent.draft_greeting_mode,
            ending_text=agent.draft_ending_text,
            agent_name=agent.name,
            timezone=agent.draft_timezone,
        )

        if not compiled.can_publish:
            raise ValueError(
                f"Cannot publish agent: prompt exceeds hard limit of 350 tokens (current: {compiled.token_count})."
            )

        new_version_no = agent.current_version_no + 1
        new_version = AgentVersion(
            id=generate_prefixed_id("ver"),
            agent_id=agent.id,
            version_no=new_version_no,
            voice_id=agent.draft_voice_id,
            greeting_text=agent.draft_greeting_text,
            greeting_mode=agent.draft_greeting_mode,
            system_prompt=agent.draft_system_prompt,
            ending_text=agent.draft_ending_text,
            end_silence_sec=agent.draft_end_silence_sec,
            max_duration_sec=agent.draft_max_duration_sec,
            timezone=agent.draft_timezone,
            compiled_token_count=compiled.token_count,
            change_note=change_note or f"Release v{new_version_no}",
        )
        self.db.add(new_version)
        await self.db.flush()

        agent.current_version_no = new_version_no
        agent.published_version_id = new_version.id
        agent.status = "published"

        await self.db.flush()
        await self.db.refresh(agent)
        return new_version

    async def get_version(self, agent_id: str, version_no: int) -> Optional[AgentVersion]:
        """Fetches a specific immutable version of an agent."""
        stmt = select(AgentVersion).where(
            AgentVersion.agent_id == agent_id,
            AgentVersion.version_no == version_no,
        )
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def restore_agent(self, agent_id: str, version_no: int) -> Optional[Agent]:
        """Restores the working draft to the exact configuration of version_no."""
        agent = await self.get_agent(agent_id)
        if not agent:
            return None

        ver = await self.get_version(agent_id, version_no)
        if not ver:
            return None

        agent.draft_voice_id = ver.voice_id
        agent.draft_greeting_text = ver.greeting_text
        agent.draft_greeting_mode = ver.greeting_mode
        agent.draft_system_prompt = ver.system_prompt
        agent.draft_ending_text = ver.ending_text
        agent.draft_end_silence_sec = ver.end_silence_sec
        agent.draft_max_duration_sec = ver.max_duration_sec
        agent.draft_timezone = ver.timezone
        agent.published_version_id = ver.id
        agent.status = "published"

        await self.db.flush()
        await self.db.refresh(agent)
        return agent

    async def duplicate_agent(self, agent_id: str, new_name: Optional[str] = None) -> Optional[Agent]:
        """Duplicates an existing agent and its draft state into a new agent."""
        orig = await self.get_agent(agent_id)
        if not orig:
            return None

        copy_name = new_name or f"{orig.name} (Copy)"
        return await self.create_agent(
            name=copy_name,
            voice_id=orig.draft_voice_id,
            greeting_text=orig.draft_greeting_text,
            greeting_mode=orig.draft_greeting_mode,
            system_prompt=orig.draft_system_prompt,
            ending_text=orig.draft_ending_text,
            end_silence_sec=orig.draft_end_silence_sec,
            max_duration_sec=orig.draft_max_duration_sec,
            timezone=orig.draft_timezone,
            auto_publish=True,
            change_note=f"Duplicated from {orig.name}",
        )

    async def delete_agent(self, agent_id: str) -> bool:
        """Deletes an agent and its versions."""
        from .models import CallSession, CallTurn
        session_subq = select(CallSession.id).where(CallSession.agent_id == agent_id)
        await self.db.execute(delete(CallTurn).where(CallTurn.session_id.in_(session_subq)))
        await self.db.execute(delete(CallSession).where(CallSession.agent_id == agent_id))
        await self.db.execute(delete(AgentVersion).where(AgentVersion.agent_id == agent_id))
        stmt = delete(Agent).where(Agent.id == agent_id)
        res = await self.db.execute(stmt)
        await self.db.flush()
        return (res.rowcount or 0) > 0


class CallSessionService:
    """Service managing call sessions and transcript turn records."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_session(
        self,
        agent_id: str,
        agent_version_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> CallSession:
        from .models import CallSession
        sid = session_id or generate_prefixed_id("ses")
        call = CallSession(
            id=sid,
            agent_id=agent_id,
            agent_version_id=agent_version_id,
            duration_sec=0.0,
            end_reason="",
            ttfa_ms=0.0,
            handshake_ms=0.0,
        )
        self.db.add(call)
        await self.db.flush()
        return call

    async def add_turn(
        self,
        session_id: str,
        idx: int,
        role: str,
        text: str,
        started_ms: float = 0.0,
    ) -> CallTurn:
        from .models import CallTurn
        turn = CallTurn(
            id=generate_prefixed_id("trn"),
            session_id=session_id,
            idx=idx,
            role=role,
            text=text,
            started_ms=started_ms,
        )
        self.db.add(turn)
        await self.db.flush()
        return turn

    async def end_session(
        self,
        session_id: str,
        end_reason: str = "user_hangup",
        duration_sec: float = 0.0,
        ttfa_ms: float = 0.0,
        handshake_ms: float = 0.0,
    ) -> Optional[CallSession]:
        import datetime
        from .models import CallSession
        stmt = select(CallSession).where(CallSession.id == session_id)
        res = await self.db.execute(stmt)
        session = res.scalar_one_or_none()
        if not session:
            return None

        session.ended_at = datetime.datetime.now(datetime.timezone.utc)
        session.end_reason = end_reason
        session.duration_sec = duration_sec
        if ttfa_ms > 0:
            session.ttfa_ms = ttfa_ms
        if handshake_ms > 0:
            session.handshake_ms = handshake_ms

        await self.db.flush()
        return session

    async def get_session(self, session_id: str) -> Optional[CallSession]:
        from .models import CallSession
        stmt = (
            select(CallSession)
            .options(selectinload(CallSession.turns))
            .where(CallSession.id == session_id)
        )
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

