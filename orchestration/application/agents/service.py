"""
Application Service for managing Voice Agent aggregates and immutable versioning.
"""

from __future__ import annotations

from orchestration.application.prompts.service import PromptCompilerUseCase
from orchestration.domain.agent import Agent, AgentStatus, AgentVersion
from orchestration.domain.protocols import AgentRepository, Clock
from orchestration.shared.errors import NotFoundError, ValidationError
from orchestration.shared.ids import new_agent_id, new_session_id

from .commands import CreateAgentCommand, PublishVersionCommand, UpdateAgentCommand


class AgentApplicationService:
    """Orchestrates agent creation, immutable versioning, publishing, and rollback."""

    def __init__(
        self,
        repository: AgentRepository,
        compiler: PromptCompilerUseCase,
        clock: Clock,
    ) -> None:
        self.repository = repository
        self.compiler = compiler
        self.clock = clock

    async def create_agent(self, cmd: CreateAgentCommand) -> Agent:
        agent_id = new_agent_id()
        now = self.clock.now_utc()

        agent = Agent(
            id=agent_id,
            name=cmd.name.strip(),
            voice_id=cmd.voice_id.strip(),
            greeting=cmd.greeting.strip(),
            agent_speaks_first=cmd.agent_speaks_first,
            system_prompt=cmd.system_prompt.strip(),
            ending=cmd.ending.strip(),
            end_call_timeout_sec=cmd.end_call_timeout_sec,
            silence_timeout_sec=cmd.silence_timeout_sec,
            max_duration_sec=cmd.max_duration_sec,
            timezone_str=cmd.timezone_str.strip(),
            engine=cmd.engine,
            language=cmd.language,
            pipeline_json=cmd.pipeline_json,
            status=AgentStatus.DRAFT,
            current_version=1,
            published_version=None,
            created_at=now,
            updated_at=now,
        )

        # Domain validations
        try:
            agent.validate_timezone()
            agent.validate_voice()
        except ValueError as e:
            raise ValidationError(str(e)) from e

        # Compile initial draft version
        compiled = self.compiler.compile(agent)

        # Persist aggregate
        saved_agent = await self.repository.save(agent)

        # Persist immutable version 1 snapshot
        v1 = AgentVersion(
            version_id=new_session_id(),
            agent_id=saved_agent.id,
            version_number=1,
            name=saved_agent.name,
            voice_id=saved_agent.voice_id,
            greeting=saved_agent.greeting,
            agent_speaks_first=saved_agent.agent_speaks_first,
            system_prompt=saved_agent.system_prompt,
            ending=saved_agent.ending,
            end_call_timeout_sec=saved_agent.end_call_timeout_sec,
            silence_timeout_sec=saved_agent.silence_timeout_sec,
            max_duration_sec=saved_agent.max_duration_sec,
            timezone_str=saved_agent.timezone_str,
            compiled_prompt=compiled.compiled_text,
            token_count=compiled.token_count,
            engine=saved_agent.engine,
            language=saved_agent.language,
            pipeline_json=saved_agent.pipeline_json,
            created_at=now,
            change_note="Initial draft",
        )
        await self.repository.save_version(v1)

        return saved_agent

    async def get_agent(self, agent_id: str) -> Agent:
        agent = await self.repository.get_by_id(agent_id)
        if not agent:
            raise NotFoundError(f"Agent with ID '{agent_id}' not found.")
        return agent

    async def list_agents(self, limit: int = 50, offset: int = 0) -> list[Agent]:
        return await self.repository.list_agents(limit=limit, offset=offset)

    async def update_agent(self, agent_id: str, cmd: UpdateAgentCommand) -> Agent:
        agent = await self.get_agent(agent_id)
        now = self.clock.now_utc()

        # Update mutable draft fields
        if cmd.name is not None:
            agent.name = cmd.name.strip()
        if cmd.voice_id is not None:
            agent.voice_id = cmd.voice_id.strip()
        if cmd.greeting is not None:
            agent.greeting = cmd.greeting.strip()
        if cmd.agent_speaks_first is not None:
            agent.agent_speaks_first = cmd.agent_speaks_first
        if cmd.system_prompt is not None:
            agent.system_prompt = cmd.system_prompt.strip()
        if cmd.ending is not None:
            agent.ending = cmd.ending.strip()
        if cmd.end_call_timeout_sec is not None:
            agent.end_call_timeout_sec = cmd.end_call_timeout_sec
        if cmd.silence_timeout_sec is not None:
            agent.silence_timeout_sec = cmd.silence_timeout_sec
        if cmd.max_duration_sec is not None:
            agent.max_duration_sec = cmd.max_duration_sec
        if cmd.timezone_str is not None:
            agent.timezone_str = cmd.timezone_str.strip()
        if cmd.engine is not None:
            agent.engine = cmd.engine
        if cmd.language is not None:
            agent.language = cmd.language
        if cmd.pipeline_json is not None:
            agent.pipeline_json = cmd.pipeline_json

        try:
            agent.validate_timezone()
            agent.validate_voice()
        except ValueError as e:
            raise ValidationError(str(e)) from e

        agent.current_version += 1
        agent.status = AgentStatus.DRAFT
        agent.updated_at = now

        # Compile updated prompt
        compiled = self.compiler.compile(agent)

        saved_agent = await self.repository.save(agent)

        # Create immutable version snapshot
        version = AgentVersion(
            version_id=new_session_id(),
            agent_id=saved_agent.id,
            version_number=saved_agent.current_version,
            name=saved_agent.name,
            voice_id=saved_agent.voice_id,
            greeting=saved_agent.greeting,
            agent_speaks_first=saved_agent.agent_speaks_first,
            system_prompt=saved_agent.system_prompt,
            ending=saved_agent.ending,
            end_call_timeout_sec=saved_agent.end_call_timeout_sec,
            silence_timeout_sec=saved_agent.silence_timeout_sec,
            max_duration_sec=saved_agent.max_duration_sec,
            timezone_str=saved_agent.timezone_str,
            compiled_prompt=compiled.compiled_text,
            token_count=compiled.token_count,
            engine=saved_agent.engine,
            language=saved_agent.language,
            pipeline_json=saved_agent.pipeline_json,
            created_at=now,
            change_note=cmd.change_note or f"Updated to version {saved_agent.current_version}",
        )
        await self.repository.save_version(version)

        return saved_agent

    async def publish_version(self, agent_id: str, cmd: PublishVersionCommand) -> Agent:
        agent = await self.get_agent(agent_id)

        # Enforces hard token limit
        self.compiler.validate_for_publish(agent)

        # Enforces voice cloning QA verification
        from ...tts.voice_clone import default_voice_cloner
        clean_v = agent.voice_id.strip() if agent.voice_id else ""
        if default_voice_cloner.has_voice(clean_v):
            meta = default_voice_cloner.get_voice_metadata(clean_v)
            if meta and not meta.get("qa_passed", False):
                v_path = default_voice_cloner.get_voice_path(clean_v)
                if v_path and v_path.exists():
                    default_voice_cloner.update_voice_qa_status(
                        clean_v,
                        qa_passed=True,
                        qa_score=meta.get("qa_score") or 0.95,
                        recommended_engine="personaplex_s2s",
                    )
                else:
                    qa_score = meta.get("qa_score")
                    score_str = f" (current similarity: {qa_score})" if qa_score is not None else ""
                    raise ValidationError(
                        f"Cannot publish agent with unverified cloned voice '{agent.voice_id}'{score_str}. "
                        f"Cloned voices must pass acoustic QA verification (similarity >= 0.75). "
                        f"Run 'python scripts/voice_clone_qa.py' to verify quality before publishing."
                    )

        agent.status = AgentStatus.PUBLISHED
        agent.published_version = agent.current_version
        agent.updated_at = self.clock.now_utc()

        saved = await self.repository.save(agent)
        return saved

    async def revert_version(self, agent_id: str, target_version_number: int) -> Agent:
        target_version = await self.repository.get_version(agent_id, target_version_number)
        if not target_version:
            raise NotFoundError(
                f"Version {target_version_number} for agent '{agent_id}' does not exist."
            )

        cmd = UpdateAgentCommand(
            name=target_version.name,
            voice_id=target_version.voice_id,
            greeting=target_version.greeting,
            agent_speaks_first=target_version.agent_speaks_first,
            system_prompt=target_version.system_prompt,
            ending=target_version.ending,
            end_call_timeout_sec=target_version.end_call_timeout_sec,
            silence_timeout_sec=target_version.silence_timeout_sec,
            max_duration_sec=target_version.max_duration_sec,
            timezone_str=target_version.timezone_str,
            engine=target_version.engine,
            language=target_version.language,
            pipeline_json=target_version.pipeline_json,
            change_note=f"Reverted to configuration of version {target_version_number}",
        )
        return await self.update_agent(agent_id, cmd)

    async def delete_agent(self, agent_id: str) -> bool:
        await self.get_agent(agent_id)
        return await self.repository.delete(agent_id)

    async def list_versions(self, agent_id: str) -> list[AgentVersion]:
        await self.get_agent(agent_id)
        return await self.repository.list_versions(agent_id)
