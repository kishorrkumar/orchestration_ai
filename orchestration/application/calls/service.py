"""
Application Service for managing real-time speech call sessions and turn logging.
"""

from __future__ import annotations

from orchestration.domain.protocols import AgentRepository, CallSessionRepository, Clock
from orchestration.domain.session import CallSession, CallStatus, CallTurn, EndReason, TurnSpeaker
from orchestration.shared.errors import NotFoundError
from orchestration.shared.ids import new_session_id, new_turn_id


class CallApplicationService:
    """Manages the persistence lifecycle and turn logging of voice calls."""

    def __init__(
        self,
        session_repo: CallSessionRepository,
        agent_repo: AgentRepository,
        clock: Clock,
    ) -> None:
        self.session_repo = session_repo
        self.agent_repo = agent_repo
        self.clock = clock

    async def start_session(
        self,
        agent_id: str,
        client_type: str = "web",
    ) -> CallSession:
        agent = await self.agent_repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError(f"Agent with ID '{agent_id}' does not exist.")

        session_id = new_session_id()
        now = self.clock.now_utc()

        session = CallSession(
            id=session_id,
            agent_id=agent.id,
            agent_version=agent.current_version,
            status=CallStatus.CONNECTED,
            client_type=client_type,
            duration_sec=0.0,
            end_reason=None,
            total_turns=0,
            created_at=now,
            ended_at=None,
        )

        return await self.session_repo.save_session(session)

    async def record_turn(
        self,
        session_id: str,
        speaker: TurnSpeaker,
        text: str,
        started_at_sec: float,
        ended_at_sec: float,
        latency_ms: float | None = None,
    ) -> CallTurn:
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise NotFoundError(f"Session with ID '{session_id}' not found.")

        session.total_turns += 1
        turn = CallTurn(
            id=new_turn_id(),
            session_id=session_id,
            turn_index=session.total_turns,
            speaker=speaker,
            text=text.strip(),
            started_at_sec=started_at_sec,
            ended_at_sec=ended_at_sec,
            latency_ms=latency_ms,
            created_at=self.clock.now_utc(),
        )

        await self.session_repo.save_session(session)
        return await self.session_repo.save_turn(turn)

    async def end_session(
        self,
        session_id: str,
        end_reason: EndReason,
        duration_sec: float,
    ) -> CallSession:
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise NotFoundError(f"Session with ID '{session_id}' not found.")

        session.status = CallStatus.COMPLETED if end_reason != EndReason.ERROR else CallStatus.FAILED
        session.end_reason = end_reason
        session.duration_sec = duration_sec
        session.ended_at = self.clock.now_utc()

        return await self.session_repo.save_session(session)

    async def get_session(self, session_id: str) -> CallSession:
        session = await self.session_repo.get_by_id(session_id)
        if not session:
            raise NotFoundError(f"Session with ID '{session_id}' not found.")
        return session

    async def list_sessions(
        self,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CallSession]:
        return await self.session_repo.list_sessions(agent_id=agent_id, limit=limit, offset=offset)

    async def list_turns(self, session_id: str) -> list[CallTurn]:
        await self.get_session(session_id)
        return await self.session_repo.list_turns(session_id)
