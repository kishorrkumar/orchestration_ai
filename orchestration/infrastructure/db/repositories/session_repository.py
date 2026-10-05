"""
SQLAlchemy 2.0 repository implementing domain CallSessionRepository protocol.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orchestration.db import models as db_models
from orchestration.domain.protocols import CallSessionRepository
from orchestration.domain.session import CallSession, CallStatus, CallTurn, EndReason, TurnSpeaker


class SqlAlchemySessionRepository(CallSessionRepository):
    """Concrete repository mapping between domain CallSession/CallTurn and SQLAlchemy models."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _to_domain_session(rec: db_models.CallSession) -> CallSession:
        return CallSession(
            id=rec.id,
            agent_id=rec.agent_id,
            agent_version=rec.agent_version_number or 1,
            status=CallStatus(rec.status),
            client_type=rec.client_type,
            duration_sec=rec.duration_sec,
            end_reason=EndReason(rec.end_reason) if rec.end_reason else None,
            total_turns=rec.turn_count,
            created_at=rec.created_at,
            ended_at=rec.ended_at,
        )

    @staticmethod
    def _to_domain_turn(rec: db_models.CallTurn) -> CallTurn:
        return CallTurn(
            id=rec.id,
            session_id=rec.session_id,
            turn_index=rec.turn_index,
            speaker=TurnSpeaker(rec.speaker),
            text=rec.transcript or "",
            started_at_sec=rec.started_at_sec or 0.0,
            ended_at_sec=rec.ended_at_sec or 0.0,
            latency_ms=rec.latency_ms,
            created_at=rec.created_at,
        )

    async def get_by_id(self, session_id: str) -> CallSession | None:
        stmt = select(db_models.CallSession).where(db_models.CallSession.id == session_id)
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()
        return self._to_domain_session(record) if record else None

    async def list_sessions(
        self,
        agent_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CallSession]:
        stmt = select(db_models.CallSession).order_by(db_models.CallSession.created_at.desc()).limit(limit).offset(offset)
        if agent_id:
            stmt = stmt.where(db_models.CallSession.agent_id == agent_id)
        result = await self.session.execute(stmt)
        records = result.scalars().all()
        return [self._to_domain_session(r) for r in records]

    async def save_session(self, session: CallSession) -> CallSession:
        stmt = select(db_models.CallSession).where(db_models.CallSession.id == session.id)
        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            record = db_models.CallSession(
                id=session.id,
                workspace_id="wks_default",
                agent_id=session.agent_id,
                agent_version_number=session.agent_version,
                client_type=session.client_type,
                status=session.status.value,
                duration_sec=session.duration_sec,
                end_reason=session.end_reason.value if session.end_reason else None,
                turn_count=session.total_turns,
                created_at=session.created_at,
                ended_at=session.ended_at,
            )
            self.session.add(record)
        else:
            record.status = session.status.value
            record.duration_sec = session.duration_sec
            record.end_reason = session.end_reason.value if session.end_reason else None
            record.turn_count = session.total_turns
            record.ended_at = session.ended_at

        await self.session.commit()
        await self.session.refresh(record)
        return self._to_domain_session(record)

    async def save_turn(self, turn: CallTurn) -> CallTurn:
        record = db_models.CallTurn(
            id=turn.id,
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            speaker=turn.speaker.value,
            transcript=turn.text,
            started_at_sec=turn.started_at_sec,
            ended_at_sec=turn.ended_at_sec,
            latency_ms=turn.latency_ms,
            created_at=turn.created_at,
        )
        self.session.add(record)
        await self.session.commit()
        return turn

    async def list_turns(self, session_id: str) -> list[CallTurn]:
        stmt = (
            select(db_models.CallTurn)
            .where(db_models.CallTurn.session_id == session_id)
            .order_by(db_models.CallTurn.turn_index.asc())
        )
        result = await self.session.execute(stmt)
        records = result.scalars().all()
        return [self._to_domain_turn(r) for r in records]
