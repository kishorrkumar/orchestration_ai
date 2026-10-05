"""
SQLAlchemy 2.0 repository implementing domain CallSessionRepository protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

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
        end_r = None
        if rec.end_reason:
            try:
                end_r = EndReason(rec.end_reason)
            except ValueError:
                end_r = EndReason.AGENT_CLOSED
        status = CallStatus.COMPLETED if rec.ended_at else CallStatus.CONNECTED
        turn_cnt = len(rec.turns) if ("turns" in rec.__dict__ and rec.turns is not None) else 0

        return CallSession(
            id=rec.id,
            agent_id=rec.agent_id,
            agent_version=1,
            status=status,
            client_type="web",
            duration_sec=rec.duration_sec,
            end_reason=end_r,
            total_turns=turn_cnt,
            created_at=rec.started_at,
            ended_at=rec.ended_at,
        )

    @staticmethod
    def _to_domain_turn(rec: db_models.CallTurn) -> CallTurn:
        speaker = TurnSpeaker.AGENT if rec.role == "agent" else TurnSpeaker.CALLER
        return CallTurn(
            id=rec.id,
            session_id=rec.session_id,
            turn_index=rec.idx,
            speaker=speaker,
            text=rec.text,
            started_at_sec=rec.started_ms / 1000.0,
            ended_at_sec=rec.started_ms / 1000.0,
            latency_ms=None,
            created_at=datetime.now(UTC),
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
        stmt = select(db_models.CallSession).order_by(db_models.CallSession.started_at.desc()).limit(limit).offset(offset)
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
                agent_id=session.agent_id,
                agent_version_id=None,
                duration_sec=session.duration_sec,
                end_reason=session.end_reason.value if session.end_reason else "",
                started_at=session.created_at,
                ended_at=session.ended_at,
                ttfa_ms=0.0,
                handshake_ms=0.0,
            )
            self.session.add(record)
        else:
            record.duration_sec = session.duration_sec
            record.end_reason = session.end_reason.value if session.end_reason else ""
            record.ended_at = session.ended_at

        await self.session.commit()
        await self.session.refresh(record)
        return self._to_domain_session(record)

    async def save_turn(self, turn: CallTurn) -> CallTurn:
        record = db_models.CallTurn(
            id=turn.id,
            session_id=turn.session_id,
            idx=turn.turn_index,
            role="agent" if turn.speaker == TurnSpeaker.AGENT else "user",
            text=turn.text,
            started_ms=turn.started_at_sec * 1000.0,
        )
        self.session.add(record)
        await self.session.commit()
        return turn

    async def list_turns(self, session_id: str) -> list[CallTurn]:
        stmt = (
            select(db_models.CallTurn)
            .where(db_models.CallTurn.session_id == session_id)
            .order_by(db_models.CallTurn.idx.asc())
        )
        result = await self.session.execute(stmt)
        records = result.scalars().all()
        return [self._to_domain_turn(r) for r in records]
