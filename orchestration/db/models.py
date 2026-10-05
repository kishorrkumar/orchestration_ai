"""
Trimmed Relational Database Models for Lean S2S Voice Agent Platform.
Designed for SQLAlchemy 2.0 with full support for SQLite and Neon PostgreSQL.
"""

from __future__ import annotations

import datetime
from typing import List

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, generate_prefixed_id


class Workspace(Base, TimestampMixin):
    """Default Workspace container."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default="wks_default")
    name: Mapped[str] = mapped_column(String(128), default="Default Workspace", nullable=False)
    slug: Mapped[str] = mapped_column(String(64), default="default", unique=True, nullable=False)

    agents: Mapped[List[Agent]] = relationship("Agent", back_populates="workspace", cascade="all, delete-orphan")


class Agent(Base, TimestampMixin):
    """
    Top-level Agent entity.
    Maintains mutable working draft fields and points to the published immutable version.
    """

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("agt")
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), default="wks_default", nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)  # Also used as {{agent_name}}
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True, nullable=False)  # draft | published | archived
    published_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_version_no: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Draft working configuration (autosaved)
    draft_voice_id: Mapped[str] = mapped_column(String(64), default="NATF0.pt", nullable=False)
    draft_greeting_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    draft_greeting_mode: Mapped[str] = mapped_column(String(32), default="agent_first", nullable=False)  # agent_first | user_first
    draft_system_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    draft_ending_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    draft_end_silence_sec: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    draft_max_duration_sec: Mapped[int] = mapped_column(Integer, default=600, nullable=False)
    draft_timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata", nullable=False)

    workspace: Mapped[Workspace] = relationship("Workspace", back_populates="agents")
    versions: Mapped[List[AgentVersion]] = relationship(
        "AgentVersion",
        back_populates="agent",
        cascade="all, delete-orphan",
        order_by="desc(AgentVersion.version_no)",
        lazy="selectin",
    )
    sessions: Mapped[List[CallSession]] = relationship("CallSession", back_populates="agent")


class AgentVersion(Base):
    """
    Immutable Version Snapshot of an agent.
    Pinned when publishing; cannot be modified.
    """

    __tablename__ = "agent_versions"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ver")
    )
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), index=True, nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(128), default="", nullable=False)

    voice_id: Mapped[str] = mapped_column(String(64), nullable=False)
    greeting_text: Mapped[str] = mapped_column(Text, nullable=False)
    greeting_mode: Mapped[str] = mapped_column(String(32), default="agent_first", nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    ending_text: Mapped[str] = mapped_column(Text, nullable=False)
    end_silence_sec: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    max_duration_sec: Mapped[int] = mapped_column(Integer, default=600, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata", nullable=False)

    compiled_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    compiled_token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    change_note: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC),
        server_default=func.now(),
        nullable=False,
    )

    agent: Mapped[Agent] = relationship("Agent", back_populates="versions")
    sessions: Mapped[List[CallSession]] = relationship("CallSession", back_populates="version")

    __table_args__ = (
        Index("idx_agent_version_unique", "agent_id", "version_no", unique=True),
    )


class CallSession(Base):
    """Call Session record tracking overall call telemetry."""

    __tablename__ = "call_sessions"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("ses")
    )
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"), index=True, nullable=False)
    agent_version_id: Mapped[str | None] = mapped_column(ForeignKey("agent_versions.id", ondelete="SET NULL"), nullable=True)

    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC),
        server_default=func.now(),
        nullable=False,
    )
    ended_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    end_reason: Mapped[str] = mapped_column(String(64), default="", nullable=False)  # agent_closed | silence_timeout | max_duration | user_hangup | error
    ttfa_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    handshake_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    agent: Mapped[Agent] = relationship("Agent", back_populates="sessions")
    version: Mapped[AgentVersion | None] = relationship("AgentVersion", back_populates="sessions")
    turns: Mapped[List[CallTurn]] = relationship(
        "CallTurn", back_populates="session", cascade="all, delete-orphan", order_by="CallTurn.idx"
    )


class CallTurn(Base):
    """Text transcript turn recorded from opcode 0x02 text tokens."""

    __tablename__ = "call_turns"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: generate_prefixed_id("trn")
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("call_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | agent
    text: Mapped[str] = mapped_column(Text, nullable=False)
    started_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    session: Mapped[CallSession] = relationship("CallSession", back_populates="turns")
