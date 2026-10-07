"""
Database Session Management and Connection Engine Factory.
Supports SQLite (local dev) and PostgreSQL (Neon Cloud / Production).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from . import models  # noqa: F401 - Ensure all models are registered on Base.metadata
from .base import Base

# Default to SQLite local database if DATABASE_URL is not set
DEFAULT_SQLITE_PATH = Path("data/platform.db")


def normalize_database_url(raw_url: str | None = None) -> str:
    """
    Normalizes a database URL to an async-compatible driver URL:
    - postgresql:// or postgresql+psycopg:// -> postgresql+asyncpg://
    - sqlite:/// -> sqlite+aiosqlite:///
    """
    if not raw_url:
        raw_url = os.environ.get("DATABASE_URL", "").strip()

    if not raw_url:
        DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

    # Handle Postgres normalization
    if raw_url.startswith("postgres://") or raw_url.startswith("postgresql://") or raw_url.startswith("postgresql+psycopg://"):
        parsed = urlparse(raw_url)
        # Convert scheme to postgresql+asyncpg
        new_scheme = "postgresql+asyncpg"

        # Sanitize query parameters for asyncpg (e.g., sslmode -> ssl, drop channel_binding)
        query_params = parse_qs(parsed.query)
        new_query = {}
        for k, v in query_params.items():
            if k == "sslmode":
                new_query["ssl"] = v[0]
            elif k == "channel_binding":
                continue  # asyncpg handles TLS internally
            else:
                new_query[k] = v[0]

        if "ssl" not in new_query and "neon.tech" in (parsed.hostname or ""):
            new_query["ssl"] = "require"

        new_query_str = urlencode(new_query)
        normalized = urlunparse((
            new_scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query_str,
            parsed.fragment,
        ))
        return normalized

    if raw_url.startswith("sqlite://") and not raw_url.startswith("sqlite+aiosqlite://"):
        path_part = raw_url[len("sqlite://"):]
        return f"sqlite+aiosqlite://{path_part}"

    return raw_url


# Global engine and session factory
_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        db_url = normalize_database_url()
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _async_engine = create_async_engine(
            db_url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


def async_session_factory() -> AsyncSession:
    """Return a new AsyncSession instance from the active session factory."""
    return get_session_factory()()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency for obtaining an async database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables in the database if they do not already exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.url.drivername.startswith("sqlite"):
            from sqlalchemy import text
            for col, col_type in [
                ("name", "VARCHAR(128) DEFAULT ''"),
                ("compiled_prompt", "TEXT DEFAULT ''"),
                ("engine", "VARCHAR(32) DEFAULT 'personaplex_s2s'"),
                ("language", "VARCHAR(32) DEFAULT 'en'"),
                ("pipeline_json", "TEXT DEFAULT '{}'"),
            ]:
                try:
                    await conn.execute(text(f"ALTER TABLE agent_versions ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass

            for col, col_type in [
                ("draft_engine", "VARCHAR(32) DEFAULT 'personaplex_s2s'"),
                ("draft_language", "VARCHAR(32) DEFAULT 'en'"),
                ("draft_pipeline_json", "TEXT DEFAULT '{}'"),
            ]:
                try:
                    await conn.execute(text(f"ALTER TABLE agents ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass

            for col, col_type in [
                ("engine", "VARCHAR(32) DEFAULT 'personaplex_s2s'"),
                ("providers_used_json", "TEXT DEFAULT '{}'"),
            ]:
                try:
                    await conn.execute(text(f"ALTER TABLE call_sessions ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass

            for col, col_type in [
                ("eot_ms", "FLOAT"),
                ("stt_ms", "FLOAT"),
                ("llm_ttft_ms", "FLOAT"),
                ("tts_ttfa_ms", "FLOAT"),
                ("voice_to_voice_ms", "FLOAT"),
            ]:
                try:
                    await conn.execute(text(f"ALTER TABLE call_turns ADD COLUMN {col} {col_type}"))
                except Exception:
                    pass


async def close_db() -> None:
    """Cleanly dispose of database connection pool."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
