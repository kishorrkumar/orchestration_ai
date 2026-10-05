"""
FastAPI dependency injection wiring for clean architecture application services.
"""

from __future__ import annotations

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from orchestration.application.agents.service import AgentApplicationService
from orchestration.application.calls.service import CallApplicationService
from orchestration.application.prompts.service import PromptCompilerUseCase
from orchestration.db.session import async_session_factory
from orchestration.infrastructure.clock.system_clock import SystemClock
from orchestration.infrastructure.db.repositories.agent_repository import SqlAlchemyAgentRepository
from orchestration.infrastructure.db.repositories.session_repository import SqlAlchemySessionRepository
from orchestration.infrastructure.tokenizer.sentencepiece_adapter import SentencePieceTokenizerAdapter
from orchestration.shared.settings import settings

# Global shared infrastructure adapters
_clock = SystemClock()
_tokenizer = SentencePieceTokenizerAdapter(model_path=settings.tokenizer_path)
_prompt_compiler = PromptCompilerUseCase(tokenizer=_tokenizer, clock=_clock)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield database session with automated rollback on unhandled error."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_prompt_compiler() -> PromptCompilerUseCase:
    return _prompt_compiler


def get_agent_service(
    session: AsyncSession = Depends(get_db_session),
    compiler: PromptCompilerUseCase = Depends(get_prompt_compiler),
) -> AgentApplicationService:
    repo = SqlAlchemyAgentRepository(session)
    return AgentApplicationService(repository=repo, compiler=compiler, clock=_clock)


def get_call_service(
    session: AsyncSession = Depends(get_db_session),
) -> CallApplicationService:
    agent_repo = SqlAlchemyAgentRepository(session)
    session_repo = SqlAlchemySessionRepository(session)
    return CallApplicationService(session_repo=session_repo, agent_repo=agent_repo, clock=_clock)
