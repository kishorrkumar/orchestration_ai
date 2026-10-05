"""FastAPI REST controller for prompt compiling, token counting, and voice linting."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from orchestration.application.prompts.service import PromptCompilerUseCase
from orchestration.domain.agent import Agent, AgentStatus
from orchestration.domain.prompt import HARD_TOKEN_LIMIT, RECOMMENDED_TOKEN_LIMIT
from orchestration.interfaces.http.dependencies import get_prompt_compiler
from orchestration.interfaces.http.schemas.prompt_schemas import (
    CompilePromptRequest,
    CompilePromptResponse,
    LintWarningResponse,
)

router = APIRouter(prefix="/v2/prompts", tags=["Prompts"])


@router.post("/compile", response_model=CompilePromptResponse)
async def compile_prompt(
    req: CompilePromptRequest,
    compiler: PromptCompilerUseCase = Depends(get_prompt_compiler),
) -> CompilePromptResponse:
    """Live compile an agent prompt, compute SentencePiece token count, and return voice lint hints."""
    # Synthetic lightweight agent model for compilation
    temp_agent = Agent(
        id="preview",
        name=req.agent_name,
        voice_id="natural_calm",
        greeting=req.greeting,
        agent_speaks_first=True,
        system_prompt=req.system_prompt,
        ending=req.ending,
        end_call_timeout_sec=2.0,
        silence_timeout_sec=12.0,
        max_duration_sec=600.0,
        timezone_str=req.timezone_str,
        status=AgentStatus.DRAFT,
    )

    compiled = compiler.compile(
        temp_agent,
        customer_name=req.customer_name,
        company=req.company,
    )

    return CompilePromptResponse(
        compiled_text=compiled.compiled_text,
        wrapped_text=compiled.wrapped_text,
        token_count=compiled.token_count,
        hard_limit=HARD_TOKEN_LIMIT,
        recommended_limit=RECOMMENDED_TOKEN_LIMIT,
        is_within_hard_limit=compiled.is_within_hard_limit,
        is_fast_start=compiled.is_fast_start,
        local_time_line=compiled.local_time_line,
        warnings=[
            LintWarningResponse(
                rule=w.rule,
                message=w.message,
                severity=w.severity,
                match=w.match,
            )
            for w in compiled.warnings
        ],
    )
