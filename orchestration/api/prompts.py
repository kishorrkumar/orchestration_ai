"""
Prompt Compilation and Timezone API Router.
"""

from __future__ import annotations

import zoneinfo
from typing import List

from fastapi import APIRouter

from ..prompts.compiler import (
    IDEAL_SYSTEM_PROMPT_TOKENS,
    MAX_SYSTEM_PROMPT_TOKENS,
    compile_prompt,
    get_local_time_context,
)
from .schemas import (
    CompilePromptRequest,
    CompilePromptResponse,
    TimezoneOption,
)

router = APIRouter(prefix="/v1", tags=["Prompts & Timezones"])

# Top common IANA timezones plus all available in Python zoneinfo
COMMON_TIMEZONES = [
    "Asia/Kolkata",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Dubai",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Australia/Sydney",
    "UTC",
]


@router.post("/prompts/compile", response_model=CompilePromptResponse)
@router.post("/prompts/lint", response_model=CompilePromptResponse)
async def compile_prompt_endpoint(req: CompilePromptRequest):
    """
    Compiles an agent prompt, resolves time and variables, counts tokens
    using the official SentencePiece model, and runs voice linter checks.
    """
    res = compile_prompt(
        system_prompt=req.system_prompt,
        greeting_text=req.greeting_text,
        greeting_mode=req.greeting_mode,
        ending_text=req.ending_text,
        agent_name=req.agent_name,
        timezone=req.timezone,
        variables=req.variables,
    )
    return CompilePromptResponse(
        compiled_text=res.text,
        token_count=res.token_count,
        ideal_limit=IDEAL_SYSTEM_PROMPT_TOKENS,
        hard_limit=MAX_SYSTEM_PROMPT_TOKENS,
        can_publish=res.can_publish,
        warnings=res.warnings,
        unrendered_variables=res.unrendered_variables,
        time_line=res.time_line,
        day_part=res.day_part,
    )


@router.get("/timezones", response_model=List[TimezoneOption])
async def list_timezones():
    """Returns a list of searchable IANA timezones with current live time preview."""
    results = []
    # Include common first, then sorted unique set
    all_zones = COMMON_TIMEZONES + sorted(list(zoneinfo.available_timezones()))
    seen = set()

    for tz in all_zones:
        if tz in seen:
            continue
        seen.add(tz)
        try:
            ctx = get_local_time_context(tz)
            results.append(
                TimezoneOption(
                    name=tz,
                    current_time=ctx["current_time"],
                    weekday=ctx["weekday"],
                    day_part=ctx["day_part"],
                )
            )
        except Exception:
            continue

        if len(results) >= 60:
            break

    return results
