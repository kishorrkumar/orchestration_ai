"""
PersonaPlex S2S Prompt Compiler, Token Counter, and Voice Linter.
Computes local time via zoneinfo, renders {{variables}}, and counts tokens with SentencePiece.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import zoneinfo

import sentencepiece

from ..protocol.prompt import (
    IDEAL_SYSTEM_PROMPT_TOKENS,
    MAX_SYSTEM_PROMPT_TOKENS,
    wrap_system_prompt,
)

# Lazy-loaded SentencePiece singleton
_SP_TOKENIZER: Optional[sentencepiece.SentencePieceProcessor] = None
TOKENIZER_MODEL_PATH = Path("models/tokenizer_spm_32k_3.model")


def get_tokenizer() -> sentencepiece.SentencePieceProcessor:
    """Returns the cached SentencePiece tokenizer."""
    global _SP_TOKENIZER
    if _SP_TOKENIZER is None:
        if not TOKENIZER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found at {TOKENIZER_MODEL_PATH.resolve()}. "
                "Ensure models/tokenizer_spm_32k_3.model exists."
            )
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(str(TOKENIZER_MODEL_PATH))
        _SP_TOKENIZER = sp
    return _SP_TOKENIZER


def get_local_time_context(tz_name: str = "Asia/Kolkata", dt: Optional[datetime.datetime] = None) -> dict[str, str]:
    """
    Computes local weekday, time, and day-part for a given IANA timezone.
    Day-part boundaries:
      - 05:00 - 11:59: morning
      - 12:00 - 16:59: afternoon
      - 17:00 - 20:59: evening
      - 21:00 - 04:59: night
    """
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")

    now = dt.astimezone(tz) if dt else datetime.datetime.now(tz)
    weekday = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    # Clean leading 0 from 12-hour format: '03:15 PM' -> '3:15 PM'
    time_str = now.strftime("%I:%M %p").lstrip("0")
    if not time_str:
        time_str = now.strftime("%I:%M %p")

    hour = now.hour
    if 5 <= hour < 12:
        day_part = "morning"
    elif 12 <= hour < 17:
        day_part = "afternoon"
    elif 17 <= hour < 21:
        day_part = "evening"
    else:
        day_part = "night"

    time_line = f"It is {weekday}, {time_str} ({day_part}) for the caller."

    return {
        "timezone": tz_name,
        "weekday": weekday,
        "date": date_str,
        "current_time": time_str,
        "day_part": day_part,
        "time_line": time_line,
    }


@dataclass
class CompiledPrompt:
    """Output of the prompt compiler."""
    text: str
    token_count: int
    ideal_limit: int = IDEAL_SYSTEM_PROMPT_TOKENS
    hard_limit: int = MAX_SYSTEM_PROMPT_TOKENS
    can_publish: bool = True
    warnings: list[str] = field(default_factory=list)
    unrendered_variables: list[str] = field(default_factory=list)
    time_line: str = ""
    day_part: str = "morning"

    @property
    def formatted_prompt(self) -> str:
        return self.text


def count_tokens(text: str) -> int:
    """Counts tokens using the official 32k PersonaPlex SentencePiece model."""
    sp = get_tokenizer()
    return len(sp.encode(text))


def lint_prompt(
    system_prompt: str,
    greeting_text: str,
    ending_text: str,
    compiled_tokens: int,
    unrendered_vars: list[str],
) -> list[str]:
    """
    Lean voice-specific linter:
    - Token budget checks (>150 ideal, >350 hard limit)
    - Non-English character detection
    - Spoken markdown/bullets/emoji detection
    - Long lists of NEVER/ALWAYS rules
    - Missing goal / missing wrap-up
    """
    warnings = []

    # 1. Budget checks
    if compiled_tokens > MAX_SYSTEM_PROMPT_TOKENS:
        warnings.append(
            f"Prompt exceeds hard limit ({compiled_tokens}/{MAX_SYSTEM_PROMPT_TOKENS} tokens). "
            "Publish is blocked. Shorten system prompt to prevent connection handshake timeouts."
        )
    elif compiled_tokens > IDEAL_SYSTEM_PROMPT_TOKENS:
        warnings.append(
            f"Prompt is above recommended budget ({compiled_tokens}/{IDEAL_SYSTEM_PROMPT_TOKENS} tokens). "
            "Handshake will take ~2-4s. Aim for under 150 tokens for lowest latency."
        )

    # 2. Markdown / formatting
    full_text = f"{system_prompt} {greeting_text} {ending_text}"
    if re.search(r"(\*\*|__|\#\#|\* |- |\d+\. )", full_text):
        warnings.append("Remove markdown formatting (asterisks, bullet dashes, numbered lists) meant for spoken text.")

    # 3. Emojis
    if re.search(r"[\U00010000-\U0010ffff]", full_text):
        warnings.append("Remove emojis; S2S speech models may attempt to vocalize emoji unicode characters.")

    # 4. Long lists of NEVER / ALWAYS
    rule_counts = len(re.findall(r"\b(NEVER|ALWAYS|DO NOT)\b", full_text))
    if rule_counts >= 4:
        warnings.append(
            f"Found {rule_counts} strict 'NEVER/ALWAYS/DO NOT' rules. "
            "S2S models perform better with natural conversational prose than negative constraint lists."
        )

    # 5. Non-English script detection
    non_ascii_chars = [c for c in full_text if ord(c) > 127 and not (0x2018 <= ord(c) <= 0x201D)]
    if len(non_ascii_chars) > 2:
        warnings.append(
            "Non-English characters detected. PersonaPlex is strictly English-only and will produce phonetic hallucinations."
        )

    # 6. Unrendered variables
    if unrendered_vars:
        warnings.append(f"Unfilled variables detected: {', '.join(unrendered_vars)}. Fill them or remove braces before testing.")

    # 7. Missing goal
    if "goal" not in system_prompt.lower() and "help" not in system_prompt.lower():
        warnings.append("Consider explicitly stating the agent's goal (e.g., 'Your goal: ...') for clearer turn-taking.")

    return warnings


def compile_prompt(
    system_prompt: str,
    greeting_text: str,
    greeting_mode: str = "agent_first",
    ending_text: str = "",
    agent_name: str = "Assistant",
    timezone: str = "Asia/Kolkata",
    variables: Optional[dict[str, Any]] = None,
    caller_name: Optional[str] = None,
    customer_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    dt: Optional[datetime.datetime] = None,
    **kwargs: Any,
) -> CompiledPrompt:
    """
    Compiles agent fields into the exact PersonaPlex <system> ... <system> prompt.
    """
    time_ctx = get_local_time_context(timezone, dt=dt)
    var_dict = {
        "agent_name": agent_name,
        "current_time": time_ctx["current_time"],
        "weekday": time_ctx["weekday"],
        "date": time_ctx["date"],
        "day_part": time_ctx["day_part"],
    }
    if caller_name:
        var_dict["caller_name"] = caller_name
    if customer_name:
        var_dict["customer_name"] = customer_name
    if phone_number:
        var_dict["phone_number"] = phone_number

    if variables:
        for k, v in variables.items():
            var_dict[k] = str(v)
    for k, v in kwargs.items():
        var_dict[k] = str(v)

    def render_vars(text: str) -> str:
        for k, v in var_dict.items():
            text = text.replace(f"{{{{{k}}}}}", v)
        return text

    rendered_system = render_vars(system_prompt).strip()
    rendered_greeting = render_vars(greeting_text).strip()
    rendered_ending = render_vars(ending_text).strip()

    # Find any remaining unrendered variables
    all_combined = f"{rendered_system} {rendered_greeting} {rendered_ending}"
    unrendered = list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", all_combined)))

    # Greeting instruction
    if greeting_mode == "user_first":
        start_instruction = 'Start: Wait for the caller to speak first.'
    else:
        start_instruction = f'Start: Open the call by saying: "{rendered_greeting}"'

    # Ending instruction
    if rendered_ending:
        close_instruction = f'Close: When the conversation is done, say: "{rendered_ending}"'
    else:
        close_instruction = 'Close: When the conversation is done, say: "Thank you for calling. Goodbye!"'

    # Assemble the tight prompt
    body_lines = [
        rendered_system,
        time_ctx["time_line"],
        start_instruction,
        close_instruction,
    ]
    compiled_body = "\n".join([line for line in body_lines if line])
    final_prompt = wrap_system_prompt(compiled_body)

    token_cnt = count_tokens(final_prompt)
    warnings = lint_prompt(
        system_prompt=rendered_system,
        greeting_text=rendered_greeting,
        ending_text=rendered_ending,
        compiled_tokens=token_cnt,
        unrendered_vars=unrendered,
    )

    return CompiledPrompt(
        text=final_prompt,
        token_count=token_cnt,
        can_publish=(token_cnt <= MAX_SYSTEM_PROMPT_TOKENS),
        warnings=warnings,
        unrendered_variables=unrendered,
        time_line=time_ctx["time_line"],
        day_part=time_ctx["day_part"],
    )
