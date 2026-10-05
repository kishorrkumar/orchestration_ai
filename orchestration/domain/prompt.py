"""
Domain rules for prompt formatting, time context injection, and voice linting.
Verified delimiter: both open and close tags are literally `<system>`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SYSTEM_TAG_OPEN = "<system>"
SYSTEM_TAG_CLOSE = "<system>"
HARD_TOKEN_LIMIT = 350
RECOMMENDED_TOKEN_LIMIT = 150


@dataclass
class LintWarning:
    rule: str
    message: str
    severity: str  # "warning" or "suggestion"
    match: str | None = None


@dataclass
class CompiledPrompt:
    """Result of prompt compilation with token telemetry."""
    raw_prompt: str
    compiled_text: str
    wrapped_text: str
    token_count: int
    timezone_str: str
    local_time_line: str
    warnings: list[LintWarning] = field(default_factory=list)

    @property
    def is_within_hard_limit(self) -> bool:
        return self.token_count <= HARD_TOKEN_LIMIT

    @property
    def is_fast_start(self) -> bool:
        return self.token_count <= RECOMMENDED_TOKEN_LIMIT


class PromptLinter:
    """Pure domain linter checking written conventions that sound unnatural when spoken."""

    _URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    _MARKDOWN_HEADER = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
    _BULLET_LIST = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
    _NUMBERED_LIST = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
    _PARENTHETICAL = re.compile(r"\([^)]*\)")
    _ALL_CAPS_ACRONYM = re.compile(r"\b[A-Z]{3,}\b")

    @classmethod
    def lint(cls, text: str) -> list[LintWarning]:
        warnings: list[LintWarning] = []

        if cls._URL_PATTERN.search(text):
            warnings.append(
                LintWarning(
                    rule="no_urls",
                    message="URLs sound unnatural when spoken. Speak domain names plainly (e.g. 'our website') instead.",
                    severity="warning",
                )
            )

        if cls._MARKDOWN_HEADER.search(text) or cls._MARKDOWN_LINK.search(text):
            warnings.append(
                LintWarning(
                    rule="no_markdown",
                    message="Markdown headers or links confuse acoustic tokenizers. Use plain conversational prose.",
                    severity="warning",
                )
            )

        if cls._BULLET_LIST.search(text) or cls._NUMBERED_LIST.search(text):
            warnings.append(
                LintWarning(
                    rule="no_bullet_points",
                    message="Bullet points induce robotic rhythm. Frame lists into connected conversational sentences.",
                    severity="suggestion",
                )
            )

        if cls._PARENTHETICAL.search(text):
            warnings.append(
                LintWarning(
                    rule="no_parentheticals",
                    message="Parentheticals are often mispronounced or spoken without natural cadence.",
                    severity="suggestion",
                )
            )

        acronyms = cls._ALL_CAPS_ACRONYM.findall(text)
        if acronyms:
            sample = ", ".join(list(set(acronyms))[:3])
            warnings.append(
                LintWarning(
                    rule="acronym_clarity",
                    message=f"All-caps acronyms ({sample}) may be read letter-by-letter. Separate letters with hyphens if needed.",
                    severity="suggestion",
                    match=sample,
                )
            )

        return warnings


def compute_local_time_context(
    tz_str: str,
    reference_dt: datetime | None = None,
) -> tuple[str, str]:
    """
    Format local date and time string from timezone.
    Returns (local_time_line, day_part).
    """
    try:
        tz = ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")

    now = reference_dt or datetime.now(UTC)
    local_dt = now.astimezone(tz)

    hour = local_dt.hour
    if 5 <= hour < 12:
        day_part = "morning"
    elif 12 <= hour < 17:
        day_part = "afternoon"
    elif 17 <= hour < 21:
        day_part = "evening"
    else:
        day_part = "night"

    formatted_time = local_dt.strftime("%A, %I:%M %p").replace(" 0", " ")
    local_time_line = f"Current local time: {formatted_time}, {day_part} ({tz_str})."
    return local_time_line, day_part


def wrap_system_prompt(content: str) -> str:
    """
    Wrap compiled prompt with verified PersonaPlex system tags.
    Both tags are literally `<system>`, verified upstream.
    """
    cleaned = content.strip()
    if cleaned.startswith(SYSTEM_TAG_OPEN) and cleaned.endswith(SYSTEM_TAG_CLOSE):
        return cleaned
    return f"{SYSTEM_TAG_OPEN} {cleaned} {SYSTEM_TAG_CLOSE}"
