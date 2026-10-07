"""
PersonaPlex S2S Prompt Compiler, Token Counter, and Voice Linter.
Computes local time via zoneinfo, renders {{variables}}, and counts tokens with SentencePiece.
"""

from __future__ import annotations

import datetime
import re
import glob
import logging
import os
import re
import zoneinfo
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException
import sentencepiece

from ..protocol.prompt import (
    IDEAL_SYSTEM_PROMPT_TOKENS,
    MAX_SYSTEM_PROMPT_TOKENS,
    wrap_system_prompt,
)

logger = logging.getLogger("orchestration.prompts.compiler")

# Built-in defaults for template placeholders so un-configured agents never produce literal {{...}}
DEFAULT_VARIABLE_VALUES: dict[str, str] = {
    "company": "BrightNet",
    "customer_name": "the caller",
    "caller_name": "the caller",
    "agent_name": "Assistant",
    "goal": "assist the caller with their request",
}


class TemplateResolutionError(HTTPException):
    """Raised when required template variables remain unrendered."""

    def __init__(self, missing_variables: list[str] | set[str]):
        var_names = ", ".join(sorted(missing_variables))
        super().__init__(
            status_code=422,
            detail=f"Unresolved template variables: {var_names}. Please provide values or defaults before publishing or compiling.",
        )
        self.missing_variables = sorted(list(missing_variables))


# Lazy-loaded SentencePiece singleton
_SP_TOKENIZER: sentencepiece.SentencePieceProcessor | None = None
_TOKENIZER_CHECKED: bool = False


def _find_tokenizer_path() -> Path | None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate_paths = [
        Path("models/tokenizer_spm_32k_3.model"),
        repo_root / "models" / "tokenizer_spm_32k_3.model",
    ]
    for p in candidate_paths:
        if p.exists():
            return p

    # Search in Hugging Face cache directories
    hf_dirs = [
        os.environ.get("HF_HOME"),
        "/workspace/huggingface",
        os.path.expanduser("~/.cache/huggingface"),
    ]
    for hf_dir in hf_dirs:
        if hf_dir and os.path.isdir(hf_dir):
            matches = glob.glob(f"{hf_dir}/**/tokenizer_spm_32k_3.model", recursive=True)
            if matches:
                return Path(matches[0])

    return None


def get_tokenizer() -> sentencepiece.SentencePieceProcessor | None:
    """Returns the cached SentencePiece tokenizer if available, else None."""
    global _SP_TOKENIZER, _TOKENIZER_CHECKED
    if not _TOKENIZER_CHECKED:
        _TOKENIZER_CHECKED = True
        model_path = _find_tokenizer_path()
        if model_path:
            try:
                sp = sentencepiece.SentencePieceProcessor()
                sp.load(str(model_path))
                _SP_TOKENIZER = sp
                logger.info(f"Loaded SentencePiece tokenizer from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load SentencePiece model from {model_path}: {e}")
                _SP_TOKENIZER = None
        else:
            logger.info("SentencePiece model not found on disk or in HF cache. Using BPE estimator.")
            _SP_TOKENIZER = None

    return _SP_TOKENIZER


def sanitize_prompt_text(text: str) -> tuple[str, list[str]]:
    """
    Strips markdown formatting, bullets, emojis, and rigid quotation-mark scripts from prompts.
    Returns the cleaned plain prose and a list of warnings describing what was stripped.
    """
    warnings: list[str] = []
    cleaned = text

    # Protect template placeholders like {{company}} or {{agent_name}} during markdown cleanup
    placeholders: list[str] = []
    def _save_ph(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"__TEMPLATE_PH_{len(placeholders) - 1}__"

    cleaned = re.sub(r"\{\{[a-zA-Z0-9_]+\}\}", _save_ph, cleaned)

    # 1. Emojis
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50]")
    if emoji_pattern.search(cleaned):
        cleaned = emoji_pattern.sub("", cleaned)
        warnings.append("Removed emojis from prompt (speech models may hallucinate emoji characters)")

    # 2. Markdown headers (e.g. ## Header)
    if re.search(r"^\s*#+\s*", cleaned, flags=re.MULTILINE):
        cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
        warnings.append("Removed markdown heading symbols (#)")

    # 3. Bold/italics (**bold**, *italic*, __bold__, _italic_)
    if re.search(r"\*\*([^*]+)\*\*", cleaned):
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        warnings.append("Removed markdown bold formatting (**)")
    if re.search(r"\*([^*]+)\*", cleaned):
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        warnings.append("Removed markdown italic formatting (*)")
    if re.search(r"(?<!\w)__([^_]+)__(?!\w)", cleaned):
        cleaned = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", cleaned)
        warnings.append("Removed markdown bold formatting (__) ")
    if re.search(r"(?<!\w)_([^_]+)_(?!\w)", cleaned):
        cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
        warnings.append("Removed markdown italic formatting (_)")

    # 4. Bullet points and numbered lists
    if re.search(r"^\s*[-*+]\s+", cleaned, flags=re.MULTILINE):
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        warnings.append("Removed bullet point markers (- / * / +)")

    if re.search(r"^\s*\d+\.\s+", cleaned, flags=re.MULTILINE):
        cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
        warnings.append("Removed numbered list markers")

    # 5. Rigid scripted quotes (e.g. Start: Open the call by saying: "..." or Close: When the conversation is done, say: "...")
    if re.search(r"(?:Start|Close):\s*(?:Open the call by saying|When the conversation is done, say)?[:\s]*\"[^\"]*\"", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"(?:Start|Close):\s*(?:Open the call by saying|When the conversation is done, say)?[:\s]*\"([^\"]*)\"", r"\1", cleaned, flags=re.IGNORECASE)
        warnings.append("Removed scripted quotation commands ('Start:' / 'Close:')")

    # Restore placeholders
    for idx, ph in enumerate(placeholders):
        cleaned = cleaned.replace(f"__TEMPLATE_PH_{idx}__", ph)

    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned).strip()

    return cleaned, warnings


def get_local_time_context(tz_name: str = "Asia/Kolkata", dt: datetime.datetime | None = None) -> dict[str, str]:
    """
    Computes local weekday, time, and day-part for a given IANA timezone.
    Phased naturally for PersonaPlex conversational style:
    "It is Wednesday morning for the caller."
    """
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")

    now = dt.astimezone(tz) if dt else datetime.datetime.now(tz)
    weekday = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
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

    time_line = f"It is {weekday} {day_part} for the caller."

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

    @property
    def is_critical_overflow(self) -> bool:
        return not self.can_publish or self.token_count > self.hard_limit


def count_tokens(text: str) -> int:
    """Counts tokens using official 32k PersonaPlex SentencePiece model or BPE estimator."""
    sp = get_tokenizer()
    if sp is not None:
        try:
            return len(sp.encode(text))
        except Exception as e:
            logger.warning(f"SentencePiece encoding error: {e}")

    # Fallback BPE estimator heuristic (~3.8 chars per token for English prose)
    cleaned = text.strip()
    if not cleaned:
        return 0
    words = len(cleaned.split())
    chars = len(cleaned)
    return max(1, int(chars / 3.8 + words * 0.1))


def lint_prompt(
    system_prompt: str,
    greeting_text: str,
    ending_text: str,
    compiled_tokens: int,
    unrendered_vars: list[str],
) -> list[str]:
    """Lean voice-specific linter."""
    warnings = []

    # 1. Budget checks
    if compiled_tokens > MAX_SYSTEM_PROMPT_TOKENS:
        warnings.append(
            f"Prompt exceeds hard limit ({compiled_tokens}/{MAX_SYSTEM_PROMPT_TOKENS} tokens). "
            "Publish is blocked. Shorten system prompt to prevent connection handshake timeouts."
        )
    elif compiled_tokens > 300:
        warnings.append(
            f"Prompt is above recommended conversational budget ({compiled_tokens}/200 tokens). "
            "Handshake will take ~2-4s. Aim for under 200 tokens for lowest latency."
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

    return warnings


def compile_prompt(
    system_prompt: str,
    greeting_text: str = "",
    greeting_mode: str = "agent_first",
    ending_text: str = "",
    agent_name: str = "Assistant",
    timezone: str = "Asia/Kolkata",
    variables: dict[str, Any] | None = None,
    caller_name: str | None = None,
    customer_name: str | None = None,
    phone_number: str | None = None,
    dt: datetime.datetime | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> CompiledPrompt:
    """
    Compiles agent fields into the exact PersonaPlex <system> ... <system> prompt.
    Applies strict variable resolution, prompt sanitization, natural prose assembly,
    and token counting.
    """
    time_ctx = get_local_time_context(timezone, dt=dt)

    # 1. Sanitize text inputs
    clean_sys, sys_warn = sanitize_prompt_text(system_prompt)
    clean_greet, greet_warn = sanitize_prompt_text(greeting_text)
    clean_end, end_warn = sanitize_prompt_text(ending_text)
    collected_warnings = list(dict.fromkeys(sys_warn + greet_warn + end_warn))

    # 2. Build full variable dictionary starting from defaults
    var_dict = DEFAULT_VARIABLE_VALUES.copy()
    var_dict["agent_name"] = agent_name
    var_dict["current_time"] = time_ctx["current_time"]
    var_dict["weekday"] = time_ctx["weekday"]
    var_dict["date"] = time_ctx["date"]
    var_dict["day_part"] = time_ctx["day_part"]

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

    rendered_system = render_vars(clean_sys).strip()
    rendered_greeting = render_vars(clean_greet).strip()
    rendered_ending = render_vars(clean_end).strip()

    # 3. Check for any remaining unrendered variables
    all_combined = f"{rendered_system} {rendered_greeting} {rendered_ending}"
    unrendered = sorted(list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", all_combined))))
    has_stray_braces = ("{{" in all_combined) or ("}}" in all_combined)

    if strict and (unrendered or has_stray_braces):
        missing = unrendered if unrendered else ["unresolved_template_variable"]
        raise TemplateResolutionError(missing)

    # 4. Assemble natural conversational scenario in plain prose (no scripted Start:/Close: quotes)
    body_lines: list[str] = [rendered_system, time_ctx["time_line"]]

    if greeting_mode == "user_first":
        body_lines.append("The caller will speak first. Listen before responding.")
    elif greeting_mode == "agent_first":
        if rendered_greeting and rendered_greeting.lower() not in rendered_system.lower() and "greet" not in rendered_system.lower():
            body_lines.append(f"When the call connects, greet the caller warmly: {rendered_greeting}")

    if rendered_ending and rendered_ending.lower() not in rendered_system.lower() and "goodbye" not in rendered_system.lower():
        body_lines.append(f"When concluding the call: {rendered_ending}")

    compiled_body = "\n\n".join([line for line in body_lines if line])
    final_prompt = wrap_system_prompt(compiled_body)

    # Guarantee no literal {{ or }} reaches the model
    if strict and ("{{" in final_prompt or "}}" in final_prompt):
        raise TemplateResolutionError(["unresolved_template_variable"])

    token_cnt = count_tokens(final_prompt)
    linter_warnings = lint_prompt(
        system_prompt=rendered_system,
        greeting_text=rendered_greeting,
        ending_text=rendered_ending,
        compiled_tokens=token_cnt,
        unrendered_vars=unrendered,
    )
    for lw in linter_warnings:
        if lw not in collected_warnings:
            collected_warnings.append(lw)

    return CompiledPrompt(
        text=final_prompt,
        token_count=token_cnt,
        can_publish=(token_cnt <= MAX_SYSTEM_PROMPT_TOKENS and len(unrendered) == 0),
        warnings=collected_warnings,
        unrendered_variables=unrendered,
        time_line=time_ctx["time_line"],
        day_part=time_ctx["day_part"],
    )

