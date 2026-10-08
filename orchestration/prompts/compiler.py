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
try:
    import sentencepiece
except ImportError:
    sentencepiece = None  # type: ignore

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


def get_tokenizer() -> Any | None:
    """Returns the cached SentencePiece tokenizer if available, else None."""
    global _SP_TOKENIZER, _TOKENIZER_CHECKED
    if sentencepiece is None:
        return None
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
    if re.search(r"(?:Start|Close):\s*(?:Open the call by saying|When the conversation is done, say)?[:\s]*['\"][^'\"]*['\"]", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"(?:Start|Close):\s*(?:Open the call by saying|When the conversation is done, say)?[:\s]*['\"]([^'\"]*)['\"]", r"\1", cleaned, flags=re.IGNORECASE)
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
        try:
            tz = zoneinfo.ZoneInfo("UTC")
        except Exception:
            tz = datetime.timezone.utc

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

    # 7. Rigid scripted quotes
    if re.search(r"\b(say exactly|repeat word for word|respond word for word)\b", full_text, flags=re.IGNORECASE):
        warnings.append("Avoid scripted quotes ('say exactly...'); PersonaPlex performs better with conversational role descriptions than rigid verbatim scripts.")

    # 8. Step counts / numbered sequences
    if re.search(r"\b(step\s*\d+|phase\s*\d+)\b", full_text, flags=re.IGNORECASE):
        warnings.append("Avoid rigid step-by-step numbers ('Step 1, Step 2'); describe the persona's conversational flow in natural prose.")

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
    strict: bool = False,
    **kwargs: Any,
) -> CompiledPrompt:
    """
    Compiles agent prompt into the exact PersonaPlex <system> ... <system> prompt.
    Removes variable dependencies and delivers clean, pure system prompt instructions.
    """
    time_ctx = get_local_time_context(timezone, dt=dt)

    # 1. Sanitize text inputs
    clean_sys, sys_warn = sanitize_prompt_text(system_prompt)
    collected_warnings = list(dict.fromkeys(sys_warn))

    # 2. Variable handling:
    if strict:
        raw_vars = sorted(list(set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", system_prompt))))
        provided_vars = set((variables or {}).keys()) | set(DEFAULT_VARIABLE_VALUES.keys())
        missing = [v for v in raw_vars if v not in provided_vars]
        if missing:
            raise TemplateResolutionError(missing)

    if variables:
        for k, v in variables.items():
            clean_sys = clean_sys.replace(f"{{{{{k}}}}}", str(v))
    for k, v in DEFAULT_VARIABLE_VALUES.items():
        clean_sys = clean_sys.replace(f"{{{{{k}}}}}", v)

    # Ensure zero stray brackets reach the model: {{company}} -> company
    clean_sys = re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", r"\1", clean_sys)
    clean_sys = clean_sys.replace("{{", "").replace("}}", "")

    # 3. Anchor Agent Identity and anti-leak rules for real-time S2S
    eff_name = agent_name.strip() if agent_name else "Ananya"
    # Remove Moshi default trigger phrase if present
    clean_sys = re.sub(r"^You enjoy having (a )?good conversations?\.?\s*", "", clean_sys, flags=re.IGNORECASE)

    # If the prompt does not establish agent identity, explicitly anchor it
    if eff_name.lower() not in clean_sys.lower():
        identity_prefix = (
            f"You are {eff_name}, a friendly sales rep at Snapserve. "
            f"Your name is {eff_name}. You represent Snapserve. You are NEVER Moshi. "
            "Speak casually in 1-2 short sentences. Acknowledge before asking ('Got it', 'Makes sense'). "
            "Ask ONE question at a time. Never repeat questions you already asked.\n"
        )
        clean_sys = identity_prefix + clean_sys
    else:
        # Reinforce anti-Moshi identity boundary
        clean_sys = f"You are {eff_name} at Snapserve, never Moshi.\n" + clean_sys

    compiled_body = clean_sys.strip()
    final_prompt = wrap_system_prompt(compiled_body)

    token_cnt = count_tokens(final_prompt)
    if token_cnt > MAX_SYSTEM_PROMPT_TOKENS:
        collected_warnings.append(
            f"Prompt exceeds recommended model token budget ({token_cnt}/{MAX_SYSTEM_PROMPT_TOKENS} tokens)."
        )

    linter_warnings = lint_prompt(
        system_prompt=compiled_body,
        greeting_text="",
        ending_text="",
        compiled_tokens=token_cnt,
        unrendered_vars=[],
    )
    for lw in linter_warnings:
        if lw not in collected_warnings:
            collected_warnings.append(lw)

    return CompiledPrompt(
        text=final_prompt,
        token_count=token_cnt,
        can_publish=(token_cnt <= MAX_SYSTEM_PROMPT_TOKENS),
        warnings=collected_warnings,
        unrendered_variables=[],
        time_line=time_ctx.get("time_line", ""),
        day_part=time_ctx.get("day_part", "morning"),
    )

