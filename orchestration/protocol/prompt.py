"""
Canonical System Prompt Delimiters and Constraints for NVIDIA PersonaPlex 7B.
Verified from upstream moshi/server.py and moshi/offline.py.
"""

from __future__ import annotations

# Exact, verified upstream delimiter (both opening and closing tags are literally "<system>")
PERSONAPLEX_SYSTEM_DELIMITER = "<system>"

# Token budget constants (measured via SentencePiece tokenizer_spm_32k_3.model)
IDEAL_SYSTEM_PROMPT_TOKENS = 150
MAX_SYSTEM_PROMPT_TOKENS = 350


def wrap_system_prompt(content: str) -> str:
    """
    Wraps prompt content in the exact upstream PersonaPlex format:
    "<system> {cleaned_content} <system>"
    """
    cleaned = content.strip()
    prefix = f"{PERSONAPLEX_SYSTEM_DELIMITER} "
    suffix = f" {PERSONAPLEX_SYSTEM_DELIMITER}"

    if cleaned.startswith(PERSONAPLEX_SYSTEM_DELIMITER):
        cleaned = cleaned[len(PERSONAPLEX_SYSTEM_DELIMITER):].strip()
    if cleaned.endswith(PERSONAPLEX_SYSTEM_DELIMITER):
        cleaned = cleaned[:-len(PERSONAPLEX_SYSTEM_DELIMITER)].strip()

    return f"{prefix}{cleaned}{suffix}"
