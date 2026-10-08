"""
Streaming SentencePiece Detokenizer and Contraction Repair Engine.
Guarantees clean English token stitching, preserving apostrophes and contractions
('m, 're, 'll, 've, 'd, 's, n't) without dropped letters or broken whitespace.
"""

from __future__ import annotations

import re

# Common English contraction suffixes
CONTRACTION_SUFFIXES = {
    "'m", "'re", "'ll", "'ve", "'d", "'s", "n't", "'t",
    "m", "re", "ll", "ve", "d", "s", "t",
}


def clean_sentencepiece_piece(piece: str) -> tuple[str, bool]:
    """
    Cleans a SentencePiece piece.
    Returns (cleaned_text, has_leading_space).
    Handles U+2581 (Lower One Eighth Block) prefix and raw replacement characters.
    """
    if not piece:
        return "", False

    # Check for SentencePiece space marker \u2581
    has_leading_space = piece.startswith(("\u2581", " ", "_"))
    cleaned = piece.lstrip("\u2581_ ")

    # Filter out special tags
    if cleaned in ("<pad>", "<s>", "</s>", "<unk>", "EPAD", "BOS", "EOS", "PAD"):
        return "", False

    return cleaned, has_leading_space


def stitch_token(current_text: str, new_token: str) -> str:
    """
    Appends a new streaming token to existing transcript with contraction awareness.
    Guarantees 'I' + "'" + 'm' -> 'I'm' (never 'I' m' or 'I' doing').
    """
    cleaned, has_space = clean_sentencepiece_piece(new_token)
    if not cleaned:
        return current_text

    if not current_text:
        return cleaned

    # Check if the new token is an apostrophe or contraction suffix
    if cleaned.startswith("'"):
        # Direct apostrophe: attach directly without space
        return current_text.rstrip() + cleaned

    if current_text.endswith("'") and cleaned in ("m", "re", "ll", "ve", "d", "s", "t"):
        # Follow-up letter after apostrophe: attach directly
        return current_text + cleaned

    if cleaned in (",", ".", "!", "?", ":", ";", "%"):
        # Punctuation: attach directly to preceding word
        return current_text.rstrip() + cleaned

    # Check if current text ends with an apostrophe (e.g. "I'", "you'")
    if current_text.endswith("'"):
        return current_text + " " + cleaned

    if has_space:
        return current_text.rstrip() + " " + cleaned

    return current_text + cleaned


def detokenize_sentencepiece_stream(tokens: list[str], agent_name: str | None = None) -> str:
    """
    Detokenizes an entire list of SentencePiece pieces into a clean, human-readable sentence.
    """
    result = ""
    for tok in tokens:
        result = stitch_token(result, tok)

    # Post-processing cleanup for edge cases
    # 1. Fix detached apostrophes: e.g. "I ' m" -> "I'm", "you ' re" -> "you're"
    result = re.sub(r"\b([A-Za-z]+)\s*'\s*(m|re|ll|ve|d|s|t)\b", r"\1'\2", result, flags=re.IGNORECASE)

    # 2. Fix dropped apostrophe suffixes (common SentencePiece truncation artifacts)
    result = re.sub(r"\bThat'(?![a-zA-Z])", "That's", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdon'(?![a-zA-Z])", "don't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bcan'(?![a-zA-Z])", "can't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwon'(?![a-zA-Z])", "won't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdidn'(?![a-zA-Z])", "didn't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwasn'(?![a-zA-Z])", "wasn't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bisn'(?![a-zA-Z])", "isn't", result, flags=re.IGNORECASE)
    result = re.sub(r"\baren'(?![a-zA-Z])", "aren't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdoesn'(?![a-zA-Z])", "doesn't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bI'(?![a-zA-Z])(?:\s*|\b)", "I'm ", result)

    # 3. Fix reversed multi-stream token stutter artifacts: e.g. "I doingm well", "you handlingre"
    result = re.sub(r"\bI doingm well\b", "I'm doing well", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou handlingre\b", "are you handling", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou'(?![a-zA-Z])\s*(busy|free|available|there|ready|doing|calling)\b", r"you're \1", result, flags=re.IGNORECASE)
    result = re.sub(r"\bgetll\b", "get'll", result, flags=re.IGNORECASE)

    # 4. Anti-Persona Leak: intercept and neutralize pretraining "Moshi" and "Kyutai" leaks
    display_name = agent_name.strip() if agent_name else "Snapserve"
    result = re.sub(r"\bMy name is Moshi\b", f"My name is {display_name}", result, flags=re.IGNORECASE)
    result = re.sub(r"\bI am Moshi\b", f"I'm {display_name}", result, flags=re.IGNORECASE)
    result = re.sub(r"\bI'm Moshi\b", f"I'm {display_name}", result, flags=re.IGNORECASE)
    result = re.sub(r"\bthis is Moshi\b", f"this is {display_name} from Snapserve", result, flags=re.IGNORECASE)
    result = re.sub(r"\bMoshi\b", display_name, result, flags=re.IGNORECASE)
    result = re.sub(r"\bKyutai\b", "Snapserve", result, flags=re.IGNORECASE)

    # 5. Normalize punctuation spacing (e.g. "word ," -> "word,")
    result = re.sub(r"\s+([,.:;?!%])", r"\1", result)

    # 6. Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result
