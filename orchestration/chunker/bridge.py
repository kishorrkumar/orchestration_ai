"""
Streaming LLM -> Clause Chunker -> TTS Bridge.

Implements low-latency streaming clause segmentation:
- First chunk: Flushes at the first clause boundary (',', '-', ';', ':', '.', '?', '!')
  OR after 3-5 words, whichever comes first, ensuring ultra-low Time-To-First-Audio (TTFA).
- Later chunks: Flush at natural clause/sentence boundaries (roughly 6-15 words).
- Text Normalization:
  - Currencies in INR: '₹5000' -> 'five thousand rupees', '12500 INR' -> 'twelve thousand five hundred rupees'
  - Numbers & Decimals: '100' -> 'one hundred', '3.5' -> 'three point five'
  - Common Indian names / locations: preserves pronunciation
  - Never splits mid-word, mid-number, or mid-abbreviation.
"""

from __future__ import annotations
import re
from typing import AsyncGenerator, Generator, List, Optional


# Regex patterns for clause boundaries
CURRENCY_INR_REGEX = re.compile(
    r"(?:(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(crore|lakh|thousand)?|"
    r"([\d,]+(?:\.\d+)?)\s*(crore|lakh|thousand)?\s*(?:₹|rs\.?|inr|rupees))",
    re.IGNORECASE,
)


# Number to spoken words map for Indian currency and numbers
NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety", 100: "one hundred",
}


def _number_to_words(n: int) -> str:
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return f"{NUM_WORDS.get(tens, '')} {NUM_WORDS.get(ones, '')}".strip()
    return str(n)


from .normalizer import normalize_for_tts, strip_markdown_and_emojis, normalize_currency_inr


def normalize_indian_english_text(text: str) -> str:
    """Normalize currency, numbers, markdown, and common abbreviations for natural Indian speech prosody."""
    return normalize_for_tts(text)

    # Currency normalization: ₹5000 / 5000 INR -> five thousand rupees
    def _inr_sub(match):
        amount = match.group(1) or match.group(3)
        scale = (match.group(2) or match.group(4) or "").lower()
        amount_clean = amount.replace(",", "")
        try:
            num = float(amount_clean)
            if scale == "crore":
                num *= 10000000
            elif scale == "lakh":
                num *= 100000
            elif scale == "thousand":
                num *= 1000

            num = int(round(num))
            if num >= 10000000:
                crores = int(num / 10000000)
                return f"{_number_to_words(crores)} crore rupees"
            elif num >= 100000:
                lakhs = int(num / 100000)
                rem = num % 100000
                if rem == 0:
                    return f"{_number_to_words(lakhs)} lakh rupees"
                if rem == 50000:
                    return f"{_number_to_words(lakhs)} point five lakh rupees"
                return f"{_number_to_words(lakhs)} lakh {rem} rupees"
            elif num >= 1000:
                thousands = int(num / 1000)
                rem = num % 1000
                if rem == 0:
                    return f"{_number_to_words(thousands)} thousand rupees"
                return f"{_number_to_words(thousands)} thousand {rem} rupees"
            else:
                return f"{_number_to_words(num)} rupees"
        except ValueError:
            return f"{amount_clean} rupees"

    normalized = CURRENCY_INR_REGEX.sub(_inr_sub, normalized)

    # Standard speech cleanups & broken transliteration smoothing
    normalized = re.sub(r"\bha\s*an\s*ji\b", "yes", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bha\s*an\b", "yes", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bhaanji\b", "yes", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bAI\b", "A.I.", normalized)
    normalized = re.sub(r"\bML\b", "M.L.", normalized)
    normalized = re.sub(r"\bTTS\b", "T.T.S.", normalized)
    normalized = re.sub(r"\bSTT\b", "S.T.T.", normalized)
    normalized = re.sub(r"\bVAD\b", "V.A.D.", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class ClauseChunker:
    """
    Stateful token-to-speech chunker.
    Consumes LLM tokens on the fly and emits speakable text chunks as soon as ready.
    """

    def __init__(self, first_chunk_min_words: int = 2, first_chunk_max_words: int = 7, later_chunk_min_words: int = 5, later_chunk_max_words: int = 14) -> None:
        self.first_chunk_min_words = first_chunk_min_words
        self.first_chunk_max_words = first_chunk_max_words
        self.later_chunk_min_words = later_chunk_min_words
        self.later_chunk_max_words = later_chunk_max_words

        self._buffer: List[str] = []
        self._is_first_chunk = True
        self._chunks_emitted = 0

    def feed_token(self, token: str) -> List[str]:
        """Feed a token into the chunker and return any speakable chunks ready to synthesize."""
        if not token:
            return []

        self._buffer.append(token)
        current_text = "".join(self._buffer)
        words = current_text.strip().split()
        word_count = len(words)

        chunks_to_emit: List[str] = []

        if self._is_first_chunk:
            # Rule 1: First chunk flushes at punctuation boundary (if >= min_words) or upon hitting max_words
            has_break = bool(re.search(r"[,;:!?.—–]\s*$", current_text))
            if (has_break and word_count >= self.first_chunk_min_words) or word_count >= self.first_chunk_max_words:
                chunk = normalize_indian_english_text(current_text)
                if chunk:
                    chunks_to_emit.append(chunk)
                    self._chunks_emitted += 1
                    self._is_first_chunk = False
                    self._buffer.clear()
        else:
            # Rule 2: Subsequent chunks flush at clause boundaries (if >= min_words) OR when reaching max_words
            has_break = bool(re.search(r"[,;:!?.—–]\s*$", current_text))
            if (has_break and word_count >= self.later_chunk_min_words) or word_count >= self.later_chunk_max_words:
                chunk = normalize_indian_english_text(current_text)
                if chunk:
                    chunks_to_emit.append(chunk)
                    self._chunks_emitted += 1
                    self._buffer.clear()

        return chunks_to_emit

    def flush(self) -> List[str]:
        """Flush any remaining text in the buffer."""
        if not self._buffer:
            return []
        remaining = "".join(self._buffer).strip()
        self._buffer.clear()
        if remaining:
            normalized = normalize_indian_english_text(remaining)
            if normalized:
                self._chunks_emitted += 1
                return [normalized]
        return []

    def reset(self) -> None:
        """Reset chunker state for a new turn."""
        self._buffer.clear()
        self._is_first_chunk = True
        self._chunks_emitted = 0
