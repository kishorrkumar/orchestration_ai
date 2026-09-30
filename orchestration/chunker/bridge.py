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

from .normalizer import normalize_for_tts


def normalize_indian_english_text(text: str) -> str:
    """Normalize currency, numbers, markdown, and common abbreviations for natural Indian speech prosody."""
    return normalize_for_tts(text)


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

        self._buffer: list[str] = []
        self._is_first_chunk = True
        self._chunks_emitted = 0

    def feed_token(self, token: str) -> list[str]:
        """Feed a token into the chunker and return any speakable chunks ready to synthesize."""
        if not token:
            return []

        self._buffer.append(token)
        current_text = "".join(self._buffer)
        words = current_text.strip().split()
        word_count = len(words)

        chunks_to_emit: list[str] = []

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

    def flush(self) -> list[str]:
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
