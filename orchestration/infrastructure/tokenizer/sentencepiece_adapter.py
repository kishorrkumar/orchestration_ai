"""
SentencePiece Tokenizer Adapter implementing domain Tokenizer protocol.
Uses models/tokenizer_spm_32k_3.model for verified Moshi / PersonaPlex 7B BPE token counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from orchestration.domain.protocols import Tokenizer

logger = logging.getLogger("orchestration.infrastructure.tokenizer")


class SentencePieceTokenizerAdapter(Tokenizer):
    """Concrete Tokenizer adapter utilizing the upstream 32k SentencePiece model."""

    def __init__(self, model_path: str = "models/tokenizer_spm_32k_3.model") -> None:
        self.model_path = Path(model_path)
        self._spm: Any = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        if self.model_path.exists():
            try:
                import sentencepiece as spm
                self._spm = spm.SentencePieceProcessor()
                self._spm.load(str(self.model_path))
                logger.info(f"Loaded SentencePiece model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load SentencePiece model: {e}")
                self._spm = None
        else:
            logger.info(f"SentencePiece model not found at {self.model_path}. Using BPE estimator.")
            self._spm = None

    def count_tokens(self, text: str) -> int:
        if self._spm is not None:
            try:
                pieces = self._spm.encode_as_pieces(text)
                return len(pieces)
            except Exception as e:
                logger.warning(f"SentencePiece encode error: {e}")

        # Fallback BPE token heuristic: ~4 characters per token for English prose
        cleaned = text.strip()
        if not cleaned:
            return 0
        words = len(cleaned.split())
        chars = len(cleaned)
        return max(1, int(chars / 3.8 + words * 0.1))
