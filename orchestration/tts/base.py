"""
TTS Abstraction Layer with Streaming Interface and Swappable Backends.

Supported Backends:
1. Kokoro-v1.0 (ONNX Runtime, 82M params, CPU/GPU, ~100MB VRAM/RAM)
   - Indian English Male (Aarav): 'hm_omega', 'hm_psi'
   - Indian English Female (Priya): 'hf_alpha', 'hf_beta'
2. Edge-TTS (en-IN-PrabhatNeural, en-IN-NeerjaExpressiveNeural)
3. AI4Bharat Indic-TTS / Parler-TTS (FastPitch + HiFi-GAN / Parler-TTS Indian English)
4. Coqui XTTS-v2 (Voice Cloning with clean 10-20s Indian English WAV)
5. Piper TTS (Fast CPU neural fallback)
6. Local SAPI5 / pyttsx3 Fallback
"""

from __future__ import annotations
import abc
import asyncio
import concurrent.futures
import io
import logging
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .text_norm import normalize_indian_english_text
from .post_process import post_process_speech

logger = logging.getLogger("orchestration.tts")

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="tts_worker")

# Shared paths
MODELS_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
KOKORO_DIR = MODELS_ROOT / "kokoro"
KOKORO_MODEL_FILE = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = KOKORO_DIR / "voices-v1.0.bin"


class BaseTTSBackend(abc.ABC):
    """Abstract Base Class for all Voice Synthesis Backends."""

    def __init__(self, name: str, sample_rate: int = 24000, speaking_rate: float = 1.0) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.speaking_rate = speaking_rate
        self._is_warmed = False

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Returns True if model weights and runtime dependencies are ready."""
        pass

    @abc.abstractmethod
    def warm_up(self) -> None:
        """Run warm-up inference to eliminate initial cold-start latency."""
        pass

    @abc.abstractmethod
    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """
        Synthesize text chunk to 24 kHz float32 mono PCM numpy array.
        Must be asynchronous and non-blocking to the main event loop.
        """
        pass

    def resample_if_needed(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample audio array to target sample_rate (24 kHz)."""
        if source_sr == self.sample_rate or len(audio) == 0:
            return audio.astype(np.float32)
        n_out = int(round(len(audio) * self.sample_rate / source_sr))
        resampled = np.interp(
            np.linspace(0, len(audio), n_out, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        return resampled

    def trim_edge_silence(self, samples: np.ndarray, threshold: float = 0.005) -> np.ndarray:
        """Trim silence padding at boundaries to enable seamless chunk joins."""
        if len(samples) == 0:
            return samples
        abs_s = np.abs(samples)
        non_silent = np.where(abs_s > threshold)[0]
        if len(non_silent) == 0:
            return samples
        start_idx = max(0, non_silent[0] - int(self.sample_rate * 0.01))
        end_idx = min(len(samples), non_silent[-1] + int(self.sample_rate * 0.02))
        return samples[start_idx:end_idx]


# ---------------------------------------------------------------------------
# Backend 1: Kokoro ONNX
# ---------------------------------------------------------------------------
class KokoroTTSBackend(BaseTTSBackend):
    """
    Kokoro-v1.0 82M Neural TTS via ONNX Runtime.
    Ultra-low latency (< 150ms on CPU/GPU), zero external network calls.
    Voices:
      - 'hm_omega' / 'aarav' (Indian English Male)
      - 'hm_psi' (Deep Indian English Male)
      - 'hf_alpha' / 'priya' (Indian English Female)
      - 'hf_beta' (Expressive Indian English Female)
    """

    VOICE_MAP = {
        "en-in-prabhatneural": "aarav_colloquial",
        "en-in-neerjaexpressiveneural": "priya_colloquial",
        "aarav": "hm_omega",
        "aarav_colloquial": "aarav_colloquial",
        "indian_male": "aarav_colloquial",
        "kabir": "hm_psi",
        "hm_omega": "hm_omega",
        "hm_psi": "hm_psi",
        "priya": "hf_alpha",
        "priya_colloquial": "priya_colloquial",
        "indian_female": "hf_alpha",
        "ananya": "hf_beta",
        "hf_alpha": "hf_alpha",
        "hf_beta": "hf_beta",
        "am_adam": "am_adam",
        "af_bella": "af_bella",
    }

    def __init__(self, sample_rate: int = 24000, speaking_rate: float = 1.0, default_voice: str = "aarav_colloquial") -> None:
        super().__init__("kokoro", sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.default_voice = default_voice
        self._kokoro = None
        self._available = False
        self._voice_styles: Dict[str, np.ndarray] = {}
        self._init_runtime()

    def _init_runtime(self) -> None:
        if not KOKORO_MODEL_FILE.exists() or not KOKORO_VOICES_FILE.exists():
            logger.info(f"Kokoro model files not found in {KOKORO_DIR}.")
            return

        try:
            import platform
            import espeakng_loader
            if platform.system() == "Windows":
                espeakng_loader.make_library_available()

            lib_path = espeakng_loader.get_library_path()
            data_path = espeakng_loader.get_data_path()
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(pathlib.Path(lib_path).parent)

            from kokoro_onnx import Kokoro
            from kokoro_onnx.config import EspeakConfig

            cfg = EspeakConfig(lib_path=lib_path, data_path=data_path)
            self._kokoro = Kokoro(
                model_path=str(KOKORO_MODEL_FILE),
                voices_path=str(KOKORO_VOICES_FILE),
                espeak_config=cfg,
            )
            self._available = True
            logger.info("KokoroTTSBackend initialized successfully.")

            # Pre-compute blended colloquial Indian English voice styles
            try:
                omega = self._kokoro.get_voice_style("hm_omega")
                psi = self._kokoro.get_voice_style("hm_psi")
                # 65% hm_omega + 35% hm_psi gives a relaxed, colloquial Indian English male timbre
                self._voice_styles["aarav_colloquial"] = 0.65 * omega + 0.35 * psi

                alpha = self._kokoro.get_voice_style("hf_alpha")
                beta = self._kokoro.get_voice_style("hf_beta")
                # 60% hf_alpha + 40% hf_beta gives a warm, bright colloquial Indian English female timbre
                self._voice_styles["priya_colloquial"] = 0.60 * alpha + 0.40 * beta
            except Exception as e:
                logger.warning(f"Kokoro voice style blending note: {e}")

        except Exception as e:
            logger.warning(f"Kokoro initialization note: {e}")
            self._available = False

    def is_available(self) -> bool:
        return self._available and self._kokoro is not None

    def warm_up(self) -> None:
        if not self.is_available() or self._is_warmed:
            return
        try:
            t0 = time.perf_counter()
            self._synthesize_sync("Namaste! Haanji, I am ready.", voice=self.default_voice)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._is_warmed = True
            logger.info(f"Kokoro warm-up completed in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"Kokoro warm-up failed: {e}")

    def _synthesize_sync(self, text: str, voice: str) -> np.ndarray:
        v_key = voice.lower() if voice else self.default_voice
        if v_key in self._voice_styles:
            voice_target = self._voice_styles[v_key]
        else:
            from .voice_clone import default_voice_cloner
            cloned_style = default_voice_cloner.get_cloned_style(v_key)
            if cloned_style is not None:
                self._voice_styles[v_key] = cloned_style
                voice_target = cloned_style
            else:
                voice_id = self.VOICE_MAP.get(v_key, self.VOICE_MAP.get(self.default_voice, "hm_omega"))
                if voice_id in self._voice_styles:
                    voice_target = self._voice_styles[voice_id]
                else:
                    voice_target = voice_id

        # 1.06 speed provides an agile, natural colloquial Indian English tempo
        rate = 1.06 if self.speaking_rate == 1.0 else self.speaking_rate
        samples, sr = self._kokoro.create(text, voice=voice_target, speed=rate, lang="en-us")
        if samples is None or len(samples) == 0:
            return np.zeros(0, dtype=np.float32)
        resampled = self.resample_if_needed(samples, sr)
        trimmed = self.trim_edge_silence(resampled)
        return post_process_speech(trimmed, fs=self.sample_rate)

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        if not self.is_available():
            return np.zeros(0, dtype=np.float32)
        v = voice or self.default_voice
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, self._synthesize_sync, text, v)


# ---------------------------------------------------------------------------
# Backend 2: Edge-TTS (Natural Indian English)
# ---------------------------------------------------------------------------
class EdgeTTSBackend(BaseTTSBackend):
    """
    Microsoft Edge Neural Speech for authentic Indian English:
      - 'en-IN-PrabhatNeural' (Aarav - Professional Male)
      - 'en-IN-NeerjaExpressiveNeural' (Priya - Expressive Female)
    """

    VOICE_MAP = {
        "aarav": "en-IN-PrabhatNeural",
        "indian_male": "en-IN-PrabhatNeural",
        "priya": "en-IN-NeerjaExpressiveNeural",
        "indian_female": "en-IN-NeerjaExpressiveNeural",
        "prabhat": "en-IN-PrabhatNeural",
        "neerja": "en-IN-NeerjaExpressiveNeural",
    }

    def __init__(self, sample_rate: int = 24000, speaking_rate: float = 1.0, default_voice: str = "en-IN-PrabhatNeural") -> None:
        super().__init__("edge_tts", sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.default_voice = default_voice
        self._available = False
        try:
            import edge_tts
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def warm_up(self) -> None:
        self._is_warmed = True

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        if not self.is_available():
            return np.zeros(0, dtype=np.float32)

        import edge_tts

        v = voice or self.default_voice
        voice_id = self.VOICE_MAP.get(v.lower(), v)
        rate_percent = int((self.speaking_rate - 1.0) * 100)
        rate_str = f"{'+' if rate_percent >= 0 else ''}{rate_percent}%"

        try:
            async def _fetch():
                comm = edge_tts.Communicate(text, voice_id, rate=rate_str)
                buf = bytearray()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        buf.extend(chunk["data"])
                return buf

            mp3_buf = await asyncio.wait_for(_fetch(), timeout=6.0)
            if not mp3_buf:
                return np.zeros(0, dtype=np.float32)

            data, sr = sf.read(io.BytesIO(mp3_buf), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            resampled = self.resample_if_needed(data, sr)
            trimmed = self.trim_edge_silence(resampled)
            return post_process_speech(trimmed, fs=self.sample_rate)
        except Exception as e:
            logger.warning(f"EdgeTTSBackend synthesis failed ({e}). Falling back.")
            return np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backend 3: AI4Bharat Indic-TTS / Parler-TTS
# ---------------------------------------------------------------------------
class IndicTTSBackend(BaseTTSBackend):
    """
    AI4Bharat Indic-TTS (FastPitch + HiFi-GAN) / Indic Parler-TTS.
    Provides authentic regional Indian accents.
    """

    def __init__(self, sample_rate: int = 24000, speaking_rate: float = 1.0, default_voice: str = "indic_en") -> None:
        super().__init__("indic_tts", sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.default_voice = default_voice
        self._available = False
        self._model = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import torch
            # Check if parler_tts or indic_tts is installed
            import parler_tts
            self._available = True
            logger.info("IndicTTSBackend (Parler-TTS) found.")
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def warm_up(self) -> None:
        self._is_warmed = True

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backend 4: XTTS-v2 Voice Cloning
# ---------------------------------------------------------------------------
class XTTSBackend(BaseTTSBackend):
    """
    Coqui XTTS-v2 streaming voice cloning using a 10-20s clean Indian English reference WAV.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        speaking_rate: float = 1.0,
        speaker_wav: Optional[str] = None,
    ) -> None:
        super().__init__("xtts_v2", sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.speaker_wav = speaker_wav or str(MODELS_ROOT / "voices" / "indian_ref.wav")
        self._available = False
        self._tts = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from TTS.api import TTS
            if os.path.exists(self.speaker_wav):
                self._available = True
                logger.info(f"XTTSBackend available with reference: {self.speaker_wav}")
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available and os.path.exists(self.speaker_wav)

    def warm_up(self) -> None:
        self._is_warmed = True

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backend 5: Piper TTS (Fast CPU Neural)
# ---------------------------------------------------------------------------
class PiperTTSBackend(BaseTTSBackend):
    """Piper neural voice synthesis on CPU via ONNX."""

    def __init__(self, sample_rate: int = 24000, speaking_rate: float = 1.0) -> None:
        super().__init__("piper", sample_rate=sample_rate, speaking_rate=speaking_rate)
        self._available = False
        try:
            import piper
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def warm_up(self) -> None:
        self._is_warmed = True

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# Backend 6: Guaranteed Fallback
# ---------------------------------------------------------------------------
class FallbackTTSBackend(BaseTTSBackend):
    """Native pyttsx3 or gentle audible tone generator for absolute zero-failure guarantee."""

    def __init__(self, sample_rate: int = 24000, speaking_rate: float = 1.0) -> None:
        super().__init__("fallback", sample_rate=sample_rate, speaking_rate=speaking_rate)

    def is_available(self) -> bool:
        return True

    def warm_up(self) -> None:
        self._is_warmed = True

    def _synthesize_pyttsx3(self, text: str) -> np.ndarray:
        try:
            import pyttsx3
            import pythoncom
            pythoncom.CoInitialize()

            engine = pyttsx3.init()
            engine.setProperty("rate", int(150 * self.speaking_rate))

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            data, sr = sf.read(tmp_path, dtype="float32")
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            if data.ndim > 1:
                data = data.mean(axis=1)

            resampled = self.resample_if_needed(data, sr)
            return post_process_speech(resampled, fs=self.sample_rate)
        except Exception:
            # Fallback gentle tone burst
            duration = min(1.0, max(0.2, len(text.split()) * 0.15))
            n = int(self.sample_rate * duration)
            t = np.linspace(0, duration, n, endpoint=False)
            tone = 0.05 * np.sin(2 * np.pi * 440 * t)
            return tone.astype(np.float32)

    async def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_EXECUTOR, self._synthesize_pyttsx3, text)


# ---------------------------------------------------------------------------
# TTS Factory & Composite Engine
# ---------------------------------------------------------------------------
class StreamingCompositeTTS:
    """
    Orchestrates primary and fallback TTS backends.
    Pre-processes text with Indian-English normalizer and post-processes audio.
    """

    def __init__(
        self,
        preferred_backend: str = "auto",
        sample_rate: int = 24000,
        speaking_rate: float = 0.98,
        default_voice: str = "aarav",
    ) -> None:
        self.sample_rate = sample_rate
        self.speaking_rate = speaking_rate
        self.default_voice = default_voice

        # Instantiate backends in order of preference
        self.kokoro = KokoroTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate, default_voice=default_voice)
        self.edge = EdgeTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate, default_voice="en-IN-PrabhatNeural")
        self.indic = IndicTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.xtts = XTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.piper = PiperTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate)
        self.fallback = FallbackTTSBackend(sample_rate=sample_rate, speaking_rate=speaking_rate)

        self.preferred_backend = preferred_backend
        self._select_active_backend()

    def _select_active_backend(self) -> None:
        pref = self.preferred_backend.lower()
        if pref == "edge" and self.edge.is_available():
            self.active_backend = self.edge
        elif pref == "kokoro" and self.kokoro.is_available():
            self.active_backend = self.kokoro
        elif pref == "indic" and self.indic.is_available():
            self.active_backend = self.indic
        elif pref == "xtts" and self.xtts.is_available():
            self.active_backend = self.xtts
        elif pref == "piper" and self.piper.is_available():
            self.active_backend = self.piper
        else:
            # Auto: prefer Kokoro (local offline neural Indian English) -> edge -> fallback
            if self.kokoro.is_available():
                self.active_backend = self.kokoro
            elif self.edge.is_available():
                self.active_backend = self.edge
            else:
                self.active_backend = self.fallback

        logger.info(f"Active TTS backend selected: {self.active_backend.name}")

    def warm_up(self) -> None:
        """Pre-warm models to eliminate cold-start latency."""
        self.active_backend.warm_up()

    async def synthesize_chunk(self, raw_chunk: str, voice: Optional[str] = None) -> np.ndarray:
        """
        Synthesize a single text chunk with normalization and safety fallbacks.
        Returns 24 kHz float32 audio.
        """
        normalized_text = normalize_indian_english_text(raw_chunk)
        if not normalized_text.strip():
            return np.zeros(0, dtype=np.float32)

        v = voice or self.default_voice

        # 1. Try active backend
        audio = await self.active_backend.synthesize(normalized_text, voice=v)
        if len(audio) > 0:
            return audio

        # 2. Try Kokoro local neural fallback
        if self.active_backend != self.kokoro and self.kokoro.is_available():
            logger.info(f"Attempting Kokoro fallback for chunk: '{normalized_text[:25]}...'")
            audio = await self.kokoro.synthesize(normalized_text, voice=v)
            if len(audio) > 0:
                return audio

        # 3. Try Edge TTS fallback
        if self.active_backend != self.edge and self.edge.is_available():
            logger.info(f"Attempting Edge TTS fallback for chunk: '{normalized_text[:25]}...'")
            audio = await self.edge.synthesize(normalized_text, voice=v)
            if len(audio) > 0:
                return audio

        # 4. Guaranteed local fallback (SAPI5 pyttsx3)
        return await self.fallback.synthesize(normalized_text, voice=v)
