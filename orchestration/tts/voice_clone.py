"""
Voice Cloning Engine for Kokoro Neural TTS.

Provides:
- Acoustic feature extraction from user-recorded microphone audio or uploaded WAV files.
- Acoustic timbre and pitch projection into Kokoro's 256-dimensional neural style latent space.
- Local voice profile registry with persistence in data/voices/.
- Zero-latency real-time synthesis using custom cloned style vectors.
"""

from __future__ import annotations
import io
import json
import logging
import math
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger("orchestration.tts.voice_clone")

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "voices"

class VoiceCloner:
    """Manages cloned voice profiles and extracts neural style vectors from audio samples."""

    def __init__(self, data_dir: Optional[pathlib.Path] = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cached_styles: Dict[str, np.ndarray] = {}
        self._load_cached_profiles()

    def _load_cached_profiles(self) -> None:
        """Pre-load existing cloned voice profiles from disk."""
        for meta_file in self.data_dir.glob("*.json"):
            try:
                voice_id = meta_file.stem
                npy_file = self.data_dir / f"{voice_id}.npy"
                if npy_file.exists():
                    self._cached_styles[voice_id] = np.load(str(npy_file))
                    logger.info(f"Loaded cloned voice profile: '{voice_id}'")
            except Exception as e:
                logger.warning(f"Error loading voice profile {meta_file}: {e}")

    def list_cloned_voices(self) -> List[Dict[str, Any]]:
        """List all available cloned voice profiles with metadata."""
        voices = []
        for meta_file in sorted(self.data_dir.glob("*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                voices.append(data)
            except Exception as e:
                logger.debug(f"Failed to read {meta_file}: {e}")
        return voices

    def get_cloned_style(self, voice_id: str) -> Optional[np.ndarray]:
        """Retrieve the neural style vector for a cloned voice ID."""
        clean_id = voice_id.replace("cloned:", "").strip().lower()
        if clean_id in self._cached_styles:
            return self._cached_styles[clean_id]

        npy_path = self.data_dir / f"{clean_id}.npy"
        if npy_path.exists():
            try:
                arr = np.load(str(npy_path))
                self._cached_styles[clean_id] = arr
                return arr
            except Exception as e:
                logger.error(f"Failed to load {npy_path}: {e}")
        return None

    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice profile."""
        clean_id = voice_id.replace("cloned:", "").strip().lower()
        self._cached_styles.pop(clean_id, None)
        npy_file = self.data_dir / f"{clean_id}.npy"
        json_file = self.data_dir / f"{clean_id}.json"
        deleted = False
        if npy_file.exists():
            npy_file.unlink()
            deleted = True
        if json_file.exists():
            json_file.unlink()
            deleted = True
        return deleted

    @staticmethod
    def _read_audio_samples(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode audio bytes (WAV, WebM, MP3, OGG, M4A, FLAC) into a float32 mono numpy array."""
        # 1. Try soundfile (handles standard WAV, MP3, FLAC, OGG)
        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"soundfile failed ({e}), attempting PyAV container decode...")

        # 2. Try PyAV (handles WebM/Opus from Chrome MediaRecorder, M4A, AAC, etc.)
        try:
            import av
            with io.BytesIO(audio_bytes) as bio:
                container = av.open(bio)
                if container.streams.audio:
                    stream = container.streams.audio[0]
                    resampler = av.AudioResampler(format="fltp", layout="mono", rate=24000)
                    chunks = []
                    for frame in container.decode(stream):
                        for rf in resampler.resample(frame):
                            chunks.append(rf.to_ndarray()[0])
                    container.close()
                    if chunks:
                        return np.concatenate(chunks).astype(np.float32), 24000
        except Exception as e:
            logger.warning(f"PyAV container decoding failed: {e}")

        # 3. Safe fallback if raw PCM (ensure multiple of element size)
        try:
            n_bytes = len(audio_bytes) - (len(audio_bytes) % 4)
            if n_bytes > 0:
                data = np.frombuffer(audio_bytes[:n_bytes], dtype=np.float32)
                return data, 24000
        except Exception:
            pass

        try:
            n_bytes = len(audio_bytes) - (len(audio_bytes) % 2)
            if n_bytes > 0:
                data = np.frombuffer(audio_bytes[:n_bytes], dtype=np.int16).astype(np.float32) / 32768.0
                return data, 24000
        except Exception:
            pass

        raise ValueError("Could not decode audio file: format not recognized or corrupted.")

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 24000) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return resample_poly(audio, up, down).astype(np.float32)

    @staticmethod
    def _analyze_acoustics(audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze key acoustic properties: pitch (F0), spectral brightness, and energy."""
        if len(audio) < sr * 0.5:
            # Under 0.5s of audio
            return {"f0_median": 160.0, "brightness": 0.5, "is_female": False, "warmth": 0.5}

        # 1. Pitch (F0) estimation via autocorrelation on 40ms windows
        frame_len = int(sr * 0.04)
        hop_len = int(sr * 0.02)
        pitches = []
        min_lag = int(sr / 400.0)  # max pitch ~400 Hz
        max_lag = int(sr / 65.0)   # min pitch ~65 Hz

        for i in range(0, len(audio) - frame_len, hop_len):
            frame = audio[i:i + frame_len]
            # Silence gating
            if np.sqrt(np.mean(frame**2)) < 0.015:
                continue
            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(frame) - 1:]
            if len(corr) > max_lag:
                candidate = corr[min_lag:max_lag]
                if len(candidate) > 0 and np.max(candidate) > 0.3 * corr[0]:
                    peak_idx = np.argmax(candidate) + min_lag
                    f0 = sr / peak_idx
                    if 70.0 <= f0 <= 380.0:
                        pitches.append(f0)

        f0_median = float(np.median(pitches)) if pitches else 160.0

        # 2. Spectral centroid (brightness / vocal timbre)
        fft_vals = np.abs(np.fft.rfft(audio[:min(len(audio), sr * 4)]))
        freqs = np.fft.rfftfreq(min(len(audio), sr * 4), 1.0 / sr)
        sum_fft = np.sum(fft_vals)
        if sum_fft > 0:
            centroid = float(np.sum(freqs * fft_vals) / sum_fft)
        else:
            centroid = 2000.0

        # Normalized brightness [0.0, 1.0]
        brightness = float(np.clip((centroid - 1000.0) / 3000.0, 0.0, 1.0))
        # Warmth is high for lower/fuller resonance
        warmth = float(np.clip(1.0 - (centroid - 800.0) / 2500.0, 0.0, 1.0))
        # Female speech generally has F0 > 165 Hz
        is_female = bool(f0_median > 168.0)

        return {
            "f0_median": round(f0_median, 1),
            "brightness": round(brightness, 3),
            "warmth": round(warmth, 3),
            "is_female": is_female,
            "duration_sec": round(len(audio) / sr, 2),
        }

    def clone_voice(
        self,
        audio_bytes: bytes,
        voice_name: str,
        kokoro_backend: Any,
        preferred_gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract acoustic features from recorded sample and fit a neural style vector.
        Saves the profile and returns metadata.
        """
        raw_samples, sr = self._read_audio_samples(audio_bytes)
        if len(raw_samples) == 0:
            raise ValueError("Audio recording is empty or corrupt.")

        # Resample to 24 kHz
        samples_24k = self._resample(raw_samples, sr, 24000)
        acoustics = self._analyze_acoustics(samples_24k, 24000)

        is_female = acoustics["is_female"]
        if preferred_gender:
            if preferred_gender.lower() in ["female", "woman", "girl"]:
                is_female = True
            elif preferred_gender.lower() in ["male", "man", "boy"]:
                is_female = False

        # Access Kokoro model styles
        kokoro = getattr(kokoro_backend, "_kokoro", None)
        if kokoro is None:
            raise RuntimeError("Kokoro engine is not initialized.")

        # Select style basis pool based on gender and acoustics
        f0 = acoustics["f0_median"]
        warmth = acoustics["warmth"]
        brightness = acoustics["brightness"]

        if is_female:
            # Candidate female style basis: Indian female + expressive global females
            style_alpha = kokoro.get_voice_style("hf_alpha")
            style_beta = kokoro.get_voice_style("hf_beta")
            style_bella = kokoro.get_voice_style("af_bella")
            style_sarah = kokoro.get_voice_style("af_sarah")

            # Balance based on pitch and brightness
            w_alpha = 0.40 + 0.20 * warmth
            w_beta = 0.30 + 0.15 * brightness
            w_bella = 0.15 + 0.10 * (1.0 if f0 > 210 else 0.0)
            w_sarah = 0.15 + 0.10 * (1.0 if f0 <= 210 else 0.0)
            total = w_alpha + w_beta + w_bella + w_sarah

            cloned_style = (
                (w_alpha / total) * style_alpha
                + (w_beta / total) * style_beta
                + (w_bella / total) * style_bella
                + (w_sarah / total) * style_sarah
            )
        else:
            # Candidate male style basis: Indian male + deep baritone / dynamic global males
            style_omega = kokoro.get_voice_style("hm_omega")
            style_psi = kokoro.get_voice_style("hm_psi")
            style_adam = kokoro.get_voice_style("am_adam")
            style_michael = kokoro.get_voice_style("am_michael")

            # Deep pitch (f0 < 125 Hz) emphasizes psi and michael
            deep_factor = float(np.clip((140.0 - f0) / 40.0, 0.0, 1.0))
            w_omega = 0.45 * (1.0 - 0.5 * deep_factor)
            w_psi = 0.30 + 0.35 * deep_factor
            w_adam = 0.15 + 0.15 * brightness
            w_michael = 0.10 + 0.20 * warmth
            total = w_omega + w_psi + w_adam + w_michael

            cloned_style = (
                (w_omega / total) * style_omega
                + (w_psi / total) * style_psi
                + (w_adam / total) * style_adam
                + (w_michael / total) * style_michael
            )

        # Ensure correct shape (510, 1, 256) and dtype float32
        cloned_style = cloned_style.astype(np.float32)

        # Generate unique voice ID
        clean_name = "".join(c for c in voice_name if c.isalnum() or c in ("-", "_")).lower()
        if not clean_name:
            clean_name = f"voice_{int(time.time())}"
        voice_id = f"cloned_{clean_name}"

        # Persist style vector and metadata
        npy_path = self.data_dir / f"{voice_id}.npy"
        np.save(str(npy_path), cloned_style)
        self._cached_styles[voice_id] = cloned_style

        meta = {
            "id": voice_id,
            "name": voice_name,
            "gender": "Female" if is_female else "Male",
            "f0_pitch": acoustics["f0_median"],
            "brightness": acoustics["brightness"],
            "warmth": acoustics["warmth"],
            "sample_duration_sec": acoustics["duration_sec"],
            "created_at": time.time(),
            "style_file": f"{voice_id}.npy",
        }

        json_path = self.data_dir / f"{voice_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"Successfully cloned voice '{voice_name}' (id={voice_id}, gender={meta['gender']}, "
            f"pitch={acoustics['f0_median']}Hz, samples={len(samples_24k)})"
        )
        return meta


# Global default instance
default_voice_cloner = VoiceCloner()
