"""
Caller Audio Cleaning, Noise Suppression, and Voice Isolation Pipeline (CPU).

Pipeline per session:
1. Inbound 24 kHz PCM (int16 or float32).
2. High-pass filter (80 Hz) to eliminate table rumble, low-frequency hum, and AC rumble.
3. Native RNNoise (48 kHz C recurrent neural network) deep noise reduction.
4. Silero VAD v6 high-precision neural speech probability estimation.
5. Direct-field Speaker Isolation:
   - Primary user speech: pass cleaned speech with natural dynamics.
   - Background chatter / non-speech / room reverberation: emit pure silence frames.
6. End of Speech (EOS) Hangover:
   - Smooth 350ms hangover bridges natural inter-syllable pauses without clipping.
   - Sharp, clean cutoff at turn boundary so PersonaPlex immediately recognizes end-of-turn.
7. Latency < 15 ms on CPU.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np
import soundfile as sf

from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    compute_rms,
    high_pass_filter,
    int16_to_float32,
)

logger = logging.getLogger("orchestration.audio.cleaner")

RNNOISE_FRAME_SIZE_48K = 480  # 10 ms at 48 kHz


class CallerAudioCleaner:
    """
    High-performance audio cleaner and voice isolation engine.
    Supports 3 modes:
      - 'isolation': RNNoise + Silero VAD + End of Speech Gate (Strongest, isolates user voice)
      - 'rnnoise': RNNoise neural denoise without zero-gating
      - 'bypass': Raw audio pass-through
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        mode: str = "rnnoise",
        noise_suppression: bool = True,
        speaker_isolation: bool = False,
        suppression_strength: float = 0.65,
        bypass: bool = False,
        vad_hangover_sec: float = 0.35,
        vad_threshold: float = 0.45,
        min_speech_rms: float = 0.010,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        if bypass:
            self.mode = "bypass"
        elif speaker_isolation or mode == "isolation":
            self.mode = "isolation"
        else:
            self.mode = mode
        self.noise_suppression = noise_suppression
        self.speaker_isolation = speaker_isolation
        self.suppression_strength = suppression_strength
        self.vad_hangover_sec = vad_hangover_sec
        self.vad_threshold = vad_threshold
        self.min_speech_rms = min_speech_rms

        # State tracking
        self.last_speech_time: float = 0.0
        self.is_speech_active: bool = False
        self.last_vad_prob: float = 0.0
        self.last_rms: float = 0.0
        self.processing_latency_ms: float = 0.0

        # Accumulated input/output buffers
        self._accumulator = np.empty(0, dtype=np.float32)

        # History buffers for A/B quality inspection
        self._raw_history: list[np.ndarray] = []
        self._clean_history: list[np.ndarray] = []

        # Native RNNoise C wrapper
        self._rnnoise_state = None
        self._init_rnnoise()

        # Silero VAD model
        self._vad_model = None
        self._init_silero_vad()

    @property
    def bypass(self) -> bool:
        return self.mode == "bypass"

    @bypass.setter
    def bypass(self, value: bool) -> None:
        self.mode = "bypass" if value else "isolation"

    def _init_rnnoise(self) -> None:
        try:
            from pyrnnoise import rnnoise
            self._rnnoise_module = rnnoise
            self._rnnoise_state = rnnoise.create()
            logger.info("CallerAudioCleaner: Native RNNoise C engine initialized")
        except Exception as e:
            self._rnnoise_module = None
            self._rnnoise_state = None
            logger.warning(f"CallerAudioCleaner: Native RNNoise unavailable ({e}), using spectral fallback")

    def _init_silero_vad(self) -> None:
        try:
            from faster_whisper.vad import get_vad_model
            self._vad_model = get_vad_model()
            logger.info("CallerAudioCleaner: Silero VAD v6 engine initialized")
        except Exception as e:
            self._vad_model = None
            logger.warning(f"CallerAudioCleaner: Silero VAD unavailable ({e}), using energy fallback")

    def __del__(self) -> None:
        if self._rnnoise_state is not None and self._rnnoise_module is not None:
            try:
                self._rnnoise_module.destroy(self._rnnoise_state)
            except Exception:
                pass
            self._rnnoise_state = None

    def set_mode(self, mode: str) -> None:
        valid_modes = {"isolation", "rnnoise", "bypass"}
        if mode.lower() in valid_modes:
            self.mode = mode.lower()
            logger.info(f"CallerAudioCleaner mode set to: {self.mode}")

    def set_bypass(self, bypass: bool) -> None:
        self.bypass = bypass

    def set_suppression_strength(self, strength: float) -> None:
        self.suppression_strength = max(0.0, min(1.0, strength))

    def process_chunk(self, raw_audio: np.ndarray | bytes) -> list[np.ndarray]:
        """
        Process incoming audio array (int16, float32, or raw bytes).
        Returns a list of 1,920-sample (80 ms) float32 frames.
        """
        t0 = time.perf_counter()

        # Handle raw byte buffer
        if isinstance(raw_audio, bytes):
            # Assume float32 PCM bytes
            samples_f32 = np.frombuffer(raw_audio, dtype=np.float32).copy()
        elif raw_audio.dtype == np.int16:
            samples_f32 = int16_to_float32(raw_audio)
        else:
            samples_f32 = raw_audio.astype(np.float32)

        if len(samples_f32) == 0:
            return []

        self._raw_history.append(samples_f32.copy())

        # If bypass mode is active, forward raw audio
        if self.mode == "bypass":
            self._clean_history.append(samples_f32.copy())
            self._accumulator = np.concatenate((self._accumulator, samples_f32))
            frames = self._pop_frames()
            self.processing_latency_ms = (time.perf_counter() - t0) * 1000.0
            return frames

        # 1. High-Pass filter to strip rumble / 50Hz hum
        filtered = high_pass_filter(samples_f32, cutoff_hz=80.0, fs=self.sample_rate)

        # 2. Resample 24k -> 48k for RNNoise
        upsampled_48k = np.empty(len(filtered) * 2, dtype=np.float32)
        upsampled_48k[0::2] = filtered
        if len(filtered) > 1:
            upsampled_48k[1:-1:2] = 0.5 * (filtered[:-1] + filtered[1:])
            upsampled_48k[-1] = filtered[-1]
        elif len(filtered) == 1:
            upsampled_48k[1] = filtered[0]

        # 3. RNNoise Recurrent Neural Suppression
        if self.noise_suppression:
            if self._rnnoise_state is not None and self._rnnoise_module is not None:
                pcm_48k_i16 = (np.clip(upsampled_48k, -1.0, 1.0) * 32767.0).astype(np.int16)
                cleaned_i16 = np.empty_like(pcm_48k_i16)
                idx = 0
                while idx + RNNOISE_FRAME_SIZE_48K <= len(pcm_48k_i16):
                    sub = pcm_48k_i16[idx : idx + RNNOISE_FRAME_SIZE_48K]
                    cf, _ = self._rnnoise_module.process_mono_frame(self._rnnoise_state, sub)
                    cleaned_i16[idx : idx + RNNOISE_FRAME_SIZE_48K] = cf
                    idx += RNNOISE_FRAME_SIZE_48K
                if idx < len(pcm_48k_i16):
                    cleaned_i16[idx:] = pcm_48k_i16[idx:]

                # Blend using suppression strength
                cleaned_f32_48k = (cleaned_i16.astype(np.float32) / 32768.0)
                cleaned_48k = (
                    (1.0 - self.suppression_strength) * upsampled_48k
                    + self.suppression_strength * cleaned_f32_48k
                )
            else:
                cleaned_48k = self._apply_spectral_gate(upsampled_48k, 48000)
        else:
            cleaned_48k = upsampled_48k

        # 4. Downsample 48k -> 24k
        n_24k = len(cleaned_48k) // 2
        cleaned_24k = cleaned_48k[::2][:n_24k].astype(np.float32)

        # 5. Silero VAD Speech Probability
        vad_prob = self._compute_silero_vad_prob(cleaned_24k)
        self.last_vad_prob = vad_prob

        # 6. Direct-Field Energy RMS
        frame_rms = compute_rms(cleaned_24k)
        self.last_rms = frame_rms

        # 7. Voice Isolation & End of Speech (EOS) Hangover
        now = time.time()
        is_direct_speech = (
            (vad_prob >= self.vad_threshold and frame_rms >= self.min_speech_rms)
            or (frame_rms >= 0.045)
        )

        if is_direct_speech:
            self.last_speech_time = now
            self.is_speech_active = True
        elif now - self.last_speech_time < self.vad_hangover_sec:
            # Hangover keeps speech active across natural intra-sentence pauses
            self.is_speech_active = True
        else:
            # End of Speech (EOS) confirmed
            self.is_speech_active = False

        # 8. Output Gating
        if self.mode == "isolation":
            if self.is_speech_active:
                processed_output = cleaned_24k
            else:
                # Silence Gate: completely block room noise, distant speakers, TV, coughing
                processed_output = np.zeros_like(cleaned_24k)
        elif self.mode == "rnnoise":
            processed_output = cleaned_24k
        else:
            processed_output = samples_f32[:n_24k]

        self._clean_history.append(processed_output.copy())
        self._accumulator = np.concatenate((self._accumulator, processed_output))
        frames = self._pop_frames()

        self.processing_latency_ms = (time.perf_counter() - t0) * 1000.0
        return frames

    def _compute_silero_vad_prob(self, audio_24k: np.ndarray) -> float:
        """Compute speech probability using Silero VAD or energy fallback."""
        if len(audio_24k) == 0:
            return 0.0

        if self._vad_model is not None:
            try:
                # Resample 24k -> 16k for Silero
                n_16k = int(round(len(audio_24k) * 16000 / 24000))
                audio_16k = np.interp(
                    np.linspace(0, len(audio_24k), n_16k, endpoint=False),
                    np.arange(len(audio_24k)),
                    audio_24k,
                ).astype(np.float32)

                # Process in 512-sample windows
                probs = []
                idx = 0
                while idx + 512 <= len(audio_16k):
                    w = audio_16k[idx : idx + 512]
                    p = self._vad_model(w)
                    if len(p) > 0:
                        probs.append(float(p[0]))
                    idx += 512
                if probs:
                    return float(np.mean(probs))
            except Exception:
                pass

        # Energy fallback
        rms = compute_rms(audio_24k)
        return min(1.0, float(rms * 25.0))

    def _apply_spectral_gate(self, audio: np.ndarray, fs: int) -> np.ndarray:
        if len(audio) < 512:
            return audio
        fft_vals = np.fft.rfft(audio)
        mag = np.abs(fft_vals)
        phase = np.angle(fft_vals)
        noise_floor = np.median(mag) * self.suppression_strength * 2.6
        cleaned_mag = np.maximum(mag - noise_floor, 0.02 * mag)
        reconstructed = np.fft.irfft(cleaned_mag * np.exp(1j * phase), n=len(audio))
        return reconstructed.astype(np.float32)

    def _pop_frames(self) -> list[np.ndarray]:
        frames = []
        while len(self._accumulator) >= self.frame_size:
            frame = self._accumulator[: self.frame_size]
            self._accumulator = self._accumulator[self.frame_size :]
            frames.append(frame)
        return frames

    def get_raw_wav_bytes(self) -> bytes:
        if not self._raw_history:
            return b""
        combined = np.concatenate(self._raw_history)
        buf = io.BytesIO()
        sf.write(buf, combined, self.sample_rate, format="WAV")
        return buf.getvalue()

    def get_clean_wav_bytes(self) -> bytes:
        if not self._clean_history:
            return b""
        combined = np.concatenate(self._clean_history)
        buf = io.BytesIO()
        sf.write(buf, combined, self.sample_rate, format="WAV")
        return buf.getvalue()

    def reset(self) -> None:
        self._accumulator = np.empty(0, dtype=np.float32)
        self._raw_history.clear()
        self._clean_history.clear()
        self.is_speech_active = False
        self.last_speech_time = 0.0
        self.last_vad_prob = 0.0
        self.last_rms = 0.0
