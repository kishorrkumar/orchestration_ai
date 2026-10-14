"""
Smart Turn Completion & Dual-Channel Barge-In Detector.

Combines WebRTC AEC3, dual-channel reference tracking, Silero VAD,
and linguistic syntactic/prosodic analysis to solve:
1. Mid-sentence truncation on hesitations ("umm...", "and then...").
2. Sluggish turn yields on crisp short answers ("Yes.", "Sure.", "Done.").
3. Immediate (< 60 ms) barge-in detection when caller speaks over the agent.
"""

from __future__ import annotations

import collections
import re
import time
from typing import Any

import numpy as np

from .webrtc_apm import WebRTCAPM

# Connectives, conjunctions, and prepositions that signal continuation / hesitation
CONTINUATION_TOKENS = {
    "and", "but", "or", "so", "because", "if", "when", "while", "as",
    "that", "which", "where", "how", "what", "who", "whom", "whose",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "into",
    "about", "like", "through", "after", "before", "between", "under",
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "his", "her", "their", "our", "its", "um", "uh", "umm", "uhh", "er", "ah",
}

# Crisp affirmative / negative words that represent a complete conversational turn
COMPLETE_AFFIRMATIONS = {
    "yes", "yeah", "yep", "sure", "definitely", "absolutely", "okay", "ok",
    "fine", "done", "no", "nope", "nah", "correct", "right", "exactly",
    "perfect", "got it", "makes sense", "thank you", "thanks", "bye", "goodbye",
}


class SmartTurnDetector:
    """
    Dual-Channel Acoustic & Linguistic Turn Completion Detector.
    Integrates WebRTC APM echo cancellation with dynamic turn hysteresis.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_size: int = 1920,
        base_silence_sec: float = 0.50,
        short_turn_sec: float = 0.25,
        hesitation_extra_sec: float = 0.40,
        vad: Any | None = None,
        enable_aec: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.base_silence_sec = base_silence_sec
        self.short_turn_sec = short_turn_sec
        self.hesitation_extra_sec = hesitation_extra_sec
        self.vad = vad

        # Dual-Channel WebRTC AEC3 APM
        self.apm = WebRTCAPM(
            sample_rate=sample_rate,
            frame_size=frame_size,
            enabled=enable_aec,
        )

        # State tracking
        self.is_user_speaking: bool = False
        self.speech_frames: int = 0
        self.last_speech_time: float = 0.0
        self.last_transcript: str = ""
        self.active_silence_threshold: float = base_silence_sec
        self.barge_in_triggered: bool = False

        # Audio utterance buffer
        self._utterance_frames: list[np.ndarray] = []
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=4)

    def push_agent_reference(self, ref_audio: np.ndarray | bytes) -> None:
        """Feed agent outbound audio into AEC3 reference ring buffer."""
        self.apm.push_reference(ref_audio)

    def process_frame(
        self,
        mic_frame: np.ndarray,
        is_agent_speaking: bool = False,
        current_transcript: str = "",
        now: float | None = None,
    ) -> tuple[bool, bool, bool, np.ndarray | None]:
        """
        Process incoming microphone frame.

        Returns:
            (is_speaking, turn_completed, is_barge_in, completed_audio_if_any)
        """
        if now is None:
            now = time.time()

        # 1. Dual-Channel Acoustic Echo Cancellation
        cleaned_audio, aec_meta = self.apm.process_frame(mic_frame)

        # 2. VAD & Energy Analysis
        frame_rms = float(np.sqrt(np.mean(np.square(cleaned_audio))))
        if self.vad is not None:
            speech_prob = self.vad.get_speech_probability(cleaned_audio)
        else:
            speech_prob = min(1.0, frame_rms * 35.0)

        # 3. Dynamic Thresholding based on Agent Speech Status
        if is_agent_speaking:
            # When agent is playing, user must exceed AEC residual energy
            is_speech = (speech_prob >= 0.55 and frame_rms >= 0.015) or aec_meta.get("barge_in", False)
        else:
            is_speech = (speech_prob >= 0.38) or (frame_rms >= 0.012 and speech_prob >= 0.25)

        # 4. Immediate Double-Talk Barge-In Detection
        is_barge_in = False
        if is_agent_speaking and is_speech and not self.barge_in_triggered:
            if aec_meta.get("double_talk", False) or frame_rms > 0.025:
                is_barge_in = True
                self.barge_in_triggered = True

        if not is_agent_speaking:
            self.barge_in_triggered = False

        # 5. Turn State Machine
        if is_speech:
            self.speech_frames += 1
            self.last_speech_time = now
            if not self.is_user_speaking and self.speech_frames >= 2:
                self.is_user_speaking = True
                for pf in self._preroll:
                    self._utterance_frames.append(pf)
                self._preroll.clear()

            if self.is_user_speaking:
                self._utterance_frames.append(cleaned_audio)
            else:
                self._preroll.append(cleaned_audio)

            return (self.is_user_speaking, False, is_barge_in, None)

        else:
            # Silence Frame
            self._preroll.append(cleaned_audio)
            if not self.is_user_speaking:
                if now - self.last_speech_time > 0.35:
                    self.speech_frames = 0
                return (False, False, is_barge_in, None)

            # In user speech: determine dynamic silence timeout based on linguistic cues
            silence_dur = now - self.last_speech_time
            effective_threshold = self._compute_dynamic_silence_threshold(current_transcript)

            if silence_dur >= effective_threshold:
                # Conclude turn
                self.is_user_speaking = False
                self.speech_frames = 0
                self.barge_in_triggered = False
                if self._utterance_frames:
                    complete_audio = np.concatenate(self._utterance_frames)
                    self._utterance_frames.clear()
                    return (False, True, is_barge_in, complete_audio)

            return (self.is_user_speaking, False, is_barge_in, None)

    def _compute_dynamic_silence_threshold(self, text: str) -> float:
        """
        Dynamically adjusts silence threshold based on syntactic completion:
        - Affirmation or complete statement -> 250ms (ultra-fast reply)
        - Hesitation or continuation word -> 850ms (prevents cutting off caller)
        - Default -> 500ms
        """
        if not text:
            return self.base_silence_sec

        cleaned = text.strip().lower()
        words = re.sub(r"[^\w\s]", "", cleaned).split()
        if not words:
            return self.base_silence_sec

        last_word = words[-1]

        # 1. Utterance ends in connective/hesitation or ellipsis -> Extended hold (850 ms - 900 ms)
        if last_word in CONTINUATION_TOKENS or cleaned.endswith("...") or cleaned.endswith("…"):
            return self.base_silence_sec + self.hesitation_extra_sec

        # 2. Short crisp affirmative response -> Ultra-fast turn-around (250 ms)
        if len(words) <= 3 and any(w in COMPLETE_AFFIRMATIONS for w in words):
            return self.short_turn_sec

        # 3. Utterance ends in a terminal mark (period, exclamation) with >= 4 words -> Fast turn-around
        if cleaned.endswith((".", "!")) and len(words) >= 4:
            return self.short_turn_sec + 0.05

        return self.base_silence_sec

    def reset(self) -> None:
        """Reset state and clear buffers."""
        self.apm.reset()
        self.is_user_speaking = False
        self.speech_frames = 0
        self.last_speech_time = 0.0
        self.barge_in_triggered = False
        self._utterance_frames.clear()
        self._preroll.clear()
