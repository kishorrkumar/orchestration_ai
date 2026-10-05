"""
End-of-Call Detection Engine for Full-Duplex S2S Conversational AI.
Watches agent text tokens (0x02), fuzzy-matches closing phrases, coordinates a 1.5s
quiet window post-audio, and cancels pending hang-up if the user speaks.
"""

from __future__ import annotations

import difflib
import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalizes text for reliable matching: lowercase, strip punctuation, single whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


class EndOfCallDetector:
    """
    Supervisory state machine detecting natural call closure, silence timeouts,
    and duration limits for models that cannot natively trigger hangup.
    """

    DEFAULT_FALLBACK_CLOSERS = [
        "goodbye",
        "take care bye",
        "have a great day",
        "have a good one bye",
        "talk to you later bye",
    ]

    WRAP_UP_MARKERS = [
        "thanks",
        "thank you",
        "all set",
        "sorted",
        "pleasure",
        "help you with anything else",
    ]

    def __init__(
        self,
        ending_text: str = "",
        quiet_window_sec: float = 1.5,
        silence_timeout_sec: float = 20.0,
        max_duration_sec: float = 600.0,
        fuzzy_threshold: float = 0.80,
    ):
        self.ending_text = ending_text
        self.quiet_window_sec = quiet_window_sec
        self.silence_timeout_sec = silence_timeout_sec
        self.max_duration_sec = max_duration_sec
        self.fuzzy_threshold = fuzzy_threshold

        # Extract target closing clauses from ending text
        self.target_clauses = self._extract_target_clauses(ending_text)

        # Internal state
        self.agent_rolling_text = ""
        self.matched_in_current_turn = False
        self.pending_hangup = False
        self.hangup_target_time: float | None = None
        self.last_speech_time = 0.0
        self.session_start_time = 0.0
        self.is_active = False

    def _extract_target_clauses(self, text: str) -> List[str]:
        """Splits ending text into candidate closing phrases."""
        clauses = []
        norm = normalize_text(text)
        if norm:
            clauses.append(norm)
            # Add the last 2-4 words as a key closer clause
            words = norm.split()
            if len(words) >= 2:
                clauses.append(" ".join(words[-3:]))
                clauses.append(" ".join(words[-2:]))
        for closer in self.DEFAULT_FALLBACK_CLOSERS:
            if closer not in clauses:
                clauses.append(closer)
        return list(set(clauses))

    def start_session(self, now: float) -> None:
        """Initializes the session timing state."""
        self.session_start_time = now
        self.last_speech_time = now
        self.agent_rolling_text = ""
        self.matched_in_current_turn = False
        self.pending_hangup = False
        self.hangup_target_time = None
        self.is_active = True

    def on_agent_token(self, token: str, now: float) -> bool:
        """
        Ingests a text token from opcode 0x02.
        Returns True if a closing phrase match is triggered.
        """
        if not self.is_active:
            return False

        self.last_speech_time = now
        self.agent_rolling_text += token

        # Keep rolling window to last 250 characters
        if len(self.agent_rolling_text) > 300:
            self.agent_rolling_text = self.agent_rolling_text[-250:]

        norm_rolling = normalize_text(self.agent_rolling_text)

        # Check for clause match
        for clause in self.target_clauses:
            if not clause:
                continue

            # Exact substring check
            if clause in norm_rolling:
                self.matched_in_current_turn = True
                return True

            # Fuzzy similarity check on trailing window of equal length
            clause_len = len(clause)
            if len(norm_rolling) >= clause_len:
                tail = norm_rolling[-int(clause_len * 1.3):]
                ratio = difflib.SequenceMatcher(None, clause, tail).ratio()
                if ratio >= self.fuzzy_threshold:
                    self.matched_in_current_turn = True
                    return True

        return False

    def on_agent_speaking_stopped(self, now: float) -> None:
        """
        Called when the agent stops emitting audio for this turn.
        If a closing phrase was spoken, begins the 1.5s quiet window.
        """
        if self.matched_in_current_turn and not self.pending_hangup:
            self.pending_hangup = True
            self.hangup_target_time = now + self.quiet_window_sec

    def on_user_speech(self, now: float) -> bool:
        """
        Called when the user speaks (mic VAD speech detected).
        Resets silence timer, and CRITICALLY cancels pending hangup if user wants to continue!
        Returns True if a pending hangup was canceled.
        """
        self.last_speech_time = now
        canceled = False

        if self.pending_hangup:
            # User cut in during the quiet window; they have more to say!
            self.pending_hangup = False
            self.hangup_target_time = None
            self.matched_in_current_turn = False
            canceled = True

        return canceled

    def check_termination(self, now: float) -> str | None:
        """
        Polls termination status.
        Returns end reason string if call should terminate, else None.
        """
        if not self.is_active:
            return None

        # 1. Natural agent closing (quiet window elapsed without user interruption)
        if self.pending_hangup and self.hangup_target_time is not None:
            if now >= self.hangup_target_time:
                self.is_active = False
                return "agent_closed"

        # 2. Silence timeout
        if (now - self.last_speech_time) >= self.silence_timeout_sec:
            self.is_active = False
            return "silence_timeout"

        # 3. Max call duration
        if (now - self.session_start_time) >= self.max_duration_sec:
            self.is_active = False
            return "max_duration"

        return None
