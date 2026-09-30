"""
Turn Detector for Voice Conversations.

Solves the mid-sentence cutoff problem (e.g. "Tell me a joke about... [500ms pause] ...cats"):
- Pre-roll ring buffer prevents clipped leading syllables.
- Base silence threshold: ~650 ms.
- Syntactic & lexical incompleteness detection:
  If an utterance ends with connectives/prepositions (about, and, the, to, of, with, because, but, so, like, or, for, that)
  or ends with ellipsis or is a fragment with no finite verb, extends the silence timeout by ~700 ms.
- If speech resumes within the extended timeout, merges prior and subsequent frames into ONE turn.
"""

from __future__ import annotations
import collections
import re
import time
from typing import Any, List, Optional, Tuple
import numpy as np

# Connectives, prepositions, conjunctions, and articles that signal an incomplete thought
TRAILING_INCOMPLETE_WORDS = {
    "about", "and", "the", "to", "of", "with", "because", "but", "so",
    "like", "or", "for", "that", "if", "when", "which", "as", "than",
    "then", "into", "onto", "from", "by", "my", "your", "his", "her",
    "their", "our", "its", "a", "an", "is", "are", "was", "were",
    "am", "be", "being", "been", "what", "who", "whom", "how", "why", "where"
}

# Common English verbs to detect whether a multi-word fragment contains an action
COMMON_VERBS = {
    "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "say", "says", "said", "go", "goes", "went", "gone",
    "get", "gets", "got", "make", "makes", "made", "know", "knows", "knew",
    "think", "thinks", "thought", "take", "takes", "took", "see", "sees", "saw",
    "come", "comes", "came", "want", "wants", "wanted", "look", "looks", "looked",
    "use", "uses", "used", "find", "finds", "found", "give", "gives", "gave",
    "tell", "tells", "told", "work", "works", "worked", "call", "calls", "called",
    "try", "tries", "tried", "ask", "asks", "asked", "need", "needs", "needed",
    "feel", "feels", "felt", "become", "becomes", "became", "leave", "leaves", "left",
    "put", "puts", "mean", "means", "meant", "keep", "keeps", "kept", "let", "lets",
    "begin", "begins", "began", "seem", "seems", "seemed", "help", "helps", "helped",
    "talk", "talks", "talked", "turn", "turns", "turned", "start", "starts", "started",
    "show", "shows", "showed", "hear", "hears", "heard", "play", "plays", "played",
    "run", "runs", "ran", "move", "moves", "moved", "like", "likes", "liked",
    "live", "lives", "lived", "believe", "believes", "believed", "hold", "holds", "held",
    "bring", "brings", "brought", "happen", "happens", "happened", "write", "writes", "wrote",
    "provide", "provides", "provided", "sit", "sits", "sat", "stand", "stands", "stood",
    "lose", "loses", "lost", "pay", "pays", "paid", "meet", "meets", "met",
    "include", "includes", "included", "continue", "continues", "continued", "set", "sets",
    "learn", "learns", "learnt", "learned", "change", "changes", "changed", "lead", "leads", "led",
    "understand", "understands", "understood", "watch", "watches", "watched",
    "follow", "follows", "followed", "stop", "stops", "stopped", "create", "creates", "created",
    "speak", "speaks", "spoke", "spoken", "read", "reads", "spend", "spends", "spent",
    "open", "opens", "opened", "walk", "walks", "walked", "win", "wins", "won",
    "teach", "teaches", "taught", "offer", "offers", "offered", "remember", "remembers", "remembered",
    "consider", "considers", "considered", "appear", "appears", "appeared", "buy", "buys", "bought",
    "serve", "serves", "served", "die", "dies", "died", "send", "sends", "sent",
    "build", "builds", "built", "stay", "stays", "stayed", "fall", "falls", "fell",
    "cut", "cuts", "reach", "reaches", "reached", "kill", "kills", "killed", "remain", "remains",
}


def is_utterance_unfinished(text: str) -> bool:
    """
    Check if an utterance is syntactically or lexically incomplete.
    Returns True if the text ends on a connective, preposition, ellipsis,
    or is a fragment without a finite verb.
    """
    if not text:
        return True

    cleaned = text.strip().lower()
    # Strip trailing punctuation for word check, but inspect if it ended in '...'
    if cleaned.endswith("...") or cleaned.endswith("…"):
        return True

    # Strip standard terminal punctuation
    cleaned_words = re.sub(r"[^\w\s]", "", cleaned).split()
    if not cleaned_words:
        return True

    last_word = cleaned_words[-1]
    if last_word in TRAILING_INCOMPLETE_WORDS:
        return True

    # Check for short multi-word fragments with no verb (e.g. "a blue", "the funny cat")
    if 2 <= len(cleaned_words) <= 4:
        has_verb = any(w in COMMON_VERBS for w in cleaned_words)
        if not has_verb:
            return True

    return False


class TurnDetector:
    """
    Stateful conversational turn detector with Silero VAD, pre-roll buffer,
    hysteresis, external noise rejection, and linguistic continuation extension.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_size: int = 1920,
        base_silence_sec: float = 0.65,
        extra_silence_sec: float = 0.70,
        min_speech_frames: int = 3,       # ~240 ms of speech to confirm onset
        preroll_frames: int = 4,          # ~320 ms pre-roll buffer
        speech_rms_threshold: float = 0.018,
        vad: Optional[Any] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.base_silence_sec = base_silence_sec
        self.extra_silence_sec = extra_silence_sec
        self.min_speech_frames = min_speech_frames
        self.speech_rms_threshold = speech_rms_threshold
        self.vad = vad

        # Audio buffers
        self._preroll: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_frames)
        self._active_utterance_frames: List[np.ndarray] = []

        # State tracking
        self.is_speaking: bool = False
        self.speech_frame_count: int = 0
        self.last_speech_time: float = 0.0
        self.waiting_for_extension: bool = False
        self.extension_deadline: float = 0.0
        self.last_prob: float = 0.0

    def push_frame(
        self,
        frame: np.ndarray,
        rms: Optional[float] = None,
        is_agent_speaking: bool = False,
    ) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """
        Process a single audio frame (typically 80 ms).
        
        Returns:
            (is_speaking_now, turn_completed, completed_audio_if_any)
        """
        now = time.time()
        if rms is None:
            rms = float(np.sqrt(np.mean(np.square(frame))))

        # 1. Silero VAD speech probability & noise rejection
        if self.vad is not None:
            speech_prob = self.vad.get_speech_probability(frame)
        else:
            speech_prob = min(1.0, rms * 35.0)
        self.last_prob = speech_prob

        # 2. Dynamic acoustic echo & noise gating
        if is_agent_speaking:
            # When agent is playing out of laptop speakers, require confident voice to barge in
            is_frame_speech = speech_prob > 0.68 and rms > 0.022
        else:
            # Human speech: Silero VAD is primary arbiter, rejecting background clicks and fan hum
            is_frame_speech = speech_prob >= 0.38 or (rms >= self.speech_rms_threshold and speech_prob >= 0.28)

        if is_frame_speech:
            self.speech_frame_count += 1
            self.last_speech_time = now

            # If we were waiting in an extended silence window, resume into the SAME turn!
            if self.waiting_for_extension:
                self.waiting_for_extension = False
                self.is_speaking = True

            # If onset confirmed:
            if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
                # Prepend pre-roll buffer so starting plosives/fricatives are preserved
                for pf in self._preroll:
                    self._active_utterance_frames.append(pf)
                self._preroll.clear()

            if self.is_speaking:
                self._active_utterance_frames.append(frame)
            else:
                self._preroll.append(frame)

            return (self.is_speaking, False, None)

        else:
            # Silence frame
            self._preroll.append(frame)

            if not self.is_speaking and not self.waiting_for_extension:
                if now - self.last_speech_time > 0.4:
                    self.speech_frame_count = 0
                return (False, False, None)

            # In an active turn, check silence duration
            silence_duration = now - self.last_speech_time

            # If waiting in extension window
            if self.waiting_for_extension:
                if now >= self.extension_deadline:
                    # Extended silence deadline expired: conclude turn
                    self.waiting_for_extension = False
                    self.is_speaking = False
                    self.speech_frame_count = 0
                    if self._active_utterance_frames:
                        complete_audio = np.concatenate(self._active_utterance_frames)
                        self._active_utterance_frames.clear()
                        return (False, True, complete_audio)
                return (False, False, None)

            # Check if base silence is reached
            if self.is_speaking and silence_duration >= self.base_silence_sec:
                # We reached base silence. Caller might finish or might need extension.
                # By default, finalize unless caller triggers extend_turn()
                self.is_speaking = False
                self.speech_frame_count = 0
                if self._active_utterance_frames:
                    complete_audio = np.concatenate(self._active_utterance_frames)
                    self._active_utterance_frames.clear()
                    return (False, True, complete_audio)

            return (self.is_speaking, False, None)

    def extend_turn(self, extra_sec: Optional[float] = None) -> None:
        """
        Extend the silence window when the transcript is detected as unfinished.
        Preserves active utterance frames and waits for additional speech.
        """
        add_time = extra_sec if extra_sec is not None else self.extra_silence_sec
        self.waiting_for_extension = True
        self.extension_deadline = time.time() + add_time

    def append_audio_and_extend(self, audio: np.ndarray, extra_sec: Optional[float] = None) -> None:
        """Put back previously emitted audio and extend deadline for merging."""
        self._active_utterance_frames.insert(0, audio)
        self.extend_turn(extra_sec)

    def reset(self) -> None:
        """Reset turn detector state."""
        self._preroll.clear()
        self._active_utterance_frames.clear()
        self.is_speaking = False
        self.speech_frame_count = 0
        self.last_speech_time = 0.0
        self.waiting_for_extension = False
        self.extension_deadline = 0.0
