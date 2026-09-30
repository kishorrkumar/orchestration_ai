"""
Unit tests for Conversational Turn Detector and Trailing Incompleteness Detection.
"""

import time
import numpy as np
import pytest
from orchestration.audio.turn_detector import (
    TurnDetector,
    is_utterance_unfinished,
    TRAILING_INCOMPLETE_WORDS,
)


def test_is_utterance_unfinished():
    # Incomplete utterances ending in connectives/prepositions
    assert is_utterance_unfinished("Tell me a joke about") is True
    assert is_utterance_unfinished("Can you tell me and") is True
    assert is_utterance_unfinished("I want to know because") is True
    assert is_utterance_unfinished("What is the") is True
    assert is_utterance_unfinished("So like") is True
    assert is_utterance_unfinished("Actually I was thinking...") is True

    # Complete utterances
    assert is_utterance_unfinished("Tell me a joke about cats.") is False
    assert is_utterance_unfinished("How does UPI work?") is False
    assert is_utterance_unfinished("The weather in Bengaluru is pleasant today.") is False
    assert is_utterance_unfinished("I understand.") is False


def test_turn_detector_silence_and_speech():
    detector = TurnDetector(
        sample_rate=24000,
        frame_size=1920,
        base_silence_sec=0.20,  # Shortened for rapid unit test
        extra_silence_sec=0.25,
        min_speech_frames=2,
        speech_rms_threshold=0.02,
    )

    speech_frame = np.ones(1920, dtype=np.float32) * 0.05
    silence_frame = np.zeros(1920, dtype=np.float32)

    # Frame 1: speech onset
    spk, done, _ = detector.push_frame(speech_frame)
    assert done is False

    # Frame 2: speech confirmed
    spk, done, _ = detector.push_frame(speech_frame)
    assert spk is True
    assert done is False

    # Frame 3: silence frame
    spk, done, _ = detector.push_frame(silence_frame)
    assert done is False

    # Wait past silence threshold
    time.sleep(0.22)
    spk, done, audio = detector.push_frame(silence_frame)
    assert done is True
    assert audio is not None
    assert len(audio) > 0


def test_turn_detector_extension_and_merge():
    """Verify that an unfinished sentence gets extended and merged into ONE turn."""
    detector = TurnDetector(
        sample_rate=24000,
        frame_size=1920,
        base_silence_sec=0.15,
        extra_silence_sec=0.30,
        min_speech_frames=2,
        speech_rms_threshold=0.02,
    )

    speech_part1 = np.ones(1920, dtype=np.float32) * 0.06
    silence_frame = np.zeros(1920, dtype=np.float32)
    speech_part2 = np.ones(1920, dtype=np.float32) * 0.07

    # Part 1 speech
    detector.push_frame(speech_part1)
    detector.push_frame(speech_part1)

    time.sleep(0.16)
    _, done1, audio1 = detector.push_frame(silence_frame)
    assert done1 is True
    assert audio1 is not None

    # Syntactic check says "Tell me a joke about" is unfinished!
    assert is_utterance_unfinished("Tell me a joke about") is True

    # Put back audio and extend
    detector.append_audio_and_extend(audio1, extra_sec=0.30)

    # User resumes speaking within the extension window ("...cats!")
    time.sleep(0.05)
    detector.push_frame(speech_part2)
    detector.push_frame(speech_part2)

    # Silence concludes the merged turn
    time.sleep(0.16)
    _, done2, merged_audio = detector.push_frame(silence_frame)
    assert done2 is True
    assert merged_audio is not None

    # Merged audio contains both Part 1 and Part 2 frames!
    assert len(merged_audio) >= len(audio1) + (2 * 1920)
