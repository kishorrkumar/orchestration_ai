"""
Unit and integration tests for Strongest Noise Cancellation, Silero VAD, and Voice Isolation Layer.
"""

import time

import numpy as np

from orchestration.audio.cleaner import CallerAudioCleaner
from orchestration.protocol.audio import FRAME_SIZE, SAMPLE_RATE, compute_rms


def test_cleaner_mode_switching():
    cleaner = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, mode="isolation")
    assert cleaner.mode == "isolation"
    assert not cleaner.bypass

    cleaner.set_mode("rnnoise")
    assert cleaner.mode == "rnnoise"
    assert not cleaner.bypass

    cleaner.set_mode("bypass")
    assert cleaner.mode == "bypass"
    assert cleaner.bypass

    cleaner.set_mode("isolation")
    assert cleaner.mode == "isolation"
    assert not cleaner.bypass


def test_bystander_noise_and_room_chatter_silenced():
    """
    Simulates external people, TV, or air conditioning noise.
    In 'isolation' mode, low-energy background sound must be gated to 0 (pure silence).
    """
    cleaner = CallerAudioCleaner(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        mode="isolation",
        min_speech_rms=0.012,
        vad_threshold=0.45,
    )

    # Low-energy ambient room noise (RMS ~ 0.005)
    ambient_noise = (np.random.randn(FRAME_SIZE * 5) * 0.004).astype(np.float32)
    frames = cleaner.process_chunk(ambient_noise)

    assert len(frames) == 5
    for f in frames:
        # Must be completely silenced to prevent PersonaPlex from reacting to external sounds
        assert np.max(np.abs(f)) == 0.0
        assert compute_rms(f) == 0.0
    assert not cleaner.is_speech_active


def test_direct_speech_and_end_of_speech_hangover():
    """
    Simulates active direct speech, followed by pause.
    Verifies that active speech passes, and End of Speech (EOS) activates after hangover.
    """
    cleaner = CallerAudioCleaner(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        mode="isolation",
        vad_hangover_sec=0.20,
    )

    # Active direct speech (loud audio)
    loud_speech = (np.sin(2 * np.pi * 300 * np.linspace(0, 0.16, FRAME_SIZE * 2)) * 0.15).astype(np.float32)
    frames = cleaner.process_chunk(loud_speech)

    assert len(frames) == 2
    assert cleaner.is_speech_active
    for f in frames:
        assert compute_rms(f) > 0.0

    # User stops talking (silence/low noise)
    silent_chunk = np.zeros(FRAME_SIZE, dtype=np.float32)

    # Immediately after speech, hangover keeps gate open to bridge natural pauses
    hangover_frames = cleaner.process_chunk(silent_chunk)
    assert len(hangover_frames) == 1
    assert cleaner.is_speech_active

    # Sleep longer than vad_hangover_sec (0.20s)
    time.sleep(0.25)

    # Now End of Speech (EOS) must be triggered: gate closes and outputs pure silence
    eos_frames = cleaner.process_chunk(silent_chunk)
    assert len(eos_frames) == 1
    assert not cleaner.is_speech_active
    assert compute_rms(eos_frames[0]) == 0.0
