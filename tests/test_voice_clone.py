import io
import numpy as np
import pytest
import soundfile as sf

from orchestration.tts.voice_clone import (
    VoiceCloner,
    VoiceCloningValidationError,
)
from orchestration.audio.similarity import compute_speaker_similarity, SIMILARITY_PASS_THRESHOLD


def _generate_synthetic_speech_sample(duration_sec: float = 5.0, sr: int = 24000, f0: float = 180.0) -> bytes:
    """Generate a clean synthetic harmonic vocal signal for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Formants and fundamental frequency
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Modulate envelope to simulate conversational syllables
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    audio = (signal * envelope).astype(np.float32) * 0.5

    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_voice_cloning_consent_required(tmp_path):
    cloner = VoiceCloner(data_dir=tmp_path)
    sample_wav = _generate_synthetic_speech_sample(5.0)

    # Consent = False must fail
    with pytest.raises(VoiceCloningValidationError, match="consent"):
        cloner.clone_voice(
            audio_bytes=sample_wav,
            voice_name="Test Voice",
            consent=False,
        )


def test_voice_cloning_duration_validation(tmp_path):
    cloner = VoiceCloner(data_dir=tmp_path)
    # 1.5s sample is below min 3.0s
    short_wav = _generate_synthetic_speech_sample(1.5)

    with pytest.raises(VoiceCloningValidationError, match="too short"):
        cloner.clone_voice(
            audio_bytes=short_wav,
            voice_name="Short Voice",
            consent=True,
        )


def test_voice_cloning_silent_audio_rejected(tmp_path):
    cloner = VoiceCloner(data_dir=tmp_path)
    # Generate 5 seconds of absolute silence
    silent_audio = np.zeros(24000 * 5, dtype=np.float32)
    bio = io.BytesIO()
    sf.write(bio, silent_audio, 24000, format="WAV")

    with pytest.raises(VoiceCloningValidationError, match="quiet or silent"):
        cloner.clone_voice(
            audio_bytes=bio.getvalue(),
            voice_name="Silent Voice",
            consent=True,
        )


def test_voice_cloning_success_and_lifecycle(tmp_path):
    cloner = VoiceCloner(data_dir=tmp_path)
    sample_wav = _generate_synthetic_speech_sample(6.0, f0=220.0)

    meta = cloner.clone_voice(
        audio_bytes=sample_wav,
        voice_name="Elena Clone",
        owner="test_user",
        consent=True,
        preferred_gender="Female",
    )

    assert meta["id"].startswith("cloned_elenaclone")
    assert meta["gender"] == "Female"
    assert meta["sample_rate"] == 24000
    assert meta["duration_sec"] >= 4.0
    assert cloner.has_voice(meta["id"]) is True

    # Check disk artifact
    wav_path = cloner.get_voice_path(meta["id"])
    assert wav_path is not None
    assert wav_path.exists()

    # Verify audio properties on disk
    disk_data, disk_sr = sf.read(str(wav_path))
    assert disk_sr == 24000
    assert disk_data.ndim == 1

    # Check listing
    voices = cloner.list_cloned_voices()
    assert any(v["id"] == meta["id"] for v in voices)

    # Delete voice
    assert cloner.delete_voice(meta["id"]) is True
    assert cloner.has_voice(meta["id"]) is False
    assert not wav_path.exists()


def test_speaker_similarity_metric():
    sample_a = _generate_synthetic_speech_sample(5.0, f0=150.0)
    # Similar sample with slight pitch variation (152 Hz)
    sample_a_var = _generate_synthetic_speech_sample(5.0, f0=152.0)
    # Very different sample (high pitch 320 Hz female)
    sample_b = _generate_synthetic_speech_sample(5.0, f0=320.0)

    sim_self = compute_speaker_similarity(sample_a, sample_a)
    assert sim_self >= 0.95, f"Self-similarity should be ~1.0, got {sim_self}"

    sim_close = compute_speaker_similarity(sample_a, sample_a_var)
    assert sim_close >= SIMILARITY_PASS_THRESHOLD, f"Close speaker similarity {sim_close} < {SIMILARITY_PASS_THRESHOLD}"

    sim_diff = compute_speaker_similarity(sample_a, sample_b)
    assert sim_diff < sim_close, "Different speakers must have lower similarity than same speaker"


def test_get_cloned_style(tmp_path):
    """AUDIT-012: VoiceCloner must implement get_cloned_style for TTS backend compatibility."""
    cloner = VoiceCloner(data_dir=tmp_path)
    # Non-existent voice
    assert cloner.get_cloned_style("nonexistent") is None

    sample_wav = _generate_synthetic_speech_sample(5.0)
    meta = cloner.clone_voice(
        audio_bytes=sample_wav,
        voice_name="Style Test Voice",
        consent=True,
    )
    # Voice exists with only WAV artifact
    assert cloner.get_cloned_style(meta["id"]) is None

