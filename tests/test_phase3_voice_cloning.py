"""
Phase 3: Production Voice Cloning & Safety Benchmark & Unit Tests.

Validates:
1. tools/make_voice_prompt.py CLI compilation tool.
2. Safety enforcement:
   - Explicit user consent requirement.
   - Public figure / celebrity blocklist enforcement.
3. Pre-flight acoustic quality gates:
   - Minimum SNR threshold (>= 15.0 dB).
   - Maximum digital clipping limit (< 1.0%).
   - Duration bounds (5.0s to 30.0s).
   - Silence / speech presence ratio.
4. DSP preprocessing:
   - 80 Hz high-pass filter.
   - Resampling to 24,000 Hz mono PCM.
   - Silence trimming & optimal window selection (5.0s).
   - EBU R128 loudness normalization to -24.0 LUFS.
5. Artifact generation:
   - 24 kHz mono 16-bit WAV artifact.
   - PyTorch .pt voice prompt container.
   - metadata.json with acoustic and safety telemetry.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import pytest
import numpy as np
import soundfile as sf

from tools.make_voice_prompt import (
    VoiceCloningQualityError,
    VoiceCloningSafetyError,
    decode_audio_file,
    enforce_safety_and_consent,
    high_pass_filter,
    main,
    make_voice_prompt,
    normalize_loudness,
    resample_to_24k,
    trim_silence,
    validate_audio_quality,
)


def _create_synthetic_clean_speech(
    duration_sec: float = 6.0,
    sr: int = 24000,
    f0: float = 180.0,
    rms_target: float = 0.08,
) -> np.ndarray:
    """Generate high-quality synthetic voiced speech with harmonic formants."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Voiced harmonics
    carrier = (
        0.50 * np.sin(2 * np.pi * f0 * t) +
        0.30 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.20 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.10 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # Syllable modulation (conversational bursts)
    envelope = np.clip(np.sin(2 * np.pi * 2.5 * t), 0.05, 1.0) ** 1.2
    speech = carrier * envelope
    rms = float(np.sqrt(np.mean(speech ** 2)))
    if rms > 0:
        speech = speech * (rms_target / rms)
    return speech.astype(np.float32)


def _write_temp_wav(audio: np.ndarray, path: pathlib.Path, sr: int = 24000) -> pathlib.Path:
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


# =====================================================================
# Safety & Consent Tests
# =====================================================================

def test_safety_consent_required():
    with pytest.raises(VoiceCloningSafetyError, match="explicit user consent"):
        enforce_safety_and_consent(voice_name="Kishore Voice", consent=False)


@pytest.mark.parametrize("celebrity_name", [
    "Donald Trump",
    "Barack Obama",
    "Joe Biden",
    "Narendra Modi",
    "Elon Musk",
    "Taylor Swift",
    "Morgan Freeman",
])
def test_safety_public_figure_blocked(celebrity_name: str):
    with pytest.raises(VoiceCloningSafetyError, match="strictly prohibited by safety policy"):
        enforce_safety_and_consent(voice_name=celebrity_name, consent=True)


def test_safety_valid_consent_statement():
    statement = enforce_safety_and_consent(
        voice_name="Priya Customer Support",
        consent=True,
        consent_statement="I hereby authorize Priya Customer Support voice synthesis.",
    )
    assert "Priya Customer Support" in statement


# =====================================================================
# Acoustic Pre-Flight Quality Gates Tests
# =====================================================================

def test_quality_gate_duration_bounds():
    sr = 24000
    # Too short: 3.0 seconds (< 5.0s minimum)
    short_audio = _create_synthetic_clean_speech(duration_sec=3.0, sr=sr)
    with pytest.raises(VoiceCloningQualityError, match="too short"):
        validate_audio_quality(short_audio, sr=sr, min_duration_sec=5.0)

    # Too long: 35.0 seconds (> 30.0s maximum)
    long_audio = _create_synthetic_clean_speech(duration_sec=35.0, sr=sr)
    with pytest.raises(VoiceCloningQualityError, match="exceeds maximum"):
        validate_audio_quality(long_audio, sr=sr, max_duration_sec=30.0)


def test_quality_gate_low_snr_rejection():
    sr = 24000
    clean = _create_synthetic_clean_speech(duration_sec=6.0, sr=sr, rms_target=0.08)
    # Inject heavy Gaussian noise to drop SNR to ~5 dB
    noisy = clean + (np.random.randn(len(clean)).astype(np.float32) * 0.05)

    with pytest.raises(VoiceCloningQualityError, match="excessive background noise"):
        validate_audio_quality(noisy, sr=sr, min_snr_db=15.0)


def test_quality_gate_excessive_clipping_rejection():
    sr = 24000
    speech = _create_synthetic_clean_speech(duration_sec=6.0, sr=sr, rms_target=0.50)
    # Artificially hard-clip 5% of samples
    clipped = np.clip(speech * 3.0, -1.0, 1.0)
    num_clipped = np.sum(np.abs(clipped) >= 0.999)
    assert (num_clipped / len(clipped)) > 0.015

    with pytest.raises(VoiceCloningQualityError, match="excessive digital clipping"):
        validate_audio_quality(clipped, sr=sr, max_clipping_ratio=0.01)


def test_quality_gate_quiet_silence_rejection():
    sr = 24000
    silent_audio = np.zeros(sr * 6, dtype=np.float32)
    with pytest.raises(VoiceCloningQualityError, match="too quiet or silent"):
        validate_audio_quality(silent_audio, sr=sr)


# =====================================================================
# DSP Preprocessing Tests
# =====================================================================

def test_dsp_high_pass_filter_attenuates_low_frequencies():
    fs = 24000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # 40 Hz desk rumble vs 1000 Hz speech tone
    rumble = 0.5 * np.sin(2 * np.pi * 40 * t)
    speech = 0.5 * np.sin(2 * np.pi * 1000 * t)
    mixed = rumble + speech

    filtered = high_pass_filter(mixed, cutoff_hz=80.0, fs=fs)

    # 40 Hz should be significantly attenuated (> 6 dB)
    fft_in = np.abs(np.fft.rfft(mixed))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(fs, 1 / fs)

    idx_40 = int(np.argmin(np.abs(freqs - 40)))
    idx_1000 = int(np.argmin(np.abs(freqs - 1000)))

    att_40_db = 20 * np.log10(fft_in[idx_40] / max(1e-6, fft_out[idx_40]))
    att_1000_db = 20 * np.log10(fft_in[idx_1000] / max(1e-6, fft_out[idx_1000]))

    assert att_40_db > 6.0, f"40 Hz rumble should be attenuated > 6dB, got {att_40_db:.2f}dB"
    assert att_1000_db < 1.0, f"1000 Hz speech should pass untouched, got {att_1000_db:.2f}dB"


def test_dsp_loudness_normalization_and_limiting():
    # Quiet speech
    quiet = _create_synthetic_clean_speech(duration_sec=5.0, rms_target=0.01)
    normalized = normalize_loudness(quiet, target_lufs=-24.0)

    # Output RMS should be close to -24 dBFS (~0.063)
    norm_rms = float(np.sqrt(np.mean(normalized ** 2)))
    assert 0.040 <= norm_rms <= 0.080

    # Ceiling limiter must ensure peak <= 0.92
    assert float(np.max(np.abs(normalized))) <= 0.92


def test_dsp_resampling_to_24k():
    # 16 kHz input signal
    speech_16k = _create_synthetic_clean_speech(duration_sec=2.0, sr=16000)
    speech_24k = resample_to_24k(speech_16k, orig_sr=16000, target_sr=24000)

    assert len(speech_24k) == 24000 * 2
    assert speech_24k.dtype == np.float32


# =====================================================================
# End-to-End Pipeline & Artifact Generation Tests
# =====================================================================

def test_make_voice_prompt_pipeline_e2e(tmp_path: pathlib.Path):
    clean_audio = _create_synthetic_clean_speech(duration_sec=7.0, sr=24000, f0=210.0)
    input_wav = tmp_path / "enrollment.wav"
    _write_temp_wav(clean_audio, input_wav)

    output_dir = tmp_path / "compiled_voices"

    metadata = make_voice_prompt(
        input_file=input_wav,
        output_dir=output_dir,
        voice_name="Sunita Banking Agent",
        consent=True,
        consent_statement="I authorize Sunita voice cloning.",
        preferred_gender="Female",
        target_duration_sec=5.0,
    )

    assert metadata["id"].startswith("cloned_sunitabankingagent")
    assert metadata["gender"] == "Female"
    assert metadata["duration_sec"] == 5.0
    assert metadata["sample_rate"] == 24000
    assert metadata["quality_metrics"]["snr_db"] >= 15.0
    assert metadata["quality_metrics"]["clipping_ratio"] < 0.01

    # Check WAV artifact
    wav_path = pathlib.Path(metadata["artifact_wav"])
    assert wav_path.exists()
    disk_data, disk_sr = sf.read(str(wav_path))
    assert disk_sr == 24000
    assert len(disk_data) == 24000 * 5

    # Check PT artifact
    pt_path = pathlib.Path(metadata["artifact_pt"])
    assert pt_path.exists()
    assert pt_path.stat().st_size > 1024

    # Check metadata.json file on disk
    meta_path = wav_path.parent / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, encoding="utf-8") as f:
        disk_meta = json.load(f)
    assert disk_meta["id"] == metadata["id"]
    assert disk_meta["source_sha256"] == metadata["source_sha256"]


def test_make_voice_prompt_cli_tool(tmp_path: pathlib.Path, capsys):
    clean_audio = _create_synthetic_clean_speech(duration_sec=6.0, sr=24000, f0=140.0)
    input_wav = tmp_path / "voice_input.wav"
    _write_temp_wav(clean_audio, input_wav)

    output_dir = tmp_path / "voices_out"

    # Execute CLI via main()
    cli_args = [
        "-i", str(input_wav),
        "-o", str(output_dir),
        "-n", "Rajesh Operations",
        "--consent",
        "--gender", "Male",
        "--duration", "5.0",
        "--json",
    ]
    ret_code = main(cli_args)
    assert ret_code == 0

    captured = capsys.readouterr()
    res = json.loads(captured.out)

    assert res["id"].startswith("cloned_rajeshoperations")
    assert res["gender"] == "Male"
    assert res["duration_sec"] == 5.0
    assert pathlib.Path(res["artifact_wav"]).exists()
    assert pathlib.Path(res["artifact_pt"]).exists()


def test_make_voice_prompt_cli_missing_consent_fails(tmp_path: pathlib.Path):
    clean_audio = _create_synthetic_clean_speech(duration_sec=6.0, sr=24000)
    input_wav = tmp_path / "input.wav"
    _write_temp_wav(clean_audio, input_wav)

    # Missing --consent
    cli_args = [
        "-i", str(input_wav),
        "-o", str(tmp_path / "out"),
        "-n", "Test Voice",
    ]
    ret_code = main(cli_args)
    assert ret_code == 1
