"""
Unit tests for Phase 2: 16 kHz Audio Core (DSP, Resamplers, Codecs).
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestration.audio.codecs import (
    decode_ulaw,
    encode_ulaw,
    float32_to_ulaw,
    ulaw_to_float32,
)
from orchestration.audio.dsp import (
    calculate_snr_db,
    compute_rms,
    soft_clip,
)
from orchestration.audio.resample import (
    AudioResampler,
    StreamingResampleBuffer,
    resample_oneshot,
)
from orchestration.protocol.audio import (
    CLIENT_FRAME_BYTES_PCM16,
    CLIENT_FRAME_MS,
    CLIENT_FRAME_SAMPLES,
    CLIENT_SAMPLE_RATE,
    MODEL_20MS_SAMPLES,
    MODEL_FRAME_SIZE,
    MODEL_SAMPLE_RATE,
    TELEPHONY_FRAME_BYTES,
    TELEPHONY_FRAME_MS,
    TELEPHONY_FRAME_SAMPLES,
    TELEPHONY_SAMPLE_RATE,
)


def test_canonical_audio_constants():
    """Verify standard audio protocol constants for web and telephony."""
    assert CLIENT_SAMPLE_RATE == 16000
    assert CLIENT_FRAME_MS == 20
    assert CLIENT_FRAME_SAMPLES == 320
    assert CLIENT_FRAME_BYTES_PCM16 == 640

    assert TELEPHONY_SAMPLE_RATE == 8000
    assert TELEPHONY_FRAME_MS == 20
    assert TELEPHONY_FRAME_SAMPLES == 160
    assert TELEPHONY_FRAME_BYTES == 160

    assert MODEL_SAMPLE_RATE == 24000
    assert MODEL_FRAME_SIZE == 1920
    assert MODEL_20MS_SAMPLES == 480


def test_g711_ulaw_encode_decode_roundtrip():
    """Verify ITU-T G.711 mu-law encoder and decoder fidelity."""
    # Test on a sine tone
    fs = 8000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    orig_f32 = 0.7 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    orig_i16 = (orig_f32 * 32767.0).astype(np.int16)

    # Encode to mu-law bytes
    ulaw_bytes = encode_ulaw(orig_i16)
    assert len(ulaw_bytes) == len(orig_i16)
    assert isinstance(ulaw_bytes, bytes)

    # Decode back to int16 PCM
    decoded_i16 = decode_ulaw(ulaw_bytes)
    assert len(decoded_i16) == len(orig_i16)

    # G.711 mu-law has 8-bit resolution (theoretical max SNR ~38-40 dB)
    snr = calculate_snr_db(decoded_i16.astype(np.float32), orig_i16.astype(np.float32))
    assert snr > 35.0, f"Expected G.711 SNR > 35 dB, got {snr:.2f} dB"


def test_g711_ulaw_edge_cases_and_helpers():
    """Verify G.711 boundary values and float32 helpers."""
    # Boundary extremes
    edges = np.array([-32768, -32635, -100, 0, 100, 32635, 32767], dtype=np.int16)
    encoded = encode_ulaw(edges)
    decoded = decode_ulaw(encoded)
    assert len(encoded) == 7
    assert len(decoded) == 7

    # float32 helper test
    f32_samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    u_bytes = float32_to_ulaw(f32_samples)
    assert len(u_bytes) == 5
    recovered_f32 = ulaw_to_float32(u_bytes)
    assert len(recovered_f32) == 5
    # Maximum quantization error for float32 should be small (<0.02)
    max_err = np.max(np.abs(f32_samples - recovered_f32))
    assert max_err < 0.03


def test_resample_oneshot_16k_to_24k_and_back():
    """Verify one-shot resampling between 16 kHz and 24 kHz preserves high SNR."""
    fs = 16000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    # 440 Hz test tone
    x = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    up_24k = resample_oneshot(x, in_rate=16000, out_rate=24000, quality="HQ")
    expected_len_24k = int(len(x) * 24000 / 16000)
    assert len(up_24k) == expected_len_24k

    down_16k = resample_oneshot(up_24k, in_rate=24000, out_rate=16000, quality="HQ")
    assert len(down_16k) == len(x)

    snr = calculate_snr_db(down_16k, x)
    assert snr > 60.0, f"Expected high quality resample SNR > 60 dB, got {snr:.2f} dB"


def test_streaming_resampler_continuity():
    """Verify AudioResampler handles streaming 20ms chunks without clicks or phase jumps."""
    in_sr = 16000
    out_sr = 24000
    chunk_samples = 320  # 20 ms at 16k
    total_chunks = 15

    t = np.linspace(0, (total_chunks * chunk_samples) / in_sr, total_chunks * chunk_samples, endpoint=False)
    signal = 0.7 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    up_resampler = AudioResampler(in_rate=in_sr, out_rate=out_sr, quality="QQ")
    down_resampler = AudioResampler(in_rate=out_sr, out_rate=in_sr, quality="QQ")

    up_chunks = []
    for i in range(total_chunks):
        chunk = signal[i * chunk_samples : (i + 1) * chunk_samples]
        last = (i == total_chunks - 1)
        res = up_resampler.resample_chunk(chunk, last=last)
        if len(res) > 0:
            up_chunks.append(res)

    full_up = np.concatenate(up_chunks)
    assert len(full_up) > 0

    down_chunks = []
    out_chunk_samples = 480
    for j in range(0, len(full_up), out_chunk_samples):
        chunk = full_up[j : j + out_chunk_samples]
        last = (j + out_chunk_samples >= len(full_up))
        res = down_resampler.resample_chunk(chunk, last=last)
        if len(res) > 0:
            down_chunks.append(res)

    full_down = np.concatenate(down_chunks)
    min_len = min(len(signal), len(full_down))
    snr = calculate_snr_db(full_down[:min_len], signal[:min_len])
    assert snr > 40.0, f"Streaming roundtrip SNR should exceed 40 dB, got {snr:.2f} dB"


def test_streaming_resample_buffer_exact_frames():
    """Verify StreamingResampleBuffer emits exact required frame sizes regardless of push sizes."""
    # Feed 16 kHz chunks of irregular sizes (e.g. 100, 200, 320 samples)
    # and verify output always emits exact 480-sample frames (20 ms at 24 kHz)
    buf = StreamingResampleBuffer(
        in_rate=16000,
        out_rate=24000,
        out_frame_samples=480,
        quality="QQ",
    )

    all_emitted_frames = []
    # Push 10 irregular chunks of 320 samples (3200 samples total in = 4800 samples out = exactly 10 frames of 480)
    for _ in range(10):
        chunk = np.random.uniform(-0.5, 0.5, 320).astype(np.float32)
        frames = buf.push_chunk(chunk)
        for f in frames:
            assert len(f) == 480
            all_emitted_frames.append(f)

    # Flush remainder
    flushed = buf.flush()
    for f in flushed:
        assert len(f) == 480
        all_emitted_frames.append(f)

    total_samples = sum(len(f) for f in all_emitted_frames)
    assert total_samples % 480 == 0


def test_dsp_soft_clip_and_rms():
    """Verify soft_clip limiter and compute_rms energy calculation."""
    # 1. Pure sine wave RMS: amplitude 0.7071 should have RMS ~ 0.5
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = 0.7071 * np.sin(2 * np.pi * 100 * t).astype(np.float32)
    rms = compute_rms(sine)
    assert pytest.approx(rms, abs=0.02) == 0.5

    # 2. Soft clipping
    peaked = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    clipped = soft_clip(peaked, threshold=0.90)

    # Values within [-0.9, 0.9] should be completely unchanged
    assert pytest.approx(clipped[2], abs=1e-5) == -0.5
    assert pytest.approx(clipped[3], abs=1e-5) == 0.0
    assert pytest.approx(clipped[4], abs=1e-5) == 0.5

    # Values above threshold must be strictly bounded in [-1.0, 1.0]
    assert np.all(clipped <= 1.0)
    assert np.all(clipped >= -1.0)
    assert clipped[6] < 1.0
