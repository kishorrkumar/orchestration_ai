import numpy as np
import pytest

from orchestration.protocol.audio import (
    FRAME_RATE,
    FRAME_SIZE,
    SAMPLE_RATE,
    AudioFrameBuffer,
    compute_rms,
    float32_to_int16,
    int16_to_float32,
)


def test_audio_constants():
    assert SAMPLE_RATE == 24000
    assert FRAME_RATE == 12.5
    assert FRAME_SIZE == 1920
    # 1920 samples at 24000 Hz is exactly 80ms
    assert (FRAME_SIZE / SAMPLE_RATE) == 0.08


def test_pcm_conversions():
    # Test float32 to int16 and back
    orig = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    i16 = float32_to_int16(orig)
    assert i16.dtype == np.int16
    assert i16[0] == -32767  # clipped and scaled
    assert i16[2] == 0
    assert i16[4] == 32767

    recovered = int16_to_float32(i16)
    assert recovered.dtype == np.float32
    assert np.allclose(orig, recovered, atol=1e-4)


def test_rms_energy():
    silence = np.zeros(1920, dtype=np.float32)
    assert compute_rms(silence) == 0.0

    sine = np.sin(np.linspace(0, 2 * np.pi * 10, 1920)).astype(np.float32)
    rms = compute_rms(sine)
    # RMS of sine is 1 / sqrt(2) ~= 0.7071
    assert pytest.approx(rms, rel=1e-2) == 0.7071


def test_frame_buffer_chunking():
    buffer = AudioFrameBuffer(dtype=np.float32)

    # Push 1000 samples (less than 1 frame = 1920)
    chunk1 = np.ones(1000, dtype=np.float32) * 0.1
    buffer.push_samples(chunk1)
    assert not buffer.has_frame()
    assert buffer.pop_frame() is None
    assert buffer.underruns == 1

    # Push another 1500 samples (total 2500 -> 1 frame + 580 remainder)
    chunk2 = np.ones(1500, dtype=np.float32) * 0.2
    buffer.push_samples(chunk2)
    assert buffer.has_frame()

    frame1 = buffer.pop_frame()
    assert frame1 is not None
    assert len(frame1) == 1920
    assert np.allclose(frame1[:1000], 0.1)
    assert np.allclose(frame1[1000:], 0.2)

    # Now remainder is 580 samples -> no more frame
    assert not buffer.has_frame()
    assert buffer.buffered_samples == 580


def test_frame_buffer_pop_all():
    buffer = AudioFrameBuffer(dtype=np.float32)
    # Push 4000 samples -> should yield 2 full frames (3840 samples) and leave 160
    samples = np.ones(4000, dtype=np.float32)
    buffer.push_samples(samples)

    frames = buffer.pop_all_available_frames()
    assert len(frames) == 2
    for f in frames:
        assert len(f) == FRAME_SIZE
    assert buffer.buffered_samples == 160


def test_frame_buffer_push_pcm_bytes():
    buffer = AudioFrameBuffer(dtype=np.float32)
    int16_samples = np.array([1000, 2000, -1000], dtype=np.int16)
    buffer.push_pcm_bytes(int16_samples.tobytes(), is_int16=True)
    assert buffer.buffered_samples == 3
    assert buffer._buffer.dtype == np.float32


def test_frame_buffer_partial_byte_remainder_and_odd_bytes():
    """AUDIT-004: Push odd number of bytes across chunks without crashing on np.frombuffer."""
    buffer = AudioFrameBuffer(dtype=np.float32)
    int16_samples = np.array([500, -500], dtype=np.int16)
    full_bytes = int16_samples.tobytes()  # 4 bytes
    assert len(full_bytes) == 4

    # Push 3 bytes (first sample + 1 byte of second sample)
    buffer.push_pcm_bytes(full_bytes[:3], is_int16=True)
    assert buffer.buffered_samples == 1  # Only 1 complete int16 sample so far

    # Push the remaining 1 byte of the second sample
    buffer.push_pcm_bytes(full_bytes[3:], is_int16=True)
    assert buffer.buffered_samples == 2
    assert buffer._buffer[0] == pytest.approx(500.0 / 32767.0, rel=1e-3)
    assert buffer._buffer[1] == pytest.approx(-500.0 / 32767.0, rel=1e-3)


def test_int16_to_float32_asymmetry_bounds():
    """AUDIT-013: int16 min (-32768) must not exceed -1.0."""
    extreme = np.array([-32768, 32767], dtype=np.int16)
    converted = int16_to_float32(extreme)
    assert converted[0] >= -1.0
    assert converted[0] == -1.0
    assert converted[1] <= 1.0


def test_opus_codec_roundtrip():
    """AUDIT-001: sphn OpusStreamWriter and OpusStreamReader roundtrip for upstream Moshi compatibility."""
    import sphn
    sr = 24000
    writer = sphn.OpusStreamWriter(sr)
    reader = sphn.OpusStreamReader(sr)

    # 1920-sample audio frame (80ms at 24kHz)
    t = np.linspace(0, 0.08, 1920, endpoint=False)
    original_pcm = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    total_decoded = 0
    for _ in range(5):
        writer.append_pcm(original_pcm)
        opus_bytes = writer.read_bytes()
        if opus_bytes:
            reader.append_bytes(opus_bytes)
        pcm = reader.read_pcm()
        total_decoded += len(pcm)

    assert total_decoded >= 1920, f"Expected decoded frames, got {total_decoded} samples"


