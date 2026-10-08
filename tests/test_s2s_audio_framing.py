"""
Unit tests for PersonaPlex S2S Inbound Audio Framing & Parser (Bug 1).

Verifies:
1. Valid 640-byte PCM16 audio frames decode cleanly.
2. Raw PCM16 chunks starting with byte 0x01 are NOT stripped to odd length (Bug 1 fix).
3. Odd-length network chunks are safely buffered in carryover and reassembled without ValueError.
4. Empty chunks (0 bytes) are skipped safely.
5. Split/fragmented frames across chunk boundaries reassemble into exact sample counts.
6. Non-audio control frames (0x03 interrupt, 0x06 ping, 0x00 handshake) are identified and
   NEVER sent to PCM decoding.
7. Corrupted byte sequences are caught and skipped without crashing the session.
"""

from __future__ import annotations

import numpy as np
import pytest

from orchestration.audio.framing import InboundAudioFrameProcessor


def test_framing_valid_pcm16_frame():
    """640-byte (320 samples of 16-bit PCM at 16kHz) frame processes correctly."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    samples = (np.sin(np.linspace(0, 3.14, 320)) * 10000).astype(np.int16)
    data = samples.tobytes()
    assert len(data) == 640

    event_type, frames_24k, f32_samples = processor.process_frame(data)
    assert event_type == "audio"
    assert f32_samples is not None
    assert len(f32_samples) == 320
    np.testing.assert_allclose(f32_samples, samples / 32768.0, atol=1e-4)


def test_framing_raw_pcm_with_leading_0x01_not_stripped():
    """
    CRITICAL BUG 1 REGRESSION TEST:
    A raw PCM16 frame whose first byte happens to be 0x01 (e.g. sample value 1 = b'\\x01\\x00')
    must NOT have its first byte stripped, which previously caused odd-length 639 bytes
    and raised 'ValueError: buffer size must be a multiple of element size'.
    """
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    # Construct 320 int16 samples where sample 0 is exactly 1 (little-endian b'\x01\x00')
    samples = np.zeros(320, dtype=np.int16)
    samples[0] = 1  # b'\x01\x00'
    data = samples.tobytes()
    assert data[0] == 0x01
    assert len(data) == 640

    event_type, frames_24k, f32_samples = processor.process_frame(data)
    assert event_type == "audio"
    assert f32_samples is not None
    assert len(f32_samples) == 320
    # Sample 0 must be 1 / 32767, NOT corrupted or shifted
    assert abs(f32_samples[0] - (1.0 / 32768.0)) < 1e-4


def test_framing_framed_audio_with_0x01_stripped():
    """If a client explicitly prefixes 0x01 to 640 bytes of PCM (len 641, odd), 0x01 IS stripped."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    samples = np.ones(320, dtype=np.int16) * 500
    framed_data = b"\x01" + samples.tobytes()
    assert len(framed_data) == 641

    event_type, frames_24k, f32_samples = processor.process_frame(framed_data)
    assert event_type == "audio"
    assert f32_samples is not None
    assert len(f32_samples) == 320


def test_framing_odd_length_carryover_reassembly():
    """Odd-length chunk leaves trailing byte in carry-over and reassembles on next chunk."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    samples = np.arange(320, dtype=np.int16)
    full_bytes = samples.tobytes()  # 640 bytes

    # Deliver chunk of 639 bytes (odd)
    chunk1 = full_bytes[:639]
    event1, _, f32_1 = processor.process_frame(chunk1)
    assert event1 == "audio"
    assert len(f32_1) == 319  # 638 bytes decoded (319 int16 samples)

    # Deliver second chunk with 1 byte (completes the sample) + 640 bytes
    samples2 = np.arange(320, dtype=np.int16) * 2
    chunk2 = full_bytes[639:] + samples2.tobytes()  # 1 + 640 = 641 bytes
    event2, _, f32_2 = processor.process_frame(chunk2)
    assert event2 == "audio"
    assert len(f32_2) == 321  # 1 reassembled sample + 320 new samples = 321 samples


def test_framing_split_frame():
    """A 640-byte frame split across two WebSocket messages (301 bytes + 339 bytes) reassembles cleanly."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    samples = np.arange(320, dtype=np.int16)
    full_bytes = samples.tobytes()  # 640 bytes

    chunk_a = full_bytes[:301]  # 300 bytes usable (150 samples), 1 byte in carryover
    chunk_b = full_bytes[301:]  # 339 bytes + 1 in carryover = 340 bytes (170 samples)

    event_a, _, f32_a = processor.process_frame(chunk_a)
    event_b, _, f32_b = processor.process_frame(chunk_b)

    assert event_a == "audio"
    assert event_b == "audio"
    assert len(f32_a) == 150
    assert len(f32_b) == 170
    assert len(f32_a) + len(f32_b) == 320

    combined = np.concatenate([f32_a, f32_b])
    np.testing.assert_allclose(combined, samples / 32768.0, atol=1e-4)


def test_framing_empty_chunk():
    """Empty 0-byte chunk returns 'empty' and causes no exceptions."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    event, frames_24k, f32_samples = processor.process_frame(b"")
    assert event == "empty"
    assert len(frames_24k) == 0
    assert f32_samples is None


def test_framing_control_frame_opcode_3():
    """b'\\x03\\x02' interrupt control frame is recognized as interrupt and never decoded as PCM."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    event, frames_24k, f32_samples = processor.process_frame(b"\x03\x02")
    assert event == "interrupt"
    assert len(frames_24k) == 0
    assert f32_samples is None


def test_framing_control_frame_ping_and_handshake():
    """Ping (0x06) and Handshake (0x00) control frames are identified and isolated from PCM."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)

    event_ping, _, _ = processor.process_frame(b"\x06")
    assert event_ping == "ping"

    event_hs, _, _ = processor.process_frame(b"\x00")
    assert event_hs == "control"


def test_framing_corrupt_data_does_not_crash():
    """Corrupted bytes (e.g. single odd byte with no successor) do not crash processor."""
    processor = InboundAudioFrameProcessor(codec="pcm16", sample_rate=16000)
    # A single byte cannot form an int16 sample
    event, frames_24k, f32_samples = processor.process_frame(b"\x42")
    assert event == "partial"
    assert len(frames_24k) == 0
    assert f32_samples is None
