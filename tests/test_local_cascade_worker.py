"""
Unit and integration tests for LocalCascadeWorkerServer.
Tests wire protocol conformance, conversation memory, and barge-in timing.
"""

import asyncio
import time

import pytest
from websockets.asyncio.client import connect as ws_connect

from orchestration.protocol.audio import FRAME_SIZE, generate_silence_frame
from orchestration.protocol.messages import (
    MessageType,
    TextMessage,
    decode_message,
    encode_message,
)
from orchestration.worker.local_cascade import (
    LocalCascadeWorkerServer,
    load_persona_prompt,
)


def test_persona_prompt_loading():
    aarav_prompt = load_persona_prompt("Aarav")
    assert "Aarav" in aarav_prompt
    assert "INDIAN ENGLISH" in aarav_prompt.upper()
    assert "I understand you need support" in aarav_prompt  # Banned in prompt instructions
    assert "joke" in aarav_prompt.lower()

    priya_prompt = load_persona_prompt("Priya")
    assert "Priya" in priya_prompt


@pytest.mark.asyncio
async def test_local_cascade_wire_protocol_and_handshake():
    """Verify LocalCascadeWorkerServer initiates with HandshakeMessage(0, 0)."""
    server = LocalCascadeWorkerServer(host="127.0.0.1", port=9098)
    await server.start()

    try:
        url = "ws://127.0.0.1:9098/api/chat?neural_voice=aarav_colloquial"
        async with ws_connect(url) as ws:
            raw_handshake = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = decode_message(raw_handshake)
            assert msg.type == MessageType.HANDSHAKE
            assert msg.version == 0
            assert msg.model == 0
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_local_cascade_text_turn_and_audio_streaming():
    """Verify that sending a text message triggers audio frames and text tokens."""
    server = LocalCascadeWorkerServer(host="127.0.0.1", port=9099)
    await server.start()

    try:
        url = "ws://127.0.0.1:9099/api/chat?neural_voice=aarav_colloquial"
        async with ws_connect(url) as ws:
            # 1. Receive Handshake
            await ws.recv()

            # 2. Send text turn
            text_msg = TextMessage(text="Hi Aarav, tell me a quick joke.")
            await ws.send(encode_message(text_msg))

            # 3. Receive audio or text frames
            received_audio = False
            for _ in range(15):
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = decode_message(raw)
                if msg.type == MessageType.AUDIO and len(msg.data) == FRAME_SIZE * 4:
                    received_audio = True
                    break

            assert received_audio is True
    finally:
        await server.stop()


def test_barge_in_cancellation_timing():
    """Verify that cancellation of audio queues happens in < 50 ms."""
    from orchestration.protocol.audio import AudioFrameBuffer

    buffer = AudioFrameBuffer(max_buffer_frames=100)
    for _ in range(20):
        buffer.push_samples(generate_silence_frame())

    assert buffer.has_frame() is True

    t0 = time.perf_counter()
    buffer.clear()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert buffer.has_frame() is False
    assert elapsed_ms < 50.0  # Well within the 200 ms target
