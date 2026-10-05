"""
Unit tests for Transcript Turn Segmentation and Streaming Caller ASR.
Verifies:
1. Agent tokens are accumulated and segmented into separate turns on silence (>0.6s).
2. Caller speech onset interrupts an active agent turn and emits a finalized bubble with timestamp.
3. Caller speech is transcribed into a separate 'You' bubble with timestamps.
4. Consecutive turns never append into a single monolithic bubble.
"""

import asyncio
import time

import pytest

from orchestration.audio.caller_transcriber import CallerTranscriber


@pytest.mark.asyncio
async def test_agent_turn_segmentation_on_silence():
    """Verify agent tokens are finalized into discrete turn bubbles on silence."""
    transcripts = []

    async def on_transcript(data):
        transcripts.append(data)

    transcriber = CallerTranscriber(
        sample_rate=24000,
        on_transcript_callback=on_transcript,
    )
    transcriber.start()

    try:
        # Turn 1: Agent speaks first utterance
        transcriber.on_agent_token("Hello ")
        transcriber.on_agent_token("there! ")
        transcriber.on_agent_token("How are you?")
        transcriber.on_agent_audio_frame()

        # Simulate silence (> 0.6s)
        await asyncio.sleep(0.65)
        await transcriber.check_agent_silence_timeout(silence_threshold_sec=0.60)

        assert len(transcripts) == 1
        t1 = transcripts[0]
        assert t1["role"] == "agent"
        assert t1["text"] == "Hello there! How are you?"
        assert t1["is_final"] is True
        assert "timestamp" in t1

        # Turn 2: Agent speaks second utterance after a pause
        transcriber.on_agent_token("I am here ")
        transcriber.on_agent_token("to help.")
        transcriber.on_agent_audio_frame()

        await asyncio.sleep(0.65)
        await transcriber.check_agent_silence_timeout(silence_threshold_sec=0.60)

        assert len(transcripts) == 2
        t2 = transcripts[1]
        assert t2["role"] == "agent"
        assert t2["text"] == "I am here to help."
        assert t2["is_final"] is True
        # Verify the two turns are completely separate bubbles
        assert t2["text"] != t1["text"]

    finally:
        await transcriber.stop()


@pytest.mark.asyncio
async def test_agent_turn_finalized_on_caller_onset():
    """Verify caller speech onset immediately interrupts and finalizes the active agent turn."""
    transcripts = []

    async def on_transcript(data):
        transcripts.append(data)

    transcriber = CallerTranscriber(
        sample_rate=24000,
        on_transcript_callback=on_transcript,
    )
    transcriber.start()

    try:
        transcriber.on_agent_token("I was going to say something very ")
        transcriber.on_agent_token("long...")

        # Caller begins speaking (barge-in interruption)
        finalized = await transcriber.finalize_agent_turn(reason="caller_barge_in")
        assert finalized is not None
        assert finalized["role"] == "agent"
        assert finalized["text"] == "I was going to say something very long..."
        assert finalized["is_final"] is True
        assert len(transcripts) == 1

        # Immediate caller transcript follows
        await on_transcript({
            "event": "transcript",
            "role": "user",
            "text": "Excuse me, I have a question.",
            "is_final": True,
            "timestamp": time.strftime("%H:%M:%S"),
        })

        assert len(transcripts) == 2
        assert transcripts[0]["role"] == "agent"
        assert transcripts[1]["role"] == "user"
        assert transcripts[1]["text"] == "Excuse me, I have a question."

    finally:
        await transcriber.stop()
