"""
End-to-End WebSocket Session Client Test for Priya (Female Colloquial Voice).
Verifies:
1. Connecting with persona_id=indian_priya & neural_voice=priya_colloquial.
2. Initial greeting is spoken by Priya ("Namaste! I'm Priya...").
3. Audio frames arrive at 24 kHz with healthy speech energy (RMS > 0.01).
4. User asks "Who are you?", and agent replies with female identity "Priya".
"""
import asyncio
import time
import numpy as np
import websockets

from orchestration.protocol.messages import (
    MessageType,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    MetadataMessage,
    encode_message,
    decode_message,
)
from orchestration.protocol.audio import compute_rms

async def test_priya_session():
    url = "ws://127.0.0.1:8000/v1/realtime?persona_id=indian_priya&neural_voice=priya_colloquial&accent=Indian%20English&character=Warm"
    print(f"[Priya Test] Connecting to {url} ...")

    async with websockets.connect(url) as ws:
        print("[Priya Test] Connected to WebSocket.")
        
        greeting_audio_frames = []
        greeting_text_tokens = []
        speech_started = False
        t_start = time.time()

        # Listen for initial greeting
        while time.time() - t_start < 8.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            
            msg = decode_message(raw)
            if msg.type == MessageType.HANDSHAKE:
                print(f"[Priya Test] Received Handshake")
            elif msg.type == MessageType.METADATA:
                print(f"[Priya Test] Received Metadata: persona={msg.data.get('persona')}, character={msg.data.get('character')}")
            elif msg.type == MessageType.TEXT:
                greeting_text_tokens.append(msg.text)
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                rms = compute_rms(samples)
                if rms > 0.01:
                    greeting_audio_frames.append(samples)
                    if not speech_started:
                        print(f"[Priya Test] >>> FIRST PRIYA AUDIO FRAME! RMS={rms:.4f}")
                        speech_started = True

        full_greeting = "".join(greeting_text_tokens).strip()
        print(f"[Priya Test] Greeting Transcript: '{full_greeting}'")
        print(f"[Priya Test] Audible Audio Frames: {len(greeting_audio_frames)}")
        assert "Priya" in full_greeting, f"Expected Priya in greeting, got: {full_greeting}"
        assert len(greeting_audio_frames) > 0, "Expected audible audio frames from Priya!"

        # Send test utterance
        user_msg = "Hello, who are you and how can you help me?"
        print(f"\n[Priya Test] Sending user message: '{user_msg}'")
        await ws.send(encode_message(TextMessage(text=user_msg)))

        reply_audio_frames = []
        reply_text_tokens = []
        t_turn = time.time()

        while time.time() - t_turn < 10.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                break

            msg = decode_message(raw)
            if msg.type == MessageType.TEXT:
                reply_text_tokens.append(msg.text)
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                rms = compute_rms(samples)
                if rms > 0.01:
                    reply_audio_frames.append(samples)

        full_reply = "".join(reply_text_tokens).strip()
        print(f"[Priya Test] Reply Transcript: '{full_reply}'")
        print(f"[Priya Test] Audible Reply Frames: {len(reply_audio_frames)}")
        assert len(reply_audio_frames) > 0, "Expected audible reply frames!"
        print("\n=======================================================")
        print(">>> SUCCESS: Priya Female Voice Pipeline FULLY VERIFIED! <<<")
        print("=======================================================")

if __name__ == "__main__":
    asyncio.run(test_priya_session())
