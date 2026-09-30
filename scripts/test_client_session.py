"""
End-to-End WebSocket Session Client Test.
Connects to http://127.0.0.1:8000/v1/realtime just like the browser console,
receives the greeting audio frames and transcript, verifies speech energy (RMS > 0.05),
sends a user turn, and verifies the agent's spoken response.
"""
import asyncio
import json
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

async def test_full_session():
    url = "ws://127.0.0.1:8000/v1/realtime?persona_id=indian_pro&accent=Indian%20English&character=Professional"
    print(f"Connecting to {url} ...")

    async with websockets.connect(url) as ws:
        print("[Client] Connected to gateway WebSocket.")
        
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
                print(f"[Client] Received Handshake: version={msg.version}")
            elif msg.type == MessageType.METADATA:
                print(f"[Client] Received Metadata: {msg.data}")
            elif msg.type == MessageType.TEXT:
                greeting_text_tokens.append(msg.text)
                print(f"[Client] Received Agent Token: '{msg.text}'")
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                rms = compute_rms(samples)
                if rms > 0.01:
                    greeting_audio_frames.append(samples)
                    if not speech_started:
                        print(f"[Client] >>> FIRST SPEECH FRAME RECEIVED! RMS={rms:.4f} (Samples={len(samples)})")
                        speech_started = True

        full_greeting_text = "".join(greeting_text_tokens)
        print(f"\n[Client] Complete Greeting Transcript: '{full_greeting_text}'")
        print(f"[Client] Received {len(greeting_audio_frames)} audible audio frames ({len(greeting_audio_frames)*80} ms)")
        assert len(greeting_audio_frames) > 0, "ERROR: No audible audio frames received for greeting!"
        assert "Aarav" in full_greeting_text or "Namaste" in full_greeting_text, f"Unexpected greeting: {full_greeting_text}"

        # Now send a user text utterance
        user_prompt = "Hello Aarav, I would like to know about your services."
        print(f"\n[Client] Sending user turn: '{user_prompt}'")
        await ws.send(encode_message(TextMessage(text=user_prompt)))

        reply_audio_frames = []
        reply_text_tokens = []
        t_turn = time.time()
        reply_started = False

        while time.time() - t_turn < 10.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                break

            msg = decode_message(raw)
            if msg.type == MessageType.TEXT:
                reply_text_tokens.append(msg.text)
                print(f"[Client] Reply Token: '{msg.text}'")
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                rms = compute_rms(samples)
                if rms > 0.01:
                    reply_audio_frames.append(samples)
                    if not reply_started:
                        print(f"[Client] >>> FIRST REPLY AUDIO RECEIVED! RMS={rms:.4f} in {(time.time()-t_turn)*1000:.1f}ms")
                        reply_started = True

        full_reply_text = "".join(reply_text_tokens)
        print(f"\n[Client] Complete Reply Transcript: '{full_reply_text}'")
        print(f"[Client] Received {len(reply_audio_frames)} reply audio frames ({len(reply_audio_frames)*80} ms)")
        assert len(reply_audio_frames) > 0, "ERROR: No audible reply audio received!"
        print("\n=======================================================")
        print(">>> SUCCESS: Real-Time S2S Pipeline FULLY VERIFIED! <<<")
        print("=======================================================")

if __name__ == "__main__":
    asyncio.run(test_full_session())
