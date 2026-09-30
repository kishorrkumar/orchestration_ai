"""
Test WebSocket session with custom system prompt and multi-turn context.
"""
import asyncio
import urllib.parse
import numpy as np
import websockets

from orchestration.protocol.messages import (
    MessageType,
    TextMessage,
    decode_message,
    encode_message,
)
from orchestration.protocol.audio import compute_rms

async def test_session():
    custom_prompt = (
        "You are Priya, a witty and warm voice assistant who loves telling clever, lighthearted jokes. "
        "Always speak naturally and conversationally. When asked for a joke, tell a funny, crisp one-sentence joke."
    )
    qs = urllib.parse.urlencode({
        "persona_id": "indian_priya",
        "neural_voice": "priya_colloquial",
        "accent": "Indian English",
        "character": "Warm",
        "text_prompt": custom_prompt,
    })
    url = f"ws://127.0.0.1:8000/v1/realtime?{qs}"
    print(f"Connecting to {url} ...")

    async with websockets.connect(url) as ws:
        print("[Connected]")
        
        # 1. Drain greeting
        t0 = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - t0 < 4.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.5)
            except asyncio.TimeoutError:
                break

        # 2. Turn 1: "Can you tell me about artificial intelligence?"
        print("\n>>> Sending: 'Can you tell me about artificial intelligence?'")
        await ws.send(encode_message(TextMessage(text="Can you tell me about artificial intelligence?")))

        tokens = []
        audio_frames = []
        t0 = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - t0 < 8.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
            except asyncio.TimeoutError:
                break
            msg = decode_message(raw)
            if msg.type == MessageType.TEXT:
                tokens.append(msg.text)
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                if compute_rms(samples) > 0.01:
                    audio_frames.append(samples)

        reply1 = "".join(tokens).strip()
        print(f"Turn 1 Reply: '{reply1}'")
        print(f"Turn 1 Audio Frames: {len(audio_frames)}")
        assert len(reply1) > 0, "Turn 1 reply was empty!"
        assert "ha an ji" not in reply1.lower(), f"Unexpected broken transliteration in '{reply1}'"
        assert len(audio_frames) > 0, "No audible audio received for Turn 1!"

        # 3. Turn 2: "Tell me a joke!"
        print("\n>>> Sending: 'Tell me a joke!'")
        await ws.send(encode_message(TextMessage(text="Tell me a joke!")))

        tokens2 = []
        audio_frames2 = []
        t0 = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - t0 < 8.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
            except asyncio.TimeoutError:
                break
            msg = decode_message(raw)
            if msg.type == MessageType.TEXT:
                tokens2.append(msg.text)
            elif msg.type == MessageType.AUDIO:
                samples = np.frombuffer(msg.data, dtype=np.float32)
                if compute_rms(samples) > 0.01:
                    audio_frames2.append(samples)

        reply2 = "".join(tokens2).strip()
        print(f"Turn 2 Reply (Joke): '{reply2}'")
        print(f"Turn 2 Audio Frames: {len(audio_frames2)}")
        assert len(reply2) > 0, "Turn 2 reply was empty!"
        assert len(audio_frames2) > 0, "No audible audio received for Turn 2!"
        print("\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    asyncio.run(test_session())
