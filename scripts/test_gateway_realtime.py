import sys
import os
sys.path.insert(0, os.path.abspath("."))
import asyncio
import numpy as np
import logging
from websockets.asyncio.client import connect as ws_connect
from orchestration.protocol.messages import (
    MessageType,
    AudioMessage,
    TextMessage,
    encode_message,
    decode_message,
)
from orchestration.protocol.audio import FRAME_SIZE

async def main():
    url = "ws://127.0.0.1:8000/v1/realtime?agent_id=priya_colloquial"
    print(f"Connecting to Gateway {url}...")
    async with ws_connect(url) as ws:
        # Handshake
        hs = await ws.recv()
        print("Connected! Initial server message:", decode_message(hs).type)

        # Receive initial greeting frames
        recvd = 0
        for _ in range(10):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                d = decode_message(msg)
                if d.type == MessageType.TEXT:
                    print(f"Agent Text Token: '{d.text}'")
                elif d.type == MessageType.AUDIO:
                    recvd += 1
            except asyncio.TimeoutError:
                pass
        print(f"Received {recvd} audio frames from initial greeting.")

        # Send 15 frames of audio simulating user speaking
        print("Sending 15 speech audio frames to gateway...")
        t = np.linspace(0, 0.08, FRAME_SIZE)
        sine_frame = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        for i in range(15):
            await ws.send(encode_message(AudioMessage(data=sine_frame.tobytes())))
            await asyncio.sleep(0.08)

        print("Sent 15 audio frames without disconnect! Voice pipeline working smoothly.")

if __name__ == "__main__":
    asyncio.run(main())
