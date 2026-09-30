import sys
import os
sys.path.insert(0, os.path.abspath("."))
import asyncio
import numpy as np
import logging
from websockets.asyncio.client import connect as ws_connect
from orchestration.worker.local_cascade import LocalCascadeWorkerServer
from orchestration.protocol.messages import (
    MessageType,
    AudioMessage,
    encode_message,
    decode_message,
)
from orchestration.protocol.audio import FRAME_SIZE

logging.basicConfig(level=logging.INFO)

async def test():
    server = LocalCascadeWorkerServer(host="127.0.0.1", port=9876)
    await server.start()

    url = "ws://127.0.0.1:9876/api/chat?neural_voice=priya_colloquial"
    print("Connecting client to", url)
    try:
        async with ws_connect(url) as ws:
            hs = await ws.recv()
            print("Received:", decode_message(hs).type)

            for i in range(15):
                frame = (0.05 * np.sin(2 * np.pi * 400 * np.linspace(0, 0.08, FRAME_SIZE))).astype(np.float32)
                await ws.send(encode_message(AudioMessage(data=frame.tobytes())))
                print(f"Sent frame {i+1}")
                # Also receive any audio or text from server
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.08)
                    print(f"Recv from server: {decode_message(msg).type}")
                except asyncio.TimeoutError:
                    pass
            print("Finished sending 15 frames successfully!")
    finally:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(test())
