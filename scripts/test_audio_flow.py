import asyncio
import numpy as np
from websockets.asyncio.client import connect as ws_connect
from orchestration.protocol.messages import (
    MessageType,
    HandshakeMessage,
    AudioMessage,
    encode_message,
    decode_message,
)
from orchestration.protocol.audio import FRAME_SIZE

async def main():
    url = "ws://127.0.0.1:8998/api/chat?neural_voice=priya_colloquial"
    print(f"Connecting to {url}...")
    async with ws_connect(url) as ws:
        handshake = await ws.recv()
        msg = decode_message(handshake)
        print("Handshake received:", msg.type)

        # Send 10 audio frames
        for i in range(10):
            frame = np.zeros(FRAME_SIZE, dtype=np.float32)
            # Add some speech-like sine wave
            t = np.linspace(0, 0.08, FRAME_SIZE, endpoint=False)
            frame += (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
            await ws.send(encode_message(AudioMessage(data=frame.tobytes())))
            print(f"Sent audio frame {i+1}")
            await asyncio.sleep(0.08)

        # Listen for 3 seconds
        t_end = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < t_end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                m = decode_message(raw)
                print(f"Received msg type: {m.type}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print("Recv error:", e)
                break

if __name__ == "__main__":
    asyncio.run(main())
