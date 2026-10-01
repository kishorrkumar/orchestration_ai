#!/usr/bin/env python3
"""
Minimal Reference Python Client for Upstream NVIDIA PersonaPlex (Moshi) Server.
Demonstrates exact wire protocol:
- URL query parameters (text_prompt, voice_prompt, temperatures, topk)
- 0x00 Handshake wait (priming phase)
- Bidirectional Ogg Opus streaming via sphn (24 kHz, 80 ms = 1,920 sample frames)
- Audio accumulation and output WAV export
"""

import argparse
import asyncio
import os
import sys
import time
import urllib.parse

import numpy as np
import soundfile as sf
import websockets

try:
    import sphn
except ImportError:
    print("[ERROR] 'sphn' package is required. Install with: pip install 'sphn>=0.1.4,<0.2'")
    sys.exit(1)


async def run_min_client(args):
    # 1. Format URL strictly to upstream contract
    text_prompt = args.text_prompt.strip()
    if not text_prompt.startswith("<system>"):
        text_prompt = f"<system> {text_prompt} <system>"

    query_params = {
        "text_prompt": text_prompt,
        "voice_prompt": args.voice_prompt,
        "audio_temperature": "0.8",
        "text_temperature": "0.7",
        "audio_topk": "250",
        "text_topk": "25",
    }
    qs = urllib.parse.urlencode(query_params)
    ws_url = f"ws://{args.host}:{args.port}/api/chat?{qs}"

    print(f"[INFO] Connecting to: {ws_url}")
    t_start = time.time()

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=10) as ws:
        print("[INFO] WebSocket connection established. Waiting for server conditioning & handshake (0x00)...")

        # 2. Await Handshake (0x00) with generous timeout (model priming takes 4-7s on A100)
        handshake_raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
        t_handshake = time.time() - t_start
        if not isinstance(handshake_raw, bytes) or len(handshake_raw) == 0 or handshake_raw[0] != 0x00:
            raise RuntimeError(f"Expected handshake byte 0x00, got {handshake_raw!r}")

        print(f"[SUCCESS] Handshake received in {t_handshake:.2f}s! Moshi engine is ready.")

        # 3. Initialize fresh Opus Codecs for this session
        opus_writer = sphn.OpusStreamWriter(24000)
        opus_reader = sphn.OpusStreamReader(24000)

        # 4. Prepare input audio (or continuous 80ms silence frames)
        sr = 24000
        frame_size = 1920  # 80ms @ 24kHz

        if args.input_wav and os.path.exists(args.input_wav):
            data, in_sr = sf.read(args.input_wav, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if in_sr != sr:
                # Simple resample
                num_s = int(len(data) * sr / in_sr)
                data = np.interp(np.linspace(0, len(data), num_s, endpoint=False), np.arange(len(data)), data).astype(np.float32)
            print(f"[INFO] Loaded input audio: {len(data) / sr:.2f}s from {args.input_wav}")
        else:
            # Generate 4 seconds of speech-like cadence + silence
            dur = args.duration
            t = np.linspace(0, dur, int(dur * sr), endpoint=False)
            data = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            print(f"[INFO] Generated {dur}s synthetic test tone for streaming.")

        # Segment into 1,920 sample chunks
        chunks = [data[i : i + frame_size] for i in range(0, len(data), frame_size)]
        if len(chunks[-1]) < frame_size:
            chunks[-1] = np.pad(chunks[-1], (0, frame_size - len(chunks[-1])))

        received_pcm_blocks = []
        received_tokens = []
        stop_event = asyncio.Event()

        # Sender Task: Continuous 80ms cadence
        async def send_loop():
            print(f"[INFO] Streaming {len(chunks)} audio frames to worker at 80ms cadence...")
            for i, chunk in enumerate(chunks):
                if stop_event.is_set():
                    break
                t_frame_start = time.time()
                opus_writer.append_pcm(chunk)
                payload = opus_writer.read_bytes()
                if len(payload) > 0:
                    await ws.send(b"\x01" + payload)
                # Maintain strictly 80ms cadence
                elapsed = time.time() - t_frame_start
                await asyncio.sleep(max(0.0, 0.080 - elapsed))

            # Send 2 more seconds of silence to drive trailing response
            silence = np.zeros(frame_size, dtype=np.float32)
            for _ in range(25):
                if stop_event.is_set():
                    break
                opus_writer.append_pcm(silence)
                payload = opus_writer.read_bytes()
                if len(payload) > 0:
                    await ws.send(b"\x01" + payload)
                await asyncio.sleep(0.080)

            print("[INFO] Audio streaming finished. Waiting for final response tokens...")
            await asyncio.sleep(1.5)
            stop_event.set()

        # Receiver Task: decode 0x01 Opus and 0x02 Text tokens
        async def recv_loop():
            first_audio_logged = False
            first_text_logged = False
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    print("[INFO] Upstream closed connection cleanly (code 1000).")
                    break

                if not isinstance(raw, bytes) or len(raw) == 0:
                    continue

                kind = raw[0]
                payload = raw[1:]

                if kind == 0x01:
                    if not first_audio_logged:
                        first_audio_logged = True
                        print(f"[METRIC] Time-To-First-Audio: {time.time() - t_start:.2f}s")
                    opus_reader.append_bytes(payload)
                    pcm = opus_reader.read_pcm()
                    if len(pcm) > 0:
                        received_pcm_blocks.append(pcm)

                elif kind == 0x02:
                    token = payload.decode("utf-8", errors="replace")
                    if not first_text_logged:
                        first_text_logged = True
                        print(f"[METRIC] Time-To-First-Text: {time.time() - t_start:.2f}s")
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    received_tokens.append(token)

        await asyncio.gather(send_loop(), recv_loop())

    # 5. Export received audio to WAV file
    print("")
    if received_pcm_blocks:
        full_audio = np.concatenate(received_pcm_blocks)
        sf.write(args.output_wav, full_audio, sr)
        print(f"[SUCCESS] Exported {len(full_audio) / sr:.2f}s of agent audio to {args.output_wav}")
    else:
        print("[WARNING] Zero audio samples received from upstream.")


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex Minimal Reference Client")
    parser.add_argument("--host", default="127.0.0.1", help="Worker host")
    parser.add_argument("--port", type=int, default=8998, help="Worker port")
    parser.add_argument("--voice-prompt", default="NATM0.pt", help="Voice preset")
    parser.add_argument("--text-prompt", default="You enjoy having a good conversation.", help="Conditioning prompt")
    parser.add_argument("--input-wav", default=None, help="Input WAV to stream")
    parser.add_argument("--duration", type=float, default=4.0, help="Test tone duration in seconds if no WAV")
    parser.add_argument("--output-wav", default="reply.wav", help="Path to save returned agent speech")

    args = parser.parse_args()
    asyncio.run(run_min_client(args))


if __name__ == "__main__":
    main()
