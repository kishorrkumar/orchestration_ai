#!/usr/bin/env python3
"""
Direct PersonaPlex Worker WebSocket Client Test.
Connects directly to ws://127.0.0.1:8998/api/chat (bypassing the gateway),
streams an input audio file (or generates synthetic speech / sine tone),
records the worker's audio response using sphn Ogg-Opus transcoding,
and prints priming time, TTFA, text tokens, audio duration, and RMS.
"""

import argparse
import asyncio
import math
import os
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import soundfile as sf
import sphn
import websockets

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # 80ms at 24kHz = 12.5 Hz
FRAME_DURATION_SEC = FRAME_SAMPLES / SAMPLE_RATE  # 0.08s


def compute_rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def create_or_load_wav(wav_path: str | None) -> np.ndarray:
    if wav_path and os.path.exists(wav_path):
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]  # mono
        if sr != SAMPLE_RATE:
            # Resample using sphn if needed
            pcm_in = data.astype(np.float32)[None, :]
            pcm_out = sphn.resample(pcm_in, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
            data = pcm_out[0]
        return data.astype(np.float32)

    # Fallback to test_fem.wav or test_male.wav if available
    for fallback in ["test_fem.wav", "test_male.wav", "test_joke_cats.wav"]:
        if os.path.exists(fallback):
            print(f"[INFO] Using existing file '{fallback}' as input audio.")
            data, sr = sf.read(fallback)
            if data.ndim > 1:
                data = data[:, 0]
            if sr != SAMPLE_RATE:
                pcm_in = data.astype(np.float32)[None, :]
                pcm_out = sphn.resample(pcm_in, src_sample_rate=sr, dst_sample_rate=SAMPLE_RATE)
                data = pcm_out[0]
            return data.astype(np.float32)

    # Synthesize a speech-like modulated tone
    print("[INFO] No input WAV specified or found. Synthesizing 2 seconds of test audio.")
    t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False, dtype=np.float32)
    # 440 Hz tone modulated at 4 Hz
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    return tone.astype(np.float32)


async def run_direct_test(
    url: str,
    wav_path: str | None,
    out_path: str,
    voice_prompt: str,
    text_prompt: str,
    silence_after_sec: float = 6.0,
):
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    input_audio = create_or_load_wav(wav_path)
    total_input_duration = len(input_audio) / SAMPLE_RATE
    print(f"[INFO] Loaded input audio: {len(input_audio)} samples ({total_input_duration:.2f}s)")

    # Prepare URL query parameters
    cleaned_prompt = text_prompt.strip()
    if not (cleaned_prompt.startswith("<system>") and cleaned_prompt.endswith("<system>")):
        cleaned_prompt = f"<system> {cleaned_prompt} <system>"

    query_params = {
        "text_prompt": cleaned_prompt,
        "voice_prompt": voice_prompt,
    }
    encoded_qs = urllib.parse.urlencode(query_params)
    full_url = f"{url}?{encoded_qs}"
    print(f"[INFO] Connecting to worker at: {url}")
    print(f"       Voice prompt: {voice_prompt}")
    print(f"       Text prompt:  {cleaned_prompt[:60]}...")

    opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
    opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    t_connect_start = time.perf_counter()
    received_audio_pcm = []
    received_text_tokens = []
    first_audio_time = None
    handshake_time = None

    try:
        async with websockets.connect(
            full_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print("[INFO] TCP/WebSocket connected. Awaiting priming and handshake byte (0x00)...")

            # Wait for handshake (0x00)
            first_msg = await ws.recv()
            handshake_time = time.perf_counter()
            priming_duration_ms = (handshake_time - t_connect_start) * 1000.0

            if not isinstance(first_msg, bytes) or len(first_msg) == 0 or first_msg[0] != 0x00:
                print(f"[ERROR] Expected handshake 0x00, got: {first_msg!r}")
                return

            print(f"[SUCCESS] Handshake 0x00 received in {priming_duration_ms:.1f}ms (~{priming_duration_ms/1000:.2f}s)!")

            stop_event = asyncio.Event()

            async def send_audio_loop():
                nonlocal first_audio_time
                offset = 0
                silence_frames = int(silence_after_sec / FRAME_DURATION_SEC)
                frame_count = 0

                # 1. Stream input WAV in exact 1920-sample chunks
                while offset < len(input_audio) and not stop_event.is_set():
                    chunk = input_audio[offset : offset + FRAME_SAMPLES]
                    offset += FRAME_SAMPLES
                    if len(chunk) < FRAME_SAMPLES:
                        chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))

                    opus_writer.append_pcm(chunk)
                    payload = opus_writer.read_bytes()
                    if payload:
                        await ws.send(b"\x01" + payload)

                    frame_count += 1
                    await asyncio.sleep(FRAME_DURATION_SEC)

                print(f"[INFO] Finished streaming input WAV ({frame_count} frames). Now streaming silence...")

                # 2. Stream silence so model can generate its turn
                for _ in range(silence_frames):
                    if stop_event.is_set():
                        break
                    silent_chunk = np.zeros(FRAME_SAMPLES, dtype=np.float32)
                    opus_writer.append_pcm(silent_chunk)
                    payload = opus_writer.read_bytes()
                    if payload:
                        await ws.send(b"\x01" + payload)
                    await asyncio.sleep(FRAME_DURATION_SEC)

                print("[INFO] Audio send loop completed.")
                stop_event.set()

            async def recv_loop():
                nonlocal first_audio_time
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if stop_event.is_set():
                            break
                        continue

                    if not isinstance(raw, bytes) or len(raw) == 0:
                        continue

                    kind = raw[0]
                    payload = raw[1:]

                    if kind == 0x01:  # Audio
                        opus_reader.append_bytes(payload)
                        pcm = opus_reader.read_pcm()
                        if len(pcm) > 0:
                            if first_audio_time is None:
                                first_audio_time = time.perf_counter()
                            received_audio_pcm.append(pcm)
                    elif kind == 0x02:  # Text
                        token_str = payload.decode("utf-8", errors="replace")
                        received_text_tokens.append(token_str)
                        sys.stdout.write(token_str)
                        sys.stdout.flush()

            sender_task = asyncio.create_task(send_audio_loop())
            receiver_task = asyncio.create_task(recv_loop())

            await asyncio.gather(sender_task, receiver_task)

    except Exception as e:
        print(f"[ERROR] Worker direct test encountered error: {e}")
        raise e

    print("\n" + "=" * 60)
    print(" DIRECT WORKER TEST RESULTS")
    print("=" * 60)

    # Aggregate audio
    if received_audio_pcm:
        all_audio = np.concatenate(received_audio_pcm)
        duration_sec = len(all_audio) / SAMPLE_RATE
        rms = compute_rms(all_audio)
        sf.write(out_path, all_audio, SAMPLE_RATE)
        print(f"Output File:     {out_path}")
        print(f"Output Duration: {duration_sec:.2f} s ({len(all_audio)} samples @ {SAMPLE_RATE} Hz)")
        print(f"Audio RMS:       {rms:.5f}")
        print(f"Audible:         {'YES' if rms > 0.005 else 'NO (silent)'}")
    else:
        print("Output Audio:    NONE RECEIVED (0 frames)")
        rms = 0.0
        duration_sec = 0.0

    if handshake_time and t_connect_start:
        priming_ms = (handshake_time - t_connect_start) * 1000.0
        print(f"Priming Time:    {priming_ms:.1f} ms")
    if first_audio_time and handshake_time:
        ttfa_ms = (first_audio_time - handshake_time) * 1000.0
        print(f"TTFA (First Audio): {ttfa_ms:.1f} ms from handshake")

    full_text = "".join(received_text_tokens)
    print(f"Total Tokens:    {len(received_text_tokens)}")
    print(f"Full Text:       {full_text.strip()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test upstream PersonaPlex worker directly")
    parser.add_argument("--url", default="ws://127.0.0.1:8998/api/chat", help="Worker WebSocket URL")
    parser.add_argument("--wav", default="test_fem.wav", help="Input audio WAV file")
    parser.add_argument("--out", default="eval_out/direct_reply.wav", help="Output WAV path")
    parser.add_argument("--voice", default="NATM1.pt", help="Voice prompt filename (.pt or .wav)")
    parser.add_argument("--prompt", default="You are a helpful and concise voice assistant.", help="Text prompt")
    parser.add_argument("--silence-sec", type=float, default=6.0, help="Seconds of silence to stream after WAV")
    args = parser.parse_args()

    asyncio.run(
        run_direct_test(
            url=args.url,
            wav_path=args.wav,
            out_path=args.out,
            voice_prompt=args.voice,
            text_prompt=args.prompt,
            silence_after_sec=args.silence_sec,
        )
    )


if __name__ == "__main__":
    main()
