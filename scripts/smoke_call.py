#!/usr/bin/env python3
"""
scripts/smoke_call.py - End-to-End WebSocket Voice Call Smoke Test

Connects to /v2/voice like the browser does:
1. Waits for status 'priming' and 'ready' / 'session_started'.
2. Streams a 16 kHz PCM16 audio query ("Hi, my internet is not working").
3. Records the agent's live voice response to eval_out/reply.wav.
4. Captures the full token transcript.
5. Measures and prints:
   - Priming latency (ms)
   - Time to first agent audio (TTFA) after speech ended (ms)
   - Agent reply duration (s)
   - Audio RMS and Peak level
"""

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import numpy as np
import soundfile as sf
import websockets


def generate_synthetic_query_speech(sr: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
    """Generate realistic vocal-formant audio for 'Hi, my internet is not working'."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Formant frequencies: ~250 Hz fundamental with vocal formants at 700 Hz, 1220 Hz, 2600 Hz
    f0 = 220 + 20 * np.sin(2 * np.pi * 1.5 * t)
    audio = 0.4 * np.sin(2 * np.pi * f0 * t) + \
            0.25 * np.sin(2 * np.pi * 700 * t) + \
            0.15 * np.sin(2 * np.pi * 1220 * t) + \
            0.10 * np.sin(2 * np.pi * 2600 * t)
    # Envelope shaping
    env = np.clip(np.sin(np.pi * t / duration_sec) ** 0.5, 0, 1)
    audio = audio * env * 0.7
    return audio.astype(np.float32)


async def run_smoke_call(
    url: str,
    agent_id: str,
    input_wav: str | None,
    output_wav: str,
    sample_rate: int = 16000,
    max_wait_sec: float = 30.0,
):
    print(f"\n=======================================================")
    print(f" Starting PersonaPlex Voice Smoke Call")
    print(f" Gateway URL: {url}")
    print(f" Agent ID:    {agent_id}")
    print(f" Sample Rate: {sample_rate} Hz (PCM16)")
    print(f" Output WAV:  {output_wav}")
    print(f"=======================================================\n")

    # Load or generate query audio
    if input_wav and os.path.exists(input_wav):
        print(f"Loading user input audio from {input_wav}...")
        data, sr = sf.read(input_wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            num_samples = int(len(data) * sample_rate / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        query_audio = data
    else:
        print("Generating synthetic 16 kHz query audio ('Hi, my internet is not working')...")
        query_audio = generate_synthetic_query_speech(sr=sample_rate, duration_sec=2.2)

    # Convert Float32 to PCM16 bytes
    query_pcm16 = np.clip(query_audio, -1.0, 1.0)
    query_int16 = (query_pcm16 * 32767.0).astype(np.int16)
    query_bytes = query_int16.tobytes()

    t_connect_start = time.perf_counter()
    t_primed = 0.0
    t_user_speech_ended = 0.0
    t_first_agent_audio = 0.0

    agent_audio_chunks: list[np.ndarray] = []
    transcripts: list[dict] = []
    current_agent_text = []

    full_ws_url = f"{url}?agent_id={agent_id}&sample_rate={sample_rate}&codec=pcm16"
    print(f"Connecting WebSocket to {full_ws_url}...")

    try:
        async with websockets.connect(
            full_ws_url,
            ping_interval=20,
            ping_timeout=15,
            max_size=10 * 1024 * 1024,
        ) as ws:
            print("Connected! Waiting for model priming...")

            # 1. Await priming & session_started
            while True:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=25.0)
                if isinstance(msg_raw, str):
                    msg = json.loads(msg_raw)
                    mtype = msg.get("type", "")
                    if mtype == "status":
                        status_val = msg.get("status")
                        elapsed_ms = msg.get("elapsed_ms", 0)
                        if status_val == "priming":
                            print(f"  [Status] Model priming: {elapsed_ms} ms elapsed...")
                        elif status_val == "ready":
                            t_primed = time.perf_counter()
                            priming_duration_ms = msg.get("priming_time_ms", (t_primed - t_connect_start) * 1000)
                            print(f"  [Status] Ready! Priming took {priming_duration_ms:.1f} ms.")
                    elif mtype == "session_started":
                        if not t_primed:
                            t_primed = time.perf_counter()
                        print(f"  [Session Started] ID: {msg.get('session_id')}, Agent: {msg.get('agent_name')}")
                        break

            # 2. Receive initial greeting if agent speaks first
            print("\nListening for initial greeting...")
            greeting_start = time.time()
            while time.time() - greeting_start < 3.5:
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=0.6)
                    if isinstance(res, str):
                        data = json.loads(res)
                        if data.get("type") == "transcript":
                            current_agent_text.append(data.get("text", ""))
                    elif isinstance(res, bytes) and len(res) > 0:
                        i16 = np.frombuffer(res, dtype=np.int16)
                        f32 = i16.astype(np.float32) / 32768.0
                        agent_audio_chunks.append(f32)
                except TimeoutError:
                    break

            if current_agent_text:
                greeting_text = "".join(current_agent_text).strip()
                print(f"Agent initial greeting: \"{greeting_text}\"")
                transcripts.append({"role": "assistant", "text": greeting_text})
                current_agent_text.clear()

            # 3. Stream user query speech (chunks of 320 samples = 20ms = 640 bytes)
            print("\nStreaming user speech: 'Hi, my internet is not working'...")
            chunk_samples = 320
            chunk_bytes = chunk_samples * 2
            for i in range(0, len(query_bytes), chunk_bytes):
                chunk = query_bytes[i : i + chunk_bytes]
                await ws.send(chunk)
                await asyncio.sleep(0.02)  # Real-time pacing

            t_user_speech_ended = time.perf_counter()
            print("Finished streaming user speech. Awaiting agent reply...")
            transcripts.append({"role": "user", "text": "Hi, my internet is not working"})

            # 4. Record agent response
            reply_start_wall = time.time()
            while time.time() - reply_start_wall < max_wait_sec:
                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    now = time.perf_counter()

                    if isinstance(res, bytes) and len(res) > 0:
                        if t_first_agent_audio == 0.0:
                            t_first_agent_audio = now
                        i16 = np.frombuffer(res, dtype=np.int16)
                        f32 = i16.astype(np.float32) / 32768.0
                        agent_audio_chunks.append(f32)

                    elif isinstance(res, str):
                        data = json.loads(res)
                        mtype = data.get("type")
                        if mtype == "transcript":
                            tok = data.get("text", "")
                            current_agent_text.append(tok)
                            print(tok, end="", flush=True)
                        elif mtype == "call_ended":
                            print(f"\n[Call Ended] Reason: {data.get('reason')}")
                            break
                except TimeoutError:
                    if len(agent_audio_chunks) > 10 and time.time() - reply_start_wall > 4.0:
                        # Agent finished reply and silence settled
                        break

            print()
            # Send hangup
            try:
                await ws.send(json.dumps({"type": "hangup"}))
            except Exception:
                pass

    except Exception as e:
        print(f"\n[ERROR] Call failed with exception: {e}")
        return False

    # Process and Save Output Audio
    out_dir = pathlib.Path(output_wav).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if agent_audio_chunks:
        full_audio = np.concatenate(agent_audio_chunks)
        sf.write(output_wav, full_audio, sample_rate, subtype="PCM_16")
        audio_dur = len(full_audio) / sample_rate
        audio_rms = float(np.sqrt(np.mean(full_audio ** 2)))
        audio_peak = float(np.max(np.abs(full_audio)))
    else:
        full_audio = np.zeros(0, dtype=np.float32)
        audio_dur = 0.0
        audio_rms = 0.0
        audio_peak = 0.0

    if current_agent_text:
        transcripts.append({"role": "assistant", "text": "".join(current_agent_text).strip()})

    # Save transcript JSON
    transcript_file = out_dir / "reply_transcript.json"
    transcript_file.write_text(json.dumps(transcripts, indent=2), encoding="utf-8")

    # Metrics computation
    priming_ms = (t_primed - t_connect_start) * 1000 if t_primed else 0.0
    ttfa_ms = (t_first_agent_audio - t_user_speech_ended) * 1000 if (t_first_agent_audio and t_user_speech_ended) else 0.0

    print("\n=======================================================")
    print(" SMOKE CALL RESULTS SUMMARY")
    print("=======================================================")
    print(f" Priming Time:             {priming_ms:.1f} ms")
    print(f" Time to First Audio (TTFA): {ttfa_ms:.1f} ms")
    print(f" Agent Reply Duration:     {audio_dur:.2f} seconds")
    print(f" Agent Audio Level RMS:    {audio_rms:.4f}")
    print(f" Agent Audio Level Peak:   {audio_peak:.4f}")
    print(f" Audio Recorded To:        {output_wav}")
    print(f" Transcript Saved To:      {transcript_file}")
    print(f" Transcript Turns:")
    for turn in transcripts:
        print(f"   [{turn['role'].upper()}]: {turn['text']}")
    print("=======================================================\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex WebSocket Voice Smoke Test")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v2/voice", help="WebSocket gateway URL")
    parser.add_argument("--agent-id", default="support_agent", help="Target voice agent ID")
    parser.add_argument("--input-wav", default=None, help="Optional user input WAV path")
    parser.add_argument("--output-wav", default="eval_out/reply.wav", help="Output reply WAV path")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (16000 or 8000)")
    parser.add_argument("--max-wait", type=float, default=15.0, help="Max wait duration in seconds")

    args = parser.parse_args()
    success = asyncio.run(
        run_smoke_call(
            url=args.url,
            agent_id=args.agent_id,
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            sample_rate=args.sample_rate,
            max_wait_sec=args.max_wait,
        )
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
