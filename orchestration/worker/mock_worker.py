"""
High-Fidelity Mock PersonaPlex Server for local testing, CI, and laptop development.

Emulates:
- The exact PersonaPlex / Moshi server WebSocket endpoint: `GET /api/chat`
- Handshake protocol: sends `0x00` after prompt initialization
- Full-duplex audio stream: 12.5 Hz (80ms per frame, 1,920 samples @ 24kHz)
- Text token generation matching SentencePiece token stream
- Single-concurrency mutual exclusion lock matching upstream `server.py`
- Offline local text-to-speech synthesis (pyttsx3) for natural spoken audio
- Dynamic persona-based conversational turn-taking and barge-in handling
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import math
import os
import tempfile
import time
import urllib.parse
from typing import Optional, List

import numpy as np
import soundfile as sf
import websockets
from websockets.asyncio.server import Server, serve, ServerConnection

from ..protocol.messages import (
    MessageType,
    HandshakeMessage,
    AudioMessage,
    TextMessage,
    ErrorMessage,
    encode_message,
    decode_message,
)
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    compute_rms,
    generate_silence_frame,
)

logger = logging.getLogger("orchestration.worker.mock")

# Thread pool for offline TTS synthesis
_TTS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _synthesize_speech_offline(text: str, target_sr: int = 24000) -> Optional[np.ndarray]:
    """
    Synthesize speech offline using native Windows/system TTS (pyttsx3)
    and resample to 24,000 Hz float32 mono PCM.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Set speech rate and volume
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 0.95)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        # Read generated audio
        data, sr = sf.read(tmp_path, dtype="float32")
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to target_sr (24,000 Hz)
        if sr != target_sr and len(data) > 0:
            num_samples = int(len(data) * target_sr / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        return data.astype(np.float32)
    except Exception as e:
        logger.warning(f"Offline TTS synthesis unavailable: {e}. Falling back to carrier wave.")
        return None
    finally:
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass


def generate_persona_response(user_text: str, persona_prompt: str) -> str:
    """Generate intelligent conversational responses in character for the active persona."""
    u = user_text.lower().strip()
    p = persona_prompt.lower()

    # 1. Mars Astronaut Alex
    if "mars" in p or "reactor" in p or "astronaut" in p:
        if any(w in u for w in ["coolant", "stabilize", "pump", "loop", "fix", "temperature"]):
            return "Good thinking! I'm rerouting secondary coolant to loop B right now. Core temperature is dropping back under critical. We bought some time!"
        elif any(w in u for w in ["status", "how", "report", "system", "damage"]):
            return "Core temperature is holding at 820 Kelvin. We lost sensor manifold three, and emergency batteries are at fifty percent. We need to reset the magnetic containment."
        elif any(w in u for w in ["hello", "hi", "hear", "alex", "mission"]):
            return "Mission Control, thank goodness you can hear me! The reactor core on deck four has an active thermal runaway. Please advise immediately!"
        else:
            return f"Understood regarding {user_text.rstrip('?.')}. The magnetic containment field is still fluctuating. What procedure should we initiate next?"

    # 2. CitySan Services (Ayelen Lucero)
    elif "citysan" in p or "waste" in p:
        if any(w in u for w in ["schedule", "pickup", "when", "day", "collection"]):
            return "Your regular pickup schedule is every other week. Your next collection is scheduled for Friday, April 12th."
        elif any(w in u for w in ["compost", "bin", "green", "cost", "price", "add"]):
            return "Yes! We provide green compost bins for an additional eight dollars per month. Would you like me to activate that on your account?"
        elif any(w in u for w in ["name", "torres", "omar", "verify", "account"]):
            return "I have verified your account under Omar Torres at CitySan Services. How else can I help with your service today?"
        elif any(w in u for w in ["hello", "hi", "hey"]):
            return "Hello! Thank you for calling CitySan Services. My name is Ayelen Lucero. How can I help you today?"
        else:
            return f"Regarding your question about {user_text.rstrip('?.')}, I can update your schedule or add bin services. Would you like me to do that?"

    # 3. Jerusalem Shakshuka (Owen Foster)
    elif "shakshuka" in p or "restaurant" in p:
        if any(w in u for w in ["menu", "what", "options", "price", "cost"]):
            return "We serve our Classic Shakshuka with poached eggs for nine fifty, and Spicy with jalapenos for ten twenty-five. Sides include warm pita for two fifty and Israeli salad for three dollars."
        elif any(w in u for w in ["order", "classic", "spicy", "buy", "want"]):
            return "Great order! I will have that made fresh in the kitchen right now. Would you like warm pita or Israeli salad with that?"
        elif any(w in u for w in ["hours", "open", "drive", "time", "close"]):
            return "Our drive-through is open every day until nine PM. No combo offers, but everything is cooked fresh to order!"
        elif any(w in u for w in ["hello", "hi", "hey"]):
            return "Welcome to Jerusalem Shakshuka! I'm Owen Foster. Can I get a Classic or Spicy shakshuka started for you?"
        else:
            return f"Sounds good! We can definitely prepare {user_text.rstrip('?.')} for you fresh today. Anything else I can add?"

    # 4. AeroRentals Pro (Tomaz Novak)
    elif "aerorentals" in p or "drone" in p:
        if any(w in u for w in ["price", "cost", "rate", "how much"]):
            return "The PhoenixDrone X is sixty-five dollars for four hours or one ten for eight hours. The premium SpectraDrone 9 is ninety-five for four hours."
        elif any(w in u for w in ["deposit", "require", "security"]):
            return "We require a refundable deposit of one hundred fifty dollars for standard models, or three hundred dollars for premium drones."
        elif any(w in u for w in ["hello", "hi"]):
            return "Hello! AeroRentals Pro, Tomaz Novak speaking. Are you looking to rent the PhoenixDrone X or SpectraDrone 9 today?"
        else:
            return f"The {user_text.rstrip('?.')} is available for rental today. We have fully charged battery packs ready to go."

    # 5. Sophia the Teacher (Assistant)
    else:
        if any(w in u for w in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I am Sophia, your teacher. What fascinating question or topic would you like to explore together today?"
        elif any(w in u for w in ["how are you", "doing"]):
            return "I am doing wonderfully, thank you! I love discussing ideas and answering questions. What is on your mind?"
        elif any(w in u for w in ["who are you", "what are you", "your name"]):
            return "I am Sophia, a friendly teacher and voice assistant. I am here to explain concepts clearly and help you learn."
        elif any(w in u for w in ["why", "how", "what", "explain", "tell me"]):
            return f"That is an insightful question about {user_text.rstrip('?.')}! In simple terms, it works through clear principles. Let me explain the key parts."
        else:
            return f"I understand your thoughts on {user_text.rstrip('?.')}. Let's dive deeper into that. What aspect would you like to explore first?"


class PersonaPlexMockServer:
    """
    Mock PersonaPlex server that mimics NVIDIA's server.py WebSocket behavior
    with offline speech synthesis and dynamic persona replies.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8998,
        prompt_init_delay: float = 0.05,
        frame_interval_sec: float = 0.08,  # 80ms = 12.5 Hz
    ):
        self.host = host
        self.port = port
        self.prompt_init_delay = prompt_init_delay
        self.frame_interval = frame_interval_sec

        self._server: Optional[Server] = None
        self._lock = asyncio.Lock()
        self._is_running = False

        self.active_connections = 0
        self.total_sessions_served = 0
        self.total_frames_received = 0
        self.total_frames_generated = 0

    async def start(self) -> None:
        self._is_running = True
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )
        logger.info(f"PersonaPlexMockServer started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("PersonaPlexMockServer stopped")

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        path = websocket.request.path if hasattr(websocket, "request") else ""
        self.active_connections += 1

        if self._lock.locked():
            logger.warning("Mock worker is busy. Rejecting concurrent session.")
            err_msg = encode_message(ErrorMessage(error="Worker busy: another session is currently active."))
            await websocket.send(err_msg)
            await websocket.close(code=1008, reason="Worker busy")
            self.active_connections -= 1
            return

        async with self._lock:
            self.total_sessions_served += 1
            # Parse query params
            parsed = urllib.parse.urlparse(path)
            query = urllib.parse.parse_qs(parsed.query)
            text_prompt = query.get("text_prompt", [""])[0]
            voice_prompt = query.get("voice_prompt", ["NATF2.pt"])[0]

            logger.info(f"Mock server accepted connection for prompt: {text_prompt[:50]}...")

            if self.prompt_init_delay > 0:
                await asyncio.sleep(self.prompt_init_delay)

            # Send handshake 0x00
            await websocket.send(encode_message(HandshakeMessage(version=0, model=0)))

            stop_event = asyncio.Event()
            user_speaking = False
            last_speech_time = 0.0
            audio_phase = 0.0

            # Outbound speech queue
            outbound_audio_frames: List[np.ndarray] = []
            outbound_tokens: List[str] = []

            loop = asyncio.get_running_loop()

            async def queue_agent_utterance(reply_text: str):
                nonlocal outbound_audio_frames, outbound_tokens
                # Tokenize into words
                words = reply_text.split(" ")
                tokens = [" " + w if i > 0 else w for i, w in enumerate(words)]
                outbound_tokens = tokens

                # Synthesize audio in background thread
                audio_samples = await loop.run_in_executor(
                    _TTS_EXECUTOR,
                    _synthesize_speech_offline,
                    reply_text,
                    SAMPLE_RATE,
                )

                if audio_samples is not None and len(audio_samples) > 0:
                    num_frames = int(math.ceil(len(audio_samples) / FRAME_SIZE))
                    padded = np.pad(audio_samples, (0, num_frames * FRAME_SIZE - len(audio_samples)))
                    frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]
                    outbound_audio_frames = frames
                else:
                    # Synthetic wave fallback
                    t = np.linspace(0, len(tokens) * 0.25, len(tokens) * int(0.25 * SAMPLE_RATE))
                    carrier = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
                    num_frames = int(math.ceil(len(carrier) / FRAME_SIZE))
                    padded = np.pad(carrier, (0, num_frames * FRAME_SIZE - len(carrier)))
                    outbound_audio_frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]

            # Queue initial persona greeting in background
            initial_greeting = generate_persona_response("hello", text_prompt)
            synth_task = asyncio.create_task(queue_agent_utterance(initial_greeting))

            async def receiver():
                nonlocal user_speaking, last_speech_time, outbound_audio_frames, outbound_tokens
                speech_detected_accum = 0

                try:
                    async for raw in websocket:
                        if not isinstance(raw, bytes):
                            continue
                        msg = decode_message(raw)

                        if msg.type == MessageType.TEXT:
                            logger.info(f"Mock received user text: '{msg.text}'")
                            # Barge-in: flush existing speech and formulate new answer
                            outbound_audio_frames.clear()
                            outbound_tokens.clear()
                            await websocket.send(encode_message(TextMessage(text=" ")))

                            # Generate intelligent persona response
                            reply = generate_persona_response(msg.text, text_prompt)
                            logger.info(f"Mock generating persona reply: '{reply}'")
                            await queue_agent_utterance(reply)

                        elif msg.type == MessageType.AUDIO:
                            self.total_frames_received += 1
                            audio_samples = (
                                np.frombuffer(msg.data, dtype=np.float32)
                                if len(msg.data) >= FRAME_SIZE * 4
                                else np.zeros(FRAME_SIZE, dtype=np.float32)
                            )
                            rms = compute_rms(audio_samples)

                            if rms > 0.025:
                                user_speaking = True
                                last_speech_time = time.time()
                                speech_detected_accum += 1
                                # If agent was speaking, interrupt immediately (barge-in)
                                if len(outbound_audio_frames) > 0:
                                    outbound_audio_frames.clear()
                                    outbound_tokens.clear()
                            elif time.time() - last_speech_time > 0.5:
                                if user_speaking and speech_detected_accum > 10:
                                    # User finished speaking without text recognition, generate follow-up
                                    user_speaking = False
                                    speech_detected_accum = 0
                                    reply = generate_persona_response("I hear you speaking", text_prompt)
                                    await queue_agent_utterance(reply)
                                user_speaking = False

                except websockets.ConnectionClosed:
                    pass
                finally:
                    stop_event.set()

            async def generator():
                nonlocal audio_phase
                token_step_interval = 3  # Emit a token every ~240ms
                step = 0

                while not stop_event.is_set():
                    t0 = time.time()
                    step += 1

                    if user_speaking:
                        # User speaking: backchannel or listen silently
                        out_frame = generate_silence_frame()
                        if step % 25 == 0:
                            await websocket.send(encode_message(TextMessage(text=" [backchannel: mm-hmm]")))
                    elif len(outbound_audio_frames) > 0:
                        out_frame = outbound_audio_frames.pop(0)
                        if step % token_step_interval == 0 and len(outbound_tokens) > 0:
                            token = outbound_tokens.pop(0)
                            await websocket.send(encode_message(TextMessage(text=token)))
                    else:
                        t = np.linspace(audio_phase, audio_phase + self.frame_interval, FRAME_SIZE, endpoint=False)
                        audio_phase += self.frame_interval
                        out_frame = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

                    await websocket.send(encode_message(AudioMessage(data=out_frame.tobytes())))
                    self.total_frames_generated += 1

                    elapsed = time.time() - t0
                    sleep_time = max(0.001, self.frame_interval - elapsed)
                    await asyncio.sleep(sleep_time)

            recv_task = asyncio.create_task(receiver())
            gen_task = asyncio.create_task(generator())

            done, pending = await asyncio.wait([recv_task, gen_task], return_when=asyncio.FIRST_COMPLETED)

            synth_task.cancel()
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.active_connections -= 1
