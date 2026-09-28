"""
High-Fidelity Mock PersonaPlex Server with Adaptive Noise Cancellation,
Intelligent Conversational Dialogue, and Full-Duplex Turn-Taking.
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
from typing import Optional, List, Set

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
    AdaptiveNoiseCanceller,
)

from ..rag.engine import default_rag_engine

logger = logging.getLogger("orchestration.worker.mock")

# Thread pool for offline TTS synthesis
_TTS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _synthesize_speech_offline(
    text: str, target_sr: int = 24000, voice_preset: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Synthesize speech offline using native system TTS (pyttsx3)
    and resample to 24,000 Hz float32 mono PCM.
    Selects male vs female voice based on PersonaPlex voice preset.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)  # Crisper, conversational pace
        engine.setProperty("volume", 0.95)

        if voice_preset:
            vp = voice_preset.upper()
            voices = engine.getProperty("voices")
            selected_voice = None
            if any(k in vp for k in ["NATF", "VARF", "FEMALE", "SOPHIA", "AYELEN"]):
                for v in voices:
                    name_low = v.name.lower()
                    if "zira" in name_low or "female" in getattr(v, "gender", "").lower() or "eva" in name_low:
                        selected_voice = v.id
                        break
            elif any(k in vp for k in ["NATM", "VARM", "MALE", "OWEN", "TOMAZ", "ALEX"]):
                for v in voices:
                    name_low = v.name.lower()
                    if "david" in name_low or "male" in getattr(v, "gender", "").lower() or "mark" in name_low or "george" in name_low:
                        selected_voice = v.id
                        break
            if selected_voice:
                engine.setProperty("voice", selected_voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        data, sr = sf.read(tmp_path, dtype="float32")
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if data.ndim > 1:
            data = data.mean(axis=1)

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


class DialogueSession:
    """
    Stateful conversational turn manager:
    - Maintains turn memory and history.
    - Prevents repetitive sentences.
    - Delivers concise, efficient 1-2 sentence replies.
    - Adapts dynamically to persona roles.
    """

    def __init__(self, persona_prompt: str):
        self.persona_prompt = persona_prompt.lower()
        self.turn_count = 0
        self.recent_responses: List[str] = []
        self.topics_discussed: Set[str] = set()

    def get_initial_greeting(self) -> str:
        p = self.persona_prompt
        if "mars" in p or "reactor" in p or "astronaut" in p:
            msg = "Mission Control, this is Alex on Mars transit. The reactor core is fluctuating. Please advise on emergency procedures!"
        elif "citysan" in p or "waste" in p:
            msg = "CitySan Services, Ayelen Lucero speaking. How can I help with your collection schedule or bin services today?"
        elif "shakshuka" in p or "restaurant" in p:
            msg = "Welcome to Jerusalem Shakshuka, Owen Foster here. Can I prepare our Classic or Spicy shakshuka for you?"
        elif "aerorentals" in p or "drone" in p:
            msg = "AeroRentals Pro, Tomaz Novak. Are you interested in renting the PhoenixDrone X or SpectraDrone 9 today?"
        else:
            msg = "Hello! I am Sophia. What concept or question can I explain for you today?"

        self.recent_responses.append(msg)
        return msg

    def reply(self, user_text: str) -> str:
        self.turn_count += 1
        u = user_text.lower().strip()
        p = self.persona_prompt

        # Check RAG document knowledge base first
        rag_resp = default_rag_engine.generate_grounded_response(user_text)
        if rag_resp:
            self.recent_responses.append(rag_resp)
            return rag_resp

        # Determine topic and formulate fresh, efficient reply
        if "mars" in p or "reactor" in p:
            resp = self._reply_mars(u)
        elif "citysan" in p:
            resp = self._reply_citysan(u)
        elif "shakshuka" in p:
            resp = self._reply_shakshuka(u)
        elif "aerorentals" in p or "drone" in p:
            resp = self._reply_drone(u)
        else:
            resp = self._reply_teacher(u)

        # De-duplication check: ensure we never repeat recent sentences
        if resp in self.recent_responses[-3:]:
            resp = f"Building on that, let's also examine the next critical step regarding {user_text.rstrip('?.')}."

        self.recent_responses.append(resp)
        if len(self.recent_responses) > 10:
            self.recent_responses.pop(0)

        return resp

    def _reply_mars(self, u: str) -> str:
        if any(w in u for w in ["coolant", "pump", "loop", "valve", "temperature", "stabilize"]):
            if "coolant" not in self.topics_discussed:
                self.topics_discussed.add("coolant")
                return "Rerouting coolant to loop B now. Pressure is dropping below critical! We need to check magnetic containment next."
            else:
                return "Coolant flow is holding at eighty liters per second. Containment field is stabilized. What is the status of the thruster synchronization?"
        elif any(w in u for w in ["status", "report", "how", "damage", "sensor"]):
            return "Core temperature is holding at 760 Kelvin. Manifold three is offline, but auxiliary power is stable at sixty-five percent."
        elif any(w in u for w in ["hello", "hi", "hear", "alex", "online"]):
            return "Loud and clear, Mission Control! Reactor core temperature is rising, please talk me through emergency shutdown."
        else:
            return f"Understood regarding {u.rstrip('?.')}. I am implementing that procedure now. All systems are responsive."

    def _reply_citysan(self, u: str) -> str:
        if any(w in u for w in ["schedule", "pickup", "when", "day", "collection"]):
            return "Your regular pickup is every other week. Your next collection is scheduled for Friday, April 12th."
        elif any(w in u for w in ["compost", "bin", "green", "cost", "price"]):
            return "Yes, green compost bins are available for eight dollars a month. Would you like me to add one to your service?"
        elif any(w in u for w in ["name", "torres", "omar", "verify", "account"]):
            return "Account verified under Omar Torres. Your billing is up to date."
        elif any(w in u for w in ["hello", "hi", "help"]):
            return "Hello! I can check your pickup schedule, order compost bins, or update account details. Which would you prefer?"
        else:
            return f"I have noted your request about {u.rstrip('?.')}. Is there anything else I can update on your CitySan account?"

    def _reply_shakshuka(self, u: str) -> str:
        if any(w in u for w in ["menu", "what", "options", "price", "cost"]):
            return "Classic Shakshuka is nine fifty, and Spicy with jalapenos is ten twenty-five. Both come fresh with optional warm pita."
        elif any(w in u for w in ["order", "classic", "spicy", "want", "have"]):
            return "Order placed! The kitchen is preparing it fresh now. Would you like to add warm pita or Israeli salad with that?"
        elif any(w in u for w in ["hours", "open", "time", "drive"]):
            return "Our drive-through is open daily until nine PM. Everything is made fresh to order."
        else:
            return f"Got it, {u.rstrip('?.')} added. Your total will be ready at the drive-through window in five minutes."

    def _reply_drone(self, u: str) -> str:
        if any(w in u for w in ["price", "cost", "rate", "how much"]):
            return "PhoenixDrone X is sixty-five dollars for four hours. The premium SpectraDrone 9 is ninety-five for four hours."
        elif any(w in u for w in ["deposit", "security", "requirement"]):
            return "We require a refundable deposit of one hundred fifty dollars for standard models, or three hundred for premium."
        else:
            return f"The {u.rstrip('?.')} is reserved and fully charged. When would you like to pick it up today?"

    def _reply_teacher(self, u: str) -> str:
        if any(w in u for w in ["hello", "hi", "hey"]):
            return "Hello! I am ready to explore any concept with you. What topic shall we dive into?"
        elif any(w in u for w in ["how are you", "doing"]):
            return "I am doing wonderfully! Eager to explain ideas and solve problems together. What is on your mind?"
        elif any(w in u for w in ["why", "how", "what", "explain", "tell"]):
            return f"In concise terms, {u.rstrip('?.')} operates on key fundamental principles. Let's look at the primary cause first."
        else:
            return f"That is a great perspective on {u.rstrip('?.')}. How would you like to take this investigation further?"


class PersonaPlexMockServer:
    """
    Mock PersonaPlex server that mimics NVIDIA's server.py WebSocket behavior
    with offline speech synthesis, adaptive noise cancellation, and intelligent turn-taking.
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
            parsed = urllib.parse.urlparse(path)
            query = urllib.parse.parse_qs(parsed.query)
            text_prompt = query.get("text_prompt", [""])[0]
            voice_prompt = query.get("voice_prompt", ["NATF2.pt"])[0]

            logger.info(f"Mock server accepted session for prompt: {text_prompt[:50]}... voice: {voice_prompt}")

            if self.prompt_init_delay > 0:
                await asyncio.sleep(self.prompt_init_delay)

            # Send handshake 0x00
            await websocket.send(encode_message(HandshakeMessage(version=0, model=0)))

            stop_event = asyncio.Event()
            user_speaking = False
            last_speech_time = 0.0
            audio_phase = 0.0

            dialogue = DialogueSession(text_prompt)
            noise_canceller = AdaptiveNoiseCanceller()

            outbound_audio_frames: List[np.ndarray] = []
            outbound_tokens: List[str] = []

            loop = asyncio.get_running_loop()

            async def queue_agent_utterance(reply_text: str):
                nonlocal outbound_audio_frames, outbound_tokens
                words = reply_text.split(" ")
                tokens = [" " + w if i > 0 else w for i, w in enumerate(words)]
                outbound_tokens = tokens

                audio_samples = await loop.run_in_executor(
                    _TTS_EXECUTOR,
                    _synthesize_speech_offline,
                    reply_text,
                    SAMPLE_RATE,
                    voice_prompt,
                )

                if audio_samples is not None and len(audio_samples) > 0:
                    num_frames = int(math.ceil(len(audio_samples) / FRAME_SIZE))
                    padded = np.pad(audio_samples, (0, num_frames * FRAME_SIZE - len(audio_samples)))
                    frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]
                    outbound_audio_frames = frames
                else:
                    t = np.linspace(0, len(tokens) * 0.2, len(tokens) * int(0.2 * SAMPLE_RATE))
                    carrier = (0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
                    num_frames = int(math.ceil(len(carrier) / FRAME_SIZE))
                    padded = np.pad(carrier, (0, num_frames * FRAME_SIZE - len(carrier)))
                    outbound_audio_frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]

            # Queue initial concise greeting
            initial_greeting = dialogue.get_initial_greeting()
            synth_task = asyncio.create_task(queue_agent_utterance(initial_greeting))

            async def receiver():
                nonlocal user_speaking, last_speech_time, outbound_audio_frames, outbound_tokens
                speech_frame_count = 0

                try:
                    async for raw in websocket:
                        if not isinstance(raw, bytes):
                            continue
                        msg = decode_message(raw)

                        if msg.type == MessageType.TEXT:
                            logger.info(f"User utterance: '{msg.text}'")
                            # Immediate barge-in cutoff
                            outbound_audio_frames.clear()
                            outbound_tokens.clear()
                            await websocket.send(encode_message(TextMessage(text=" ")))

                            # Formulate efficient persona reply
                            reply = dialogue.reply(msg.text)
                            logger.info(f"Agent reply: '{reply}'")
                            await queue_agent_utterance(reply)

                        elif msg.type == MessageType.AUDIO:
                            self.total_frames_received += 1
                            raw_samples = (
                                np.frombuffer(msg.data, dtype=np.float32)
                                if len(msg.data) >= FRAME_SIZE * 4
                                else np.zeros(FRAME_SIZE, dtype=np.float32)
                            )
                            # Noise Cancellation Layer
                            clean_samples = noise_canceller.clean_frame(raw_samples)
                            rms = compute_rms(clean_samples)

                            # Turn-taking VAD with 160ms onset debounce
                            if rms > 0.025:
                                speech_frame_count += 1
                                last_speech_time = time.time()
                                if speech_frame_count >= 2:
                                    if not user_speaking:
                                        user_speaking = True
                                        # Instant barge-in: cut off agent speaking immediately
                                        if len(outbound_audio_frames) > 0:
                                            outbound_audio_frames.clear()
                                            outbound_tokens.clear()
                            elif time.time() - last_speech_time > 0.6:
                                # 600ms conversational hangtime
                                if user_speaking and speech_frame_count > 10:
                                    # User spoke and paused without text recognition
                                    user_speaking = False
                                    speech_frame_count = 0
                                    reply = dialogue.reply("I hear you speaking")
                                    await queue_agent_utterance(reply)
                                user_speaking = False
                                speech_frame_count = 0

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
                        # User is speaking: listen silently
                        out_frame = generate_silence_frame()
                        if step % 30 == 0:
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
