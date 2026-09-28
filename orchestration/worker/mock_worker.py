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

from ..persona.dialogue import StrictVoiceDialogueEngine, GroundedDialogueEngine

logger = logging.getLogger("orchestration.worker.mock")

# Thread pool for offline TTS synthesis
_TTS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _synthesize_speech_offline(
    text: str,
    target_sr: int = 24000,
    voice_preset: Optional[str] = None,
    accent: Optional[str] = None,
    character: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Synthesize speech offline using native system TTS (pyttsx3)
    and resample to 24,000 Hz float32 mono PCM.
    Adapts voice characteristics according to accent and character.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()

        # Dynamic rate tuning based on character
        rate = 182
        if character:
            c = character.lower()
            if "funny" in c:
                rate = 195  # Energetic, snappy
            elif "professional" in c:
                rate = 175  # Polished, measured
            elif "warm" in c:
                rate = 182  # Grounded, warm
        engine.setProperty("rate", rate)
        engine.setProperty("volume", 0.95)

        voices = engine.getProperty("voices")
        selected_voice = None

        # Priority 1: Match system voice to accent if available
        if accent:
            acc_low = accent.lower()
            if "indian" in acc_low:
                for v in voices:
                    n = v.name.lower()
                    if any(k in n for k in ["india", "ravi", "heera", "kalpana", "en-in"]):
                        selected_voice = v.id
                        break
            elif "british" in acc_low:
                for v in voices:
                    n = v.name.lower()
                    if any(k in n for k in ["united kingdom", "great britain", "george", "hazel", "susan", "en-gb"]):
                        selected_voice = v.id
                        break

        # Priority 2: Match gender/timbre from voice preset
        if not selected_voice and voice_preset:
            vp = voice_preset.upper()
            if any(k in vp for k in ["NATF", "VARF", "FEMALE", "SOPHIA", "ANANYA", "SARAH", "MAYA", "EMMA"]):
                for v in voices:
                    name_low = v.name.lower()
                    if "zira" in name_low or "female" in getattr(v, "gender", "").lower() or "eva" in name_low:
                        selected_voice = v.id
                        break
            elif any(k in vp for k in ["NATM", "VARM", "MALE", "AARAV", "ROHAN", "JACK", "ARTHUR", "OLIVER"]):
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
    Intelligent conversational turn manager powered by StrictVoiceDialogueEngine.
    Follows the 12 Conversation Principles, truthful, zero hallucinations, tailored for 3 accents and 3 characters.
    """

    def __init__(
        self,
        persona_prompt: str,
        accent: str = "American English",
        character: str = "Confident, Warm & Concise",
    ):
        self.persona_prompt = persona_prompt
        self.engine = StrictVoiceDialogueEngine(
            accent=accent,
            character=character,
            custom_system_prompt=persona_prompt,
        )

    def get_initial_greeting(self) -> str:
        return self.engine.get_greeting()

    def reply(self, user_text: str) -> str:
        return self.engine.reply(user_text)


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
            accent = query.get("accent", ["American English"])[0]
            character = query.get("character", ["Confident, Warm & Concise"])[0]

            # Infer from prompt if present
            tp_lower = text_prompt.lower()
            if "indian" in tp_lower:
                accent = "Indian English"
            elif "british" in tp_lower:
                accent = "British English"
            elif "american" in tp_lower:
                accent = "American English"

            if "funny" in tp_lower:
                character = "Funny"
            elif "professional" in tp_lower:
                character = "Professional"
            elif "warm" in tp_lower or "concise" in tp_lower:
                character = "Confident, Warm & Concise"

            logger.info(
                f"Mock server session: accent={accent}, character={character}, voice={voice_prompt}"
            )

            if self.prompt_init_delay > 0:
                await asyncio.sleep(self.prompt_init_delay)

            # Send handshake 0x00
            await websocket.send(encode_message(HandshakeMessage(version=0, model=0)))

            stop_event = asyncio.Event()
            user_speaking = False
            last_speech_time = 0.0

            dialogue = DialogueSession(text_prompt, accent=accent, character=character)
            noise_canceller = AdaptiveNoiseCanceller()

            outbound_audio_frames: List[np.ndarray] = []
            outbound_tokens: List[str] = []

            loop = asyncio.get_running_loop()

            async def queue_agent_utterance(reply_text: str):
                nonlocal outbound_audio_frames, outbound_tokens
                words = reply_text.strip().split()
                tokens = [(" " if i > 0 else "") + w for i, w in enumerate(words)]
                outbound_tokens = tokens

                audio_samples = await loop.run_in_executor(
                    _TTS_EXECUTOR,
                    _synthesize_speech_offline,
                    reply_text,
                    SAMPLE_RATE,
                    voice_prompt,
                    accent,
                    character,
                )

                if audio_samples is not None and len(audio_samples) > 0:
                    num_frames = int(math.ceil(len(audio_samples) / FRAME_SIZE))
                    padded = np.pad(audio_samples, (0, num_frames * FRAME_SIZE - len(audio_samples)))
                    frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]
                    outbound_audio_frames = frames
                else:
                    outbound_audio_frames = [generate_silence_frame() for _ in range(max(1, len(tokens)))]

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

                            # Formulate truthful, grounded persona reply
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

                            # Turn-taking VAD with sensitive onset debounce for laptop microphones
                            if rms > 0.008:
                                speech_frame_count += 1
                                last_speech_time = time.time()
                                if speech_frame_count >= 2:
                                    if not user_speaking:
                                        user_speaking = True
                                        # Instant barge-in: cut off agent speaking immediately
                                        if len(outbound_audio_frames) > 0:
                                            outbound_audio_frames.clear()
                                            outbound_tokens.clear()
                            elif time.time() - last_speech_time > 0.5:
                                # 500ms conversational hangtime
                                if user_speaking and speech_frame_count >= 3:
                                    # User spoke audio frames and paused
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
                token_step_interval = 2  # Emit token cadence aligned with speech
                step = 0

                while not stop_event.is_set():
                    t0 = time.time()
                    step += 1

                    if user_speaking:
                        # User is speaking: listen in pure silence
                        out_frame = generate_silence_frame()
                    elif len(outbound_audio_frames) > 0:
                        out_frame = outbound_audio_frames.pop(0)
                        if step % token_step_interval == 0 and len(outbound_tokens) > 0:
                            token = outbound_tokens.pop(0)
                            await websocket.send(encode_message(TextMessage(text=token)))
                    else:
                        # Clean silence when idle (NO buzz, NO tone)
                        out_frame = generate_silence_frame()

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
