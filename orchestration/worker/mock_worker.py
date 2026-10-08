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
import re
import time
import urllib.parse

import numpy as np
import soundfile as sf
import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from ..persona.dialogue import StrictVoiceDialogueEngine
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    AdaptiveNoiseCanceller,
    compute_rms,
    generate_silence_frame,
)
from ..protocol.messages import (
    AudioMessage,
    ControlAction,
    ErrorMessage,
    HandshakeMessage,
    MessageType,
    TextMessage,
    decode_message,
    encode_message,
)

logger = logging.getLogger("orchestration.worker.mock")

# Thread pool for offline TTS synthesis
_TTS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


async def _synthesize_speech_neural(
    text: str,
    target_sr: int = 24000,
    voice_preset: str | None = None,
    accent: str | None = None,
    character: str | None = None,
    neural_voice: str | None = None,
) -> np.ndarray | None:
    """
    Synthesize high-fidelity Indian neural voice via edge_tts.
    Supports en-IN-PrabhatNeural (Professional) and en-IN-NeerjaExpressiveNeural (Friendly & Funny).
    """
    try:
        import io

        import edge_tts

        voice = neural_voice or ""
        if not voice:
            if character and "funny" in character.lower():
                voice = "en-IN-NeerjaExpressiveNeural"
            else:
                voice = "en-IN-PrabhatNeural"

        rate = "+18%"
        pitch = "+0Hz"
        if character:
            c = character.lower()
            if "funny" in c:
                rate = "+22%"
                pitch = "+2Hz"
            elif "professional" in c:
                rate = "+18%"
                pitch = "+0Hz"

        async def _stream_tts():
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf

        buf = await asyncio.wait_for(_stream_tts(), timeout=0.3)
        data, sr = sf.read(buf, dtype="float32")
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
        logger.warning(f"Neural TTS synthesis unavailable: {e}. Falling back to offline synthesis.")
        return None


def _synthesize_formant_speech(text: str, sr: int = 24000) -> np.ndarray:
    """Zero-dependency realistic vowel formant speech synthesizer for 100% offline fallback."""
    words = text.strip().split()
    if not words:
        words = ["Hello"]
    total_samples: list[float] = []
    vowels = [
        (730, 1090, 2440),
        (270, 2290, 3010),
        (530, 1840, 2480),
        (300, 870, 2240),
        (660, 1720, 2410),
    ]
    f0_base = 145.0  # Warm natural conversational pitch

    for w in words:
        duration = max(0.18, min(0.38, len(w) * 0.05))
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        pitch = f0_base + 15.0 * np.sin(np.pi * t / duration)
        glottal = np.sin(2 * np.pi * pitch * t) + 0.4 * np.sin(4 * np.pi * pitch * t)
        f1, f2, f3 = vowels[abs(hash(w)) % len(vowels)]
        formant1 = np.sin(2 * np.pi * f1 * t) * 0.6
        formant2 = np.sin(2 * np.pi * f2 * t) * 0.3
        formant3 = np.sin(2 * np.pi * f3 * t) * 0.15
        vocal = glottal * (formant1 + formant2 + formant3)
        env = np.hanning(n_samples)
        word_audio = vocal * env * 0.35
        pause = np.zeros(int(sr * 0.05), dtype=np.float32)
        total_samples.extend(word_audio.tolist())
        total_samples.extend(pause.tolist())

    return np.array(total_samples, dtype=np.float32)


def _synthesize_speech_offline(
    text: str,
    target_sr: int = 24000,
    voice_preset: str | None = None,
    accent: str | None = None,
    character: str | None = None,
) -> np.ndarray:
    """
    Synthesize speech offline using native system TTS (pyttsx3) or formant synthesis.
    Guaranteed to return valid non-null audio array.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        rate = 210
        if character:
            c = character.lower()
            if "funny" in c:
                rate = 225
            elif "professional" in c:
                rate = 210
            elif "warm" in c:
                rate = 205
        engine.setProperty("rate", rate)
        engine.setProperty("volume", 0.95)

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
        logger.warning(f"Offline TTS synthesis unavailable: {e}. Falling back to formant speech.")
        return _synthesize_formant_speech(text, sr=target_sr)
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
        prompt_init_delay: float | None = None,
        frame_interval_sec: float = 0.08,  # 80ms = 12.5 Hz
    ):
        self.host = host
        self.port = port
        self.prompt_init_delay = (
            prompt_init_delay
            if prompt_init_delay is not None
            else float(os.environ.get("PERSONAPLEX_MOCK_PRIMING_DELAY", "0.5"))
        )
        self.frame_interval = frame_interval_sec

        self._server: Server | None = None
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
        request = getattr(websocket, "request", None)
        path = request.path if request is not None else ""
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
            accent = query.get("accent", ["Indian English"])[0]
            character = query.get("character", ["Professional"])[0]
            neural_voice = query.get("neural_voice", [""])[0]

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
                f"Mock server session: accent={accent}, character={character}, voice={voice_prompt}, neural_voice={neural_voice}"
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

            outbound_audio_frames: list[np.ndarray] = []
            outbound_tokens: list[str] = []

            loop = asyncio.get_running_loop()

            async def queue_agent_utterance(reply_text: str):
                nonlocal outbound_audio_frames, outbound_tokens
                words = reply_text.strip().split()
                tokens = [(" " if i > 0 else "") + w for i, w in enumerate(words)]
                outbound_tokens = tokens

                # Segment into sentences or clauses
                chunks = [s.strip() for s in re.split(r"[.!?]+", reply_text) if s.strip()]
                if not chunks:
                    chunks = [reply_text]

                for chunk_text in chunks:
                    if stop_event.is_set() or user_speaking:
                        break

                    audio_samples = await _synthesize_speech_neural(
                        chunk_text,
                        target_sr=SAMPLE_RATE,
                        voice_preset=voice_prompt,
                        accent=accent,
                        character=character,
                        neural_voice=neural_voice,
                    )

                    if audio_samples is None:
                        audio_samples = await loop.run_in_executor(
                            _TTS_EXECUTOR,
                            _synthesize_speech_offline,
                            chunk_text,
                            SAMPLE_RATE,
                            voice_prompt,
                            accent,
                            character,
                        )

                    if audio_samples is not None and len(audio_samples) > 0:
                        num_frames = int(math.ceil(len(audio_samples) / FRAME_SIZE))
                        padded = np.pad(audio_samples, (0, num_frames * FRAME_SIZE - len(audio_samples)))
                        frames = [padded[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] for i in range(num_frames)]
                        # Add chunk frames immediately so client begins playing Chunk 1 while Chunk 2 synthesizes!
                        outbound_audio_frames.extend(frames)

                pass

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
                                # 500ms conversational hangtime - end turn cleanly
                                if user_speaking:
                                    user_speaking = False
                                    speech_frame_count = 0
                                    reply = dialogue.reply("I heard you speak. How can I assist you with your request?")
                                    logger.info(f"Mock agent reply on voice turn completion: '{reply}'")
                                    await queue_agent_utterance(reply)

                        elif msg.type == MessageType.CONTROL:
                            if msg.action == ControlAction.PAUSE:
                                outbound_audio_frames.clear()
                                outbound_tokens.clear()

                except websockets.ConnectionClosed:
                    pass
                except Exception as ee:
                    logger.debug(f"Mock receiver exception: {ee}")
                finally:
                    stop_event.set()

            async def generator():
                step = 0

                while not stop_event.is_set():
                    t0 = time.time()
                    step += 1

                    if user_speaking:
                        # User is speaking: stream silence
                        out_frame = generate_silence_frame()
                    elif len(outbound_audio_frames) > 0:
                        out_frame = outbound_audio_frames.pop(0)
                        # Stream tokens briskly in sync with speech (1 token per 80ms frame)
                        if len(outbound_tokens) > 0:
                            token = outbound_tokens.pop(0)
                            await websocket.send(encode_message(TextMessage(text=token)))
                    elif len(outbound_tokens) > 0:
                        out_frame = generate_silence_frame()
                        token = outbound_tokens.pop(0)
                        await websocket.send(encode_message(TextMessage(text=token)))
                    else:
                        out_frame = generate_silence_frame()

                    await websocket.send(encode_message(AudioMessage(data=out_frame.tobytes())))
                    self.total_frames_generated += 1

                    elapsed = time.time() - t0
                    sleep_time = max(0.001, self.frame_interval - elapsed)
                    await asyncio.sleep(sleep_time)

            recv_task = asyncio.create_task(receiver())
            gen_task = asyncio.create_task(generator())

            done, pending = await asyncio.wait([recv_task, gen_task], return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.active_connections -= 1
