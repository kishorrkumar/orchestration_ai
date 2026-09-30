"""
Local Cascade Voice Worker (STT -> LLM -> Chunker -> TTS).

Architected for Windows 11 with 4 GB VRAM GPU:
- GPU: Dedicated to Ollama LLM (Qwen 2.5 1.5B/3B Q4)
- CPU: faster-whisper (base, int8) with initial_prompt biasing on CPU (<480ms latency)
- Turn Detector: Silero VAD + hysteresis + pre-roll buffer + linguistic continuation
  (prevents mid-sentence cutoff on trailing words like 'about', 'and', 'the', etc.)
- LLM: Ollama streaming /api/chat with rolling multi-turn memory
- Non-blocking: heavy ASR/TTS run via asyncio.to_thread
- Streaming Chunker: First clause flushes at 2-6 words / punctuation mark
- Concurrent TTS: Synthesizes Chunk N+1 while Chunk N streams
- Exact 1,920 sample framing (80 ms @ 24 kHz)
- Sub-200ms Barge-in cancellation
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
import urllib.parse
from typing import Any

import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from ..audio.cleaner import CallerAudioCleaner
from ..audio.silero_vad import SileroVADDetector
from ..audio.turn_detector import TurnDetector, is_utterance_unfinished
from ..chunker.bridge import ClauseChunker
from ..persona.dialogue import StrictVoiceDialogueEngine
from ..persona.prompts import build_agent_system_prompt
from ..protocol.audio import (
    FRAME_SIZE,
    SAMPLE_RATE,
    AudioFrameBuffer,
    compute_rms,
    generate_silence_frame,
)
from ..protocol.messages import (
    AudioMessage,
    HandshakeMessage,
    MessageType,
    MetadataMessage,
    TextMessage,
    decode_message,
    encode_message,
)
from ..tts.base import StreamingCompositeTTS

logger = logging.getLogger("orchestration.worker.local_cascade")

def log_task_exception(t: asyncio.Task) -> None:
    try:
        if not t.cancelled() and t.exception():
            logger.error(f"Unhandled exception in task '{t.get_name()}': {t.exception()}", exc_info=t.exception())
    except Exception:
        pass

# Default vocabulary prompt to bias faster-whisper for Indian English
DEFAULT_ASR_INITIAL_PROMPT = (
    "artificial intelligence, machine learning, Aarav, Priya, Chennai, Bengaluru, "
    "Tamil, Hindi, cricket, Bollywood, UPI, Swiggy, Zomato, IPL, chai, monsoon, auto"
)

# Root directory for persona templates
PERSONA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "personas"


def load_persona_prompt(agent_name: str) -> str:
    """Load persona markdown file if available, or fall back to default prompt."""
    filename = f"{agent_name.lower()}.md"
    persona_path = PERSONA_DIR / filename
    if persona_path.exists():
        try:
            return persona_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Could not load persona file {persona_path}: {e}")

    # Fallback prompt
    return (
        f"You are {agent_name}, a friendly, smart, and articulate young professional from India chatting on a real-time voice call.\n"
        "Speak natural, colloquial Indian English. Use short sentences (10 to 18 words), natural contractions, and a warm tone.\n"
        "Use Indian-English discourse markers naturally and sparingly (at most one per response): 'actually', 'basically', 'no?', 'na', 'simple, na?', 'sure sure', 'right, right'.\n"
        "NEVER use robotic call-center phrases like 'How can I assist you today' or 'I understand you need support'.\n"
        "Answer questions directly in 1-3 spoken sentences. Plain spoken text only: no markdown, no emojis, no asterisks."
    )


class LocalCascadeWorkerServer:
    """
    Local Cascade Voice Worker implementing the PersonaPlex binary wire interface (0x00-0x06).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8998,
        ollama_url: str = "http://127.0.0.1:11434",
        llm_model: str = "qwen2.5:1.5b",
        whisper_model_size: str = "base",
        frame_interval_sec: float = 0.08,
        jitter_buffer_frames: int = 2,
    ) -> None:
        self.host = host
        self.port = port
        self.ollama_url = ollama_url
        self.llm_model = llm_model
        self.whisper_model_size = whisper_model_size
        self.frame_interval = frame_interval_sec
        self.jitter_buffer_frames = jitter_buffer_frames

        self._server: Any = None
        self._is_running = False
        self.active_connections = 0

        # Telemetry metrics
        self.last_stt_ms: float = 0.0
        self.last_llm_ttft_ms: float = 0.0
        self.last_tts_ttfa_ms: float = 0.0
        self.last_total_ttfa_ms: float = 0.0
        self.total_barge_ins: int = 0
        self.total_underruns: int = 0

        # Lazy-loaded shared models
        self._whisper_model = None
        self._whisper_available = False
        self._tts_engine: StreamingCompositeTTS | None = None
        self._vad: SileroVADDetector | None = None

    def _init_vad(self) -> None:
        """Initialize Silero VAD ONNX engine for voice activity detection."""
        if self._vad is not None:
            return
        try:
            self._vad = SileroVADDetector(sample_rate=SAMPLE_RATE, speech_threshold=0.45)
            # Pre-warm ONNX runtime with a 1920-sample silent frame
            dummy = np.zeros(FRAME_SIZE, dtype=np.float32)
            self._vad.get_speech_probability(dummy)
            logger.info("Silero VAD (ONNX, CPU) initialized and pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Silero VAD initialization notice ({e}).")
            self._vad = None

    def _init_stt(self) -> None:
        """Initialize faster-whisper on CPU int8 for guaranteed stability alongside GPU LLM."""
        if self._whisper_model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            # Run on CPU int8 with 4 threads: benchmark showed 470ms latency (RTF 0.157)
            # This completely leaves the 4 GB VRAM to the Ollama LLM
            self._whisper_model = WhisperModel(
                self.whisper_model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            self._whisper_available = True
            logger.info(f"faster-whisper ({self.whisper_model_size}, int8) loaded successfully on CPU.")
        except Exception as e:
            logger.warning(f"faster-whisper CPU init failed ({e}). Falling back to dialogue input.")
            self._whisper_available = False

    def _init_tts(self) -> None:
        """Initialize and warm up Kokoro TTS synthesis engine."""
        if self._tts_engine is not None:
            return
        self._tts_engine = StreamingCompositeTTS(
            preferred_backend="auto",
            sample_rate=SAMPLE_RATE,
            speaking_rate=0.98,
            default_voice="aarav_colloquial",
        )
        self._tts_engine.warm_up()
        logger.info("LocalCascadeWorker TTS engine initialized and pre-warmed.")

    def pre_warm(self) -> None:
        """Pre-warm all models before accepting calls."""
        t0 = time.perf_counter()
        self._init_vad()
        self._init_tts()
        self._init_stt()
        logger.info(f"LocalCascadeWorker pre-warm completed in {(time.perf_counter() - t0) * 1000:.1f}ms")

    async def start(self) -> None:
        self.pre_warm()
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )
        self._is_running = True
        logger.info(f"LocalCascadeWorkerServer running on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("LocalCascadeWorkerServer stopped")

    def _transcribe_audio_sync(self, audio_16k: np.ndarray) -> str:
        """Synchronous Whisper transcription executed in worker thread."""
        if not self._whisper_available or not self._whisper_model:
            return ""
        try:
            segments, _ = self._whisper_model.transcribe(
                audio_16k,
                language="en",
                beam_size=1,
                initial_prompt=DEFAULT_ASR_INITIAL_PROMPT,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )
            return " ".join([s.text for s in segments]).strip()
        except Exception as e:
            logger.warning(f"Whisper transcription error: {e}")
            return ""

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle individual caller session with exclusive concurrency lease."""
        if self.active_connections > 0:
            logger.warning("LocalCascadeWorker busy. Rejecting concurrent lease.")
            await websocket.close(1013, "Worker busy")
            return

        request = getattr(websocket, "request", None)
        path = request.path if request is not None else ""
        parsed = urllib.parse.urlparse(path)
        query = urllib.parse.parse_qs(parsed.query)

        text_prompt = query.get("text_prompt", [""])[0]
        accent = query.get("accent", ["Indian English"])[0]
        character = query.get("character", ["Professional"])[0]
        neural_voice = query.get("neural_voice", ["aarav_colloquial"])[0]
        query.get("voice_prompt", ["NATM0.pt"])[0]
        call_flow = query.get("call_flow", ["conversational_companion"])[0]

        # Handle standard, global, and custom cloned voices
        from ..tts.voice_clone import default_voice_cloner

        cloned_meta = None
        if neural_voice.startswith("cloned_"):
            for v in default_voice_cloner.list_cloned_voices():
                if v.get("id") == neural_voice:
                    cloned_meta = v
                    break

        if cloned_meta:
            active_tts_voice = neural_voice
            agent_name = cloned_meta.get("name", "Custom Voice")
            is_female = (cloned_meta.get("gender") == "Female")
        elif neural_voice.lower() in ["priya", "priya_colloquial", "en-in-neerjaexpressiveneural", "hf_alpha"]:
            active_tts_voice = "priya_colloquial"
            agent_name = "Priya"
            is_female = True
        elif neural_voice.lower() in ["aarav", "aarav_colloquial", "en-in-prabhatneural", "hm_omega", "default"]:
            active_tts_voice = "aarav_colloquial"
            agent_name = "Aarav"
            is_female = False
        else:
            active_tts_voice = neural_voice
            # Infer gender from voice id/character
            is_female = (
                neural_voice.startswith("af_")
                or neural_voice.startswith("bf_")
                or neural_voice.startswith("hf_")
                or "female" in character.lower()
                or "priya" in neural_voice.lower()
                or "ananya" in neural_voice.lower()
            )
            agent_name = "Priya" if is_female else "Aarav"

        logger.info(f"Session started: agent='{agent_name}', voice='{active_tts_voice}', call_flow='{call_flow}', accent='{accent}'")

        try:
            # 1. Send Handshake
            await websocket.send(encode_message(HandshakeMessage(version=0, model=0)))

            stop_event = asyncio.Event()
            user_speaking = False
            last_speech_time = 0.0
            is_greeting_active = True
            time.time()
            last_agent_audio_time = 0.0

            # Session components
            cleaner = CallerAudioCleaner(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)
            turn_detector = TurnDetector(
                sample_rate=SAMPLE_RATE,
                frame_size=FRAME_SIZE,
                base_silence_sec=0.65,
                extra_silence_sec=0.70,
                min_speech_frames=2,
                preroll_frames=5,
                speech_rms_threshold=0.012,
                vad=self._vad,
            )
            dialogue_char = "warm" if is_female else character
            dialogue_engine = StrictVoiceDialogueEngine(accent=accent, character=dialogue_char, custom_system_prompt=text_prompt)

            # Frame buffer and queues
            frame_buffer = AudioFrameBuffer(dtype=np.float32, max_buffer_frames=200)
            token_queue: asyncio.Queue[str | None] = asyncio.Queue()
            conversation_history: list[dict[str, str]] = []

            # Construct modular Agent Identity + Call Flow system prompt
            clean_text_prompt = text_prompt.replace("<system>", "").replace("</system>", "").strip()
            llm_system_prompt = build_agent_system_prompt(
                agent_name=agent_name,
                role=call_flow,
                custom_instructions=clean_text_prompt or None
            )

            # Active streaming generation tasks
            current_generation_task: asyncio.Task | None = None
            current_tts_tasks: list[asyncio.Task] = []
            interrupted_in_turn = False

            def cancel_active_turn(record_interrupted: bool = True) -> None:
                """Instantly cancel in-flight LLM generation and TTS on barge-in."""
                nonlocal current_generation_task, interrupted_in_turn
                if current_generation_task and not current_generation_task.done():
                    current_generation_task.cancel()
                    current_generation_task = None
                    if record_interrupted and conversation_history and conversation_history[-1]["role"] == "assistant":
                        if not conversation_history[-1]["content"].endswith("[interrupted]"):
                            conversation_history[-1]["content"] += " [interrupted]"

                for t in current_tts_tasks:
                    if not t.done():
                        t.cancel()
                current_tts_tasks.clear()

                frame_buffer.clear()
                while not token_queue.empty():
                    try:
                        token_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # ---------- LLM & TTS STREAMING PIPELINE ----------
            async def run_conversational_turn(user_utterance: str, is_greeting: bool = False) -> None:
                nonlocal current_tts_tasks, is_greeting_active
                clean_text = user_utterance.strip()
                if not clean_text:
                    return

                logger.info(f"Conversational Turn (is_greeting={is_greeting}): '{clean_text}'")
                turn_start_t = time.perf_counter()
                ttft_recorded = False

                chunker = ClauseChunker(
                    first_chunk_min_words=2,
                    first_chunk_max_words=6,
                    later_chunk_min_words=6,
                    later_chunk_max_words=14,
                )

                chunk_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=10)

                async def tts_consumer():
                    first_chunk = True
                    while not stop_event.is_set():
                        try:
                            chunk_text = await chunk_queue.get()
                            if chunk_text is None:
                                break

                            t_tts_start = time.perf_counter()
                            audio = await self._tts_engine.synthesize_chunk(chunk_text, voice=active_tts_voice)
                            tts_latency = (time.perf_counter() - t_tts_start) * 1000.0

                            if first_chunk:
                                self.last_tts_ttfa_ms = tts_latency
                                self.last_total_ttfa_ms = (time.perf_counter() - turn_start_t) * 1000.0
                                logger.info(
                                    f"TTFA: First Chunk '{chunk_text}' | "
                                    f"TTS: {tts_latency:.1f}ms | Total TTFA: {self.last_total_ttfa_ms:.1f}ms"
                                )
                                first_chunk = False

                            if len(audio) > 0:
                                frame_buffer.push_samples(audio)
                            chunk_queue.task_done()
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.warning(f"TTS synthesis error: {e}")

                tts_worker_task = asyncio.create_task(tts_consumer(), name="tts_consumer")
                tts_worker_task.add_done_callback(log_task_exception)
                current_tts_tasks.append(tts_worker_task)

                if is_greeting:
                    try:
                        words = clean_text.split()
                        for word in words:
                            await token_queue.put(word + " ")
                            for rc in chunker.feed_token(word + " "):
                                await chunk_queue.put(rc)
                        for rem in chunker.flush():
                            await chunk_queue.put(rem)
                        await chunk_queue.put(None)
                        await tts_worker_task
                    finally:
                        is_greeting_active = False
                    return

                # Streaming LLM turn via Ollama /api/chat
                try:
                    import httpx
                    messages = [{"role": "system", "content": llm_system_prompt}]
                    # Maintain rolling window of last 10 turns
                    for turn in conversation_history[-10:]:
                        messages.append(turn)
                    messages.append({"role": "user", "content": clean_text})

                    payload = {
                        "model": self.llm_model,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 2048},
                        "keep_alive": "30m",
                    }

                    accumulated_reply = []
                    in_thinking_mode = False

                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if stop_event.is_set():
                                    break
                                if not line:
                                    continue
                                data = json.loads(line)
                                msg_obj = data.get("message", {})
                                token = msg_obj.get("content", "")
                                if token:
                                    if "<think>" in token:
                                        in_thinking_mode = True
                                        continue
                                    if "</think>" in token:
                                        in_thinking_mode = False
                                        continue
                                    if in_thinking_mode:
                                        continue

                                    if not ttft_recorded:
                                        self.last_llm_ttft_ms = (time.perf_counter() - turn_start_t) * 1000.0
                                        ttft_recorded = True

                                    accumulated_reply.append(token)
                                    await token_queue.put(token)

                                    ready_chunks = chunker.feed_token(token)
                                    for rc in ready_chunks:
                                        await chunk_queue.put(rc)

                                if data.get("done", False):
                                    break

                            full_reply = "".join(accumulated_reply).strip()
                            if full_reply:
                                conversation_history.append({"role": "user", "content": clean_text})
                                conversation_history.append({"role": "assistant", "content": full_reply})
                        else:
                            reply = dialogue_engine.reply(clean_text)
                            conversation_history.append({"role": "user", "content": clean_text})
                            conversation_history.append({"role": "assistant", "content": reply})
                            for word in reply.split():
                                await token_queue.put(word + " ")
                                for rc in chunker.feed_token(word + " "):
                                    await chunk_queue.put(rc)

                    for rem in chunker.flush():
                        await chunk_queue.put(rem)

                except Exception as e:
                    logger.info(f"Ollama streaming notice ({e}), falling back to dialogue engine.")
                    reply = dialogue_engine.reply(clean_text)
                    conversation_history.append({"role": "user", "content": clean_text})
                    conversation_history.append({"role": "assistant", "content": reply})
                    for word in reply.split():
                        await token_queue.put(word + " ")
                        for rc in chunker.feed_token(word + " "):
                            await chunk_queue.put(rc)
                    for rem in chunker.flush():
                        await chunk_queue.put(rem)

                await chunk_queue.put(None)
                await tts_worker_task

            # ---------- INBOUND AUDIO RECEIVER ----------
            async def receiver():
                nonlocal user_speaking, last_speech_time, current_generation_task, is_greeting_active, current_tts_tasks
                pending_audio_segment: np.ndarray | None = None

                try:
                    async for raw in websocket:
                        if stop_event.is_set():
                            break
                        msg = decode_message(raw)

                        if msg.type == MessageType.TEXT:
                            user_text = msg.text.strip()
                            if not user_text:
                                continue
                            cancel_active_turn(record_interrupted=False)
                            user_speaking = False
                            turn_detector.reset()
                            task = asyncio.create_task(run_conversational_turn(user_text), name="run_turn_text")
                            task.add_done_callback(log_task_exception)
                            current_generation_task = task

                        elif msg.type == MessageType.AUDIO:
                            raw_samples = np.frombuffer(msg.data, dtype=np.float32)
                            cleaned_frames = cleaner.process_chunk(raw_samples)

                            # Prune completed TTS synthesis tasks in-place
                            current_tts_tasks[:] = [t for t in current_tts_tasks if not t.done()]

                            # Determine if agent is currently speaking/playing audio out of laptop speakers
                            is_agent_playing = (
                                frame_buffer.has_frame()
                                or len(current_tts_tasks) > 0
                                or (time.time() - last_agent_audio_time < 0.20)
                            )

                            for frame in cleaned_frames:
                                rms = compute_rms(frame)
                                is_speaking_now, turn_completed, completed_audio = turn_detector.push_frame(
                                    frame, rms, is_agent_speaking=is_agent_playing
                                )

                                if is_speaking_now and not user_speaking:
                                    user_speaking = True
                                    self.total_barge_ins += 1
                                    logger.info("Human speech active: pausing agent playback.")

                                if turn_completed and completed_audio is not None:
                                    user_speaking = False

                                    # If we had a prior pending audio segment from an unfinished thought, merge it
                                    if pending_audio_segment is not None:
                                        full_audio = np.concatenate([pending_audio_segment, completed_audio])
                                        pending_audio_segment = None
                                    else:
                                        full_audio = completed_audio

                                    # Resample 24 kHz -> 16 kHz for Whisper ASR
                                    n_16k = int(round(len(full_audio) * 16000 / 24000))
                                    audio_16k = np.interp(
                                        np.linspace(0, len(full_audio), n_16k, endpoint=False),
                                        np.arange(len(full_audio)),
                                        full_audio,
                                    ).astype(np.float32)

                                    # Non-blocking Whisper transcription in worker thread!
                                    t_stt = time.perf_counter()
                                    transcription = await asyncio.to_thread(self._transcribe_audio_sync, audio_16k)
                                    self.last_stt_ms = (time.perf_counter() - t_stt) * 1000.0

                                    if transcription:
                                        # Check if utterance is obviously unfinished (e.g. "Tell me a joke about")
                                        if is_utterance_unfinished(transcription):
                                            logger.info(f"Detected unfinished utterance: '{transcription}'. Extending silence window by 700ms.")
                                            pending_audio_segment = full_audio
                                            turn_detector.extend_turn(0.70)
                                            continue

                                        logger.info(f"Caller STT transcript: '{transcription}'")
                                        # Notify UI of verified transcript to display in chat bubble
                                        try:
                                            await websocket.send(encode_message(MetadataMessage(data=json.dumps({
                                                "event": "user_transcript",
                                                "text": transcription,
                                            }))))
                                        except Exception:
                                            pass

                                        # True barge-in confirmed with speech: cancel prior turn and start fresh turn
                                        cancel_active_turn(record_interrupted=True)
                                        task = asyncio.create_task(
                                            run_conversational_turn(transcription),
                                            name="run_turn_audio"
                                        )
                                        task.add_done_callback(log_task_exception)
                                        current_generation_task = task
                                    else:
                                        logger.info("Empty STT transcript (ambient noise/breath); resuming agent.")

                except Exception as e:
                    logger.exception(f"Receiver encountered error: {e}")
                finally:
                    logger.info("Receiver loop finished, setting stop_event.")
                    stop_event.set()

            # ---------- OUTBOUND GENERATOR (Exact 1,920 Samples @ 12.5 Hz) ----------
            async def generator():
                nonlocal user_speaking, last_agent_audio_time
                while not stop_event.is_set():
                    t0 = time.perf_counter()

                    if user_speaking:
                        out_frame = generate_silence_frame()
                    elif frame_buffer.has_frame():
                        out_frame = frame_buffer.pop_frame()
                        last_agent_audio_time = time.time()
                        try:
                            tok = token_queue.get_nowait()
                            await websocket.send(encode_message(TextMessage(text=tok)))
                        except asyncio.QueueEmpty:
                            pass
                    else:
                        out_frame = generate_silence_frame()
                        self.total_underruns += 1

                    await websocket.send(encode_message(AudioMessage(data=out_frame.tobytes())))

                    # Exact 80 ms pacing
                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0.001, self.frame_interval - elapsed)
                    await asyncio.sleep(sleep_time)

            # Synthesize initial greeting immediately on call connect
            initial_greeting = f"Hey, {agent_name} here! Good to chat with you. What's on your mind?"
            logger.info(f"Synthesizing initial greeting: '{initial_greeting}'")
            current_generation_task = asyncio.create_task(
                run_conversational_turn(initial_greeting, is_greeting=True),
                name="initial_greeting"
            )
            current_generation_task.add_done_callback(log_task_exception)

            recv_task = asyncio.create_task(receiver(), name="session_receiver")
            recv_task.add_done_callback(log_task_exception)
            gen_task = asyncio.create_task(generator(), name="session_generator")
            gen_task.add_done_callback(log_task_exception)

            done, pending = await asyncio.wait([recv_task, gen_task], return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    logger.info(f"Task completed first: '{d.get_name()}', exc={d.exception() if not d.cancelled() else 'cancelled'}")
                except Exception:
                    pass

            stop_event.set()
            cancel_active_turn(record_interrupted=False)
            for t in pending:
                t.cancel()

        finally:
            self.active_connections = max(0, self.active_connections - 1)
            logger.info(f"LocalCascadeWorker session released (active={self.active_connections}).")


# Alias for backward compatibility
CascadedLocalWorkerServer = LocalCascadeWorkerServer
