"""
Pipecat Pipeline Factory for Engine B.
Constructs and wires the streaming STT -> LLM -> TTS pipeline based on agent PipelineSpec and vault keys.
"""

from __future__ import annotations

import logging
from typing import Dict

from pipecat.processors.frame_processor import FrameProcessor

from orchestration.domain.engines.spec import CascadedPipelineSpec

from .fakes import FakeLLMService, FakeSTTService, FakeTTSService

logger = logging.getLogger("orchestration.engines.cascaded.factory")


def build_stt_processor(
    spec: CascadedPipelineSpec,
    credentials: Dict[str, str],
    sample_rate: int = 16000,
) -> FrameProcessor:
    provider = spec.stt.provider

    if provider == "fake_stt":
        return FakeSTTService(sample_rate=sample_rate)

    api_key = credentials.get(provider, "")

    if provider == "deepgram_stt":
        model = spec.stt.model
        if "flux" in model.lower():
            from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService, DeepgramFluxSTTSettings
            settings = DeepgramFluxSTTSettings(eager_end_of_turn=spec.turn.eager_eot)
            return DeepgramFluxSTTService(api_key=api_key, settings=settings)
        else:
            from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
            return DeepgramSTTService(api_key=api_key, settings=DeepgramSTTSettings(model=model))

    if provider == "sarvam_stt":
        from pipecat.services.sarvam.stt import SarvamRealtimeSTTService, SarvamRealtimeSTTSettings
        return SarvamRealtimeSTTService(api_key=api_key, settings=SarvamRealtimeSTTSettings(model=spec.stt.model))

    logger.warning("Unrecognized STT provider '%s', falling back to FakeSTT", provider)
    return FakeSTTService(sample_rate=sample_rate)


def build_llm_processor(
    spec: CascadedPipelineSpec,
    credentials: Dict[str, str],
    system_prompt: str,
) -> FrameProcessor:
    provider = spec.llm.provider

    if provider == "fake_llm":
        return FakeLLMService()

    api_key = credentials.get(provider, "")

    if provider == "openai_llm":
        from pipecat.services.openai.llm import OpenAILLMService
        base_url = spec.llm.options.get("base_url")
        return OpenAILLMService(
            api_key=api_key,
            model=spec.llm.model,
            base_url=base_url,
        )

    if provider == "anthropic_llm":
        from pipecat.services.anthropic.llm import AnthropicLLMService
        return AnthropicLLMService(
            api_key=api_key,
            model=spec.llm.model,
        )

    logger.warning("Unrecognized LLM provider '%s', falling back to FakeLLM", provider)
    return FakeLLMService()


def build_tts_processor(
    spec: CascadedPipelineSpec,
    credentials: Dict[str, str],
    sample_rate: int = 16000,
) -> FrameProcessor:
    provider = spec.tts.provider

    if provider == "fake_tts":
        return FakeTTSService(sample_rate=sample_rate)

    api_key = credentials.get(provider, "")

    if provider == "cartesia_tts":
        from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
        settings = CartesiaTTSSettings(
            model=spec.tts.model,
            voice=spec.tts.voice_id,
            sample_rate=sample_rate,
        )
        return CartesiaTTSService(api_key=api_key, settings=settings)

    if provider == "elevenlabs_tts":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
        settings = ElevenLabsTTSSettings(
            model=spec.tts.model,
            voice=spec.tts.voice_id,
        )
        return ElevenLabsTTSService(api_key=api_key, settings=settings)

    if provider == "deepgram_tts":
        from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
        settings = DeepgramTTSSettings(
            model=spec.tts.model,
            voice=spec.tts.voice_id,
        )
        return DeepgramTTSService(api_key=api_key, settings=settings)

    if provider == "sarvam_tts":
        from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
        settings = SarvamTTSSettings(
            model=spec.tts.model,
            speaker=spec.tts.voice_id,
        )
        return SarvamTTSService(api_key=api_key, settings=settings)

    logger.warning("Unrecognized TTS provider '%s', falling back to FakeTTS", provider)
    return FakeTTSService(sample_rate=sample_rate)
