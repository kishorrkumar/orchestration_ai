"""
Domain Specifications and Schemas for Engine Selection and Cascaded Cloud Pipelines.
Strictly framework-free.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class EngineType(enum.StrEnum):
    PERSONAPLEX_S2S = "personaplex_s2s"
    CASCADED_CLOUD = "cascaded_cloud"


class TurnStrategyType(enum.StrEnum):
    AUTO = "auto"
    PROVIDER_EOT = "provider_eot"
    VAD_SMART_TURN = "vad_smart_turn"


class STTPipelineConfig(BaseModel):
    provider: str = Field("fake_stt", description="STT Provider ID (e.g., deepgram_stt, sarvam_stt, fake_stt)")
    model: str = Field("mock-flux", description="Model ID")
    language: str = Field("en-US", description="BCP-47 language tag")
    options: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific STT settings")


class LLMPipelineConfig(BaseModel):
    provider: str = Field("fake_llm", description="LLM Provider ID (e.g., openai_llm, anthropic_llm, fake_llm)")
    model: str = Field("mock-gpt", description="Model ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(200, ge=10, le=4096)
    options: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific LLM options")


class TTSPipelineConfig(BaseModel):
    provider: str = Field("fake_tts", description="TTS Provider ID (e.g., cartesia_tts, elevenlabs_tts, sarvam_tts, fake_tts)")
    model: str = Field("mock-voice", description="Model ID")
    voice_id: str = Field("mock-alex", description="Voice ID")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    options: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific TTS options")


class TurnPipelineConfig(BaseModel):
    strategy: TurnStrategyType = Field(TurnStrategyType.AUTO, description="Turn detection strategy")
    eager_eot: bool = Field(True, description="Enable speculative LLM generation on EagerEndOfTurn")
    stop_secs: float = Field(0.2, ge=0.05, le=1.5, description="VAD silence threshold")
    thresholds: Dict[str, Any] = Field(default_factory=dict)


class PipelineLimitsConfig(BaseModel):
    first_token_timeout_ms: int = Field(3000, ge=500, le=15000, description="Max wait for first LLM token before fallback")
    max_call_seconds: int = Field(600, ge=30, le=3600, description="Max session duration")


class CascadedPipelineSpec(BaseModel):
    """Configuration spec for Engine B Cascaded Voice Pipeline."""
    stt: STTPipelineConfig = Field(default_factory=STTPipelineConfig)
    llm: LLMPipelineConfig = Field(default_factory=LLMPipelineConfig)
    tts: TTSPipelineConfig = Field(default_factory=TTSPipelineConfig)
    turn: TurnPipelineConfig = Field(default_factory=TurnPipelineConfig)
    fallbacks: Dict[str, List[str]] = Field(default_factory=dict, description="Fallback providers for LLM / TTS")
    limits: PipelineLimitsConfig = Field(default_factory=PipelineLimitsConfig)


class TurnMetrics(BaseModel):
    """Detailed per-turn latency instrumentation."""
    eot_ms: float | None = None
    stt_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfa_ms: float | None = None
    voice_to_voice_ms: float | None = None
