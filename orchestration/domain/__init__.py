"""Domain layer for Voice Agents, Immutable Versions, Call Sessions, and Delimiters."""

from .agent import OFFICIAL_PRESETS, Agent, AgentStatus, AgentVersion, VoicePreset
from .detector import DetectionAction, DetectionResult, EndOfCallDetector
from .prompt import (
    HARD_TOKEN_LIMIT,
    RECOMMENDED_TOKEN_LIMIT,
    SYSTEM_TAG_CLOSE,
    SYSTEM_TAG_OPEN,
    CompiledPrompt,
    LintWarning,
    PromptLinter,
    compute_local_time_context,
    wrap_system_prompt,
)
from .protocols import (
    AgentRepository,
    CallSessionRepository,
    Clock,
    SpeechToSpeechEngine,
    Tokenizer,
)
from .session import CallSession, CallStatus, CallTurn, EndReason, TurnSpeaker

__all__ = [
    "HARD_TOKEN_LIMIT",
    "OFFICIAL_PRESETS",
    "RECOMMENDED_TOKEN_LIMIT",
    "SYSTEM_TAG_CLOSE",
    "SYSTEM_TAG_OPEN",
    "Agent",
    "AgentRepository",
    "AgentStatus",
    "AgentVersion",
    "CallSession",
    "CallSessionRepository",
    "CallStatus",
    "CallTurn",
    "Clock",
    "CompiledPrompt",
    "DetectionAction",
    "DetectionResult",
    "EndOfCallDetector",
    "EndReason",
    "LintWarning",
    "PromptLinter",
    "SpeechToSpeechEngine",
    "Tokenizer",
    "TurnSpeaker",
    "VoicePreset",
    "compute_local_time_context",
    "wrap_system_prompt",
]
