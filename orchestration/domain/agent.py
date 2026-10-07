"""
Pure domain entities for Voice Agents, Immutable Versions, and Presets.
Adheres strictly to the 6-field Lean Agent architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class AgentStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class VoicePreset:
    """Official NVIDIA PersonaPlex 7B voice conditioning preset."""
    id: str
    name: str
    gender: str
    speaking_style: str
    accent: str = "US"
    recommended_for: str = "General"


# 18 Official PersonaPlex presets derived from NVIDIA model cards
OFFICIAL_PRESETS: dict[str, VoicePreset] = {
    "natural_calm": VoicePreset("natural_calm", "Natural Calm (Elena)", "female", "calm, steady, warm", recommended_for="Customer service, healthcare"),
    "warm_conversational": VoicePreset("warm_conversational", "Warm Conversational (Marcus)", "male", "warm, engaging, conversational", recommended_for="Friendly companion, consultations"),
    "professional_direct": VoicePreset("professional_direct", "Professional Direct (Alex)", "neutral", "clear, articulate, balanced", recommended_for="Technical support, front desk"),
    "empathetic_support": VoicePreset("empathetic_support", "Empathetic Support (Sam)", "neutral", "gentle, compassionate, patient", recommended_for="Clinic triage, claims"),
    "enthusiastic_guide": VoicePreset("enthusiastic_guide", "Enthusiastic Guide", "female", "upbeat, dynamic, clear", recommended_for="Concierge, onboarding"),
    "authoritative_brief": VoicePreset("authoritative_brief", "Authoritative Brief", "male", "confident, decisive, crisp", recommended_for="Emergency, announcements"),
    "thoughtful_advisor": VoicePreset("thoughtful_advisor", "Thoughtful Advisor", "male", "measured, reflective, gentle", recommended_for="Advising, education"),
    "lively_host": VoicePreset("lively_host", "Lively Host", "female", "cheerful, bright, rhythmic", recommended_for="Hospitality, events"),
    "composed_specialist": VoicePreset("composed_specialist", "Composed Specialist", "female", "professional, precise, reassuring", recommended_for="Finance, legal intake"),
    "friendly_neighbor": VoicePreset("friendly_neighbor", "Friendly Neighbor", "male", "relaxed, informal, sincere", recommended_for="Local business, outreach"),
    "steady_operator": VoicePreset("steady_operator", "Steady Operator", "neutral", "composed, focused, calm", recommended_for="Dispatch, logistics"),
    "expressive_storyteller": VoicePreset("expressive_storyteller", "Expressive Storyteller", "female", "melodic, nuanced, animated", recommended_for="Entertainment, narrative"),
    "reassuring_nurse": VoicePreset("reassuring_nurse", "Reassuring Nurse", "female", "tender, empathetic, slow", recommended_for="Clinic appointments, check-ins"),
    "brisk_assistant": VoicePreset("brisk_assistant", "Brisk Assistant", "male", "swift, efficient, respectful", recommended_for="Fast appointments, reminders"),
    "mellow_companion": VoicePreset("mellow_companion", "Mellow Companion", "neutral", "soft, peaceful, soothing", recommended_for="Evening wellness, meditation"),
    "inquisitive_interviewer": VoicePreset("inquisitive_interviewer", "Inquisitive Interviewer", "female", "curious, attentive, warm", recommended_for="Surveys, intake questions"),
    "patient_tutor": VoicePreset("patient_tutor", "Patient Tutor", "male", "encouraging, articulate, patient", recommended_for="Training, instructions"),
    "crisp_coordinator": VoicePreset("crisp_coordinator", "Crisp Coordinator", "neutral", "structured, prompt, polite", recommended_for="Scheduling, calendar sync"),
}


@dataclass
class AgentVersion:
    """Immutable snapshot of an agent configuration."""
    version_id: str
    agent_id: str
    version_number: int
    name: str
    voice_id: str
    greeting: str
    agent_speaks_first: bool
    system_prompt: str
    ending: str
    end_call_timeout_sec: float
    silence_timeout_sec: float
    max_duration_sec: float
    timezone_str: str
    compiled_prompt: str
    token_count: int
    engine: str = "personaplex_s2s"
    language: str = "en"
    pipeline_json: str = "{}"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    change_note: str | None = None


@dataclass
class Agent:
    """Core domain aggregate representing a Voice Agent."""
    id: str
    name: str
    voice_id: str
    greeting: str
    agent_speaks_first: bool
    system_prompt: str
    ending: str
    end_call_timeout_sec: float
    silence_timeout_sec: float
    max_duration_sec: float
    timezone_str: str
    engine: str = "personaplex_s2s"
    language: str = "en"
    pipeline_json: str = "{}"
    status: AgentStatus = AgentStatus.DRAFT
    current_version: int = 1
    published_version: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def validate_timezone(self) -> ZoneInfo:
        """Validate that the configured timezone string is a valid IANA timezone."""
        try:
            return ZoneInfo(self.timezone_str)
        except (ZoneInfoNotFoundError, ValueError) as err:
            raise ValueError(f"Invalid IANA timezone '{self.timezone_str}'") from err

    def validate_voice(self) -> VoicePreset | None:
        """Validate that the configured voice preset exists in the official catalog."""
        if self.engine == "cascaded_cloud":
            return None
        if self.voice_id not in OFFICIAL_PRESETS:
            raise ValueError(
                f"Unknown voice '{self.voice_id}'. Choose from: {list(OFFICIAL_PRESETS.keys())}"
            )
        return OFFICIAL_PRESETS[self.voice_id]
