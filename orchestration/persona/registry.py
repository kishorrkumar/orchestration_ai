"""
Persona and Voice conditioning registry for PersonaPlex.

Manages:
- 18 official PersonaPlex voice presets (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
- Text prompt sanitization, length validation, and '<system> ... <system>' wrapping
- Declarative PersonaConfig models with YAML/JSON serialization
- Built-in production personas with colloquial dialogue guidance
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .prompts import build_system_prompt

# 18 official PersonaPlex voice preset IDs
OFFICIAL_VOICE_PRESETS = [
    # Natural Female (4)
    "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
    # Natural Male (4)
    "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
    # Variety Female (5)
    "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
    # Variety Male (5)
    "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
]

import os
from pathlib import Path


def get_existing_voice_files() -> list[str]:
    """Return all voice preset filenames on disk, unioned with official presets."""
    dirs_to_check = [
        Path("voices"),
        Path(os.environ.get("HF_HOME", "/workspace/huggingface")) / "voices",
        Path.home() / ".cache" / "huggingface" / "voices",
        Path("data/cloned_voices"),
    ]
    voices = set(OFFICIAL_VOICE_PRESETS)
    for d in dirs_to_check:
        if d.is_dir():
            for p in d.glob("*.pt"):
                voices.add(p.name)
            for p in d.glob("*.wav"):
                voices.add(p.name)
            for p in d.glob("*/*.wav"):
                voices.add(p.name)
            for p in d.glob("*/*.pt"):
                voices.add(p.name)
    return sorted(voices)


PRESET_METADATA = {
    "NATF0.pt": {"gender": "female", "style": "natural", "description": "Natural conversational female (balanced, articulate)"},
    "NATF1.pt": {"gender": "female", "style": "natural", "description": "Natural conversational female (warm, empathetic)"},
    "NATF2.pt": {"gender": "female", "style": "natural", "description": "Natural conversational female (clear, instructive, authoritative)"},
    "NATF3.pt": {"gender": "female", "style": "natural", "description": "Natural conversational female (casual, relaxed)"},
    "NATM0.pt": {"gender": "male", "style": "natural", "description": "Natural conversational male (relaxed, friendly)"},
    "NATM1.pt": {"gender": "male", "style": "natural", "description": "Natural conversational male (upbeat, dynamic, consultative)"},
    "NATM2.pt": {"gender": "male", "style": "natural", "description": "Natural conversational male (deep, calm, confident)"},
    "NATM3.pt": {"gender": "male", "style": "natural", "description": "Natural conversational male (expressive, lively)"},
    "VARF0.pt": {"gender": "female", "style": "variety", "description": "Variety female profile 0 (distinct pitch/cadence)"},
    "VARF1.pt": {"gender": "female", "style": "variety", "description": "Variety female profile 1 (bright, energetic)"},
    "VARF2.pt": {"gender": "female", "style": "variety", "description": "Variety female profile 2 (mature, reassuring)"},
    "VARF3.pt": {"gender": "female", "style": "variety", "description": "Variety female profile 3 (youthful, crisp)"},
    "VARF4.pt": {"gender": "female", "style": "variety", "description": "Variety female profile 4 (intimate, conversational)"},
    "VARM0.pt": {"gender": "male", "style": "variety", "description": "Variety male profile 0 (textured, casual)"},
    "VARM1.pt": {"gender": "male", "style": "variety", "description": "Variety male profile 1 (commanding, radio)"},
    "VARM2.pt": {"gender": "male", "style": "variety", "description": "Variety male profile 2 (mellow, patient)"},
    "VARM3.pt": {"gender": "male", "style": "variety", "description": "Variety male profile 3 (upbeat, tech-savvy)"},
    "VARM4.pt": {"gender": "male", "style": "variety", "description": "Variety male profile 4 (resonant, friendly)"},
}


def sanitize_system_prompt(text: str) -> str:
    """
    Sanitize text prompt:
    - Strips control characters
    - Removes internal <system> or </system> injections
    - Normalizes multi-whitespace
    """
    if not text:
        return ""
    # Strip null bytes and non-printable control chars except \n and \t
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    # Strip existing external system tags if present before inner cleaning
    cleaned = cleaned.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        cleaned = cleaned[8:-8].strip()
    elif cleaned.startswith("<system>") and cleaned.endswith("</system>"):
        cleaned = cleaned[8:-9].strip()
    # Strip any accidental inner system tags
    cleaned = re.sub(r"</?system>", " ", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def estimate_token_count(text: str) -> int:
    """Fast conservative approximation of SentencePiece tokens (avg 1.3 tokens per word)."""
    words = text.split()
    return max(1, int(len(words) * 1.35))


def validate_system_prompt(text: str, max_tokens: int | None = None) -> str:
    """
    Validates that system prompt fits within safe latency bounds.
    Upstream steps each token sequentially during startup; long prompts cause connection timeouts.
    """
    cleaned = sanitize_system_prompt(text)
    if not cleaned:
        raise ValueError("System prompt cannot be empty.")
    if max_tokens is not None:
        tokens = estimate_token_count(cleaned)
        if tokens > max_tokens:
            raise ValueError(
                f"System prompt too long ({tokens} estimated tokens > {max_tokens} max). "
                "Please shorten to avoid initialization timeouts."
            )
    return cleaned


def wrap_with_system_tags(text: str, max_tokens: int | None = None) -> str:
    """
    Format text prompt with the exact system delimiters expected by PersonaPlex:
    '<system> {text} <system>'
    """
    sanitized = validate_system_prompt(text, max_tokens=max_tokens)
    return f"<system> {sanitized} <system>"


def normalize_voice_name(voice_name: str) -> str:
    """Normalize voice name by ensuring .pt extension if matching an official preset."""
    candidate = voice_name.strip()
    if not candidate.endswith(".pt") and not candidate.endswith(".wav"):
        if f"{candidate}.pt" in OFFICIAL_VOICE_PRESETS:
            return f"{candidate}.pt"
    return candidate


class PersonaConfig(BaseModel):
    id: str = Field(..., description="Unique identifier for the agent persona")
    name: str = Field(..., description="Display name of the agent")
    description: str = Field(default="", description="Description of the role and personality")
    system_prompt: str = Field(default="", description="Behavioral instructions and persona facts")
    voice_ref: str = Field(default="NATF2.pt", description="Voice embedding (.pt) or audio reference (.wav)")
    gender: str = Field(default="female", description="Gender of the voice: female, male, neutral")
    speaking_style: str = Field(default="colloquial", description="Speaking style: colloquial, warm, formal, witty")
    language: str = Field(default="en", description="Primary language (PersonaPlex is English-only)")
    opening_behavior: str = Field(
        default="wait_for_user",
        description="Opening turn policy: wait_for_user, speak_first, or brief_greeting"
    )
    accent: str = Field(default="American English", description="Accent description")
    character: str = Field(default="Conversational", description="Character archetype")
    neural_voice: str = Field(default="NATF2.pt", description="Neural voice identifier")
    text_prompt: str | None = Field(default=None, description="Legacy alias for system_prompt")
    voice_prompt: str | None = Field(default=None, description="Legacy alias for voice_ref")
    audio_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    text_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k_audio: int = Field(default=250, ge=1)
    top_k_text: int = Field(default=25, ge=1)
    seed: int | None = Field(default=None, description="Optional seed for deterministic generation")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    llm_model: str = Field(default="qwen2.5:1.5b")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    call_flow: str = Field(default="conversational_companion")

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Synchronize system_prompt <-> text_prompt
            if "system_prompt" not in data or not data["system_prompt"]:
                if data.get("text_prompt"):
                    data["system_prompt"] = data["text_prompt"]
            if "text_prompt" not in data or not data["text_prompt"]:
                data["text_prompt"] = data.get("system_prompt", "")

            # Synchronize voice_ref <-> voice_prompt
            if "voice_ref" not in data or not data["voice_ref"]:
                if data.get("voice_prompt"):
                    data["voice_ref"] = data["voice_prompt"]
            if "voice_prompt" not in data or not data["voice_prompt"]:
                data["voice_prompt"] = data.get("voice_ref", "NATF2.pt")
        return data

    def get_formatted_text_prompt(self, max_tokens: int | None = None) -> str:
        """Returns the system prompt properly delimited with <system> tags."""
        content = self.system_prompt or self.text_prompt or ""
        return wrap_with_system_tags(content, max_tokens=max_tokens)

    def get_normalized_voice_prompt(self) -> str:
        """Returns the normalized voice prompt filename."""
        v = self.voice_ref or self.voice_prompt or "NATF2.pt"
        return normalize_voice_name(v)


class PersonaRegistry:
    """Thread-safe catalog of agent personas and voice profiles."""

    def __init__(self) -> None:
        self._personas: dict[str, PersonaConfig] = {}
        self._register_default_personas()

    def _register_default_personas(self) -> None:
        # Persona 1: Friendly Support Agent (Alex)
        self.register(
            PersonaConfig(
                id="support_agent",
                name="Alex (Support Agent • Female)",
                description="Helpful, patient, conversational customer specialist with colloquial phrasing.",
                gender="female",
                voice_ref="NATF1.pt",
                speaking_style="colloquial, empathetic",
                language="en",
                opening_behavior="speak_first",
                system_prompt=(
                    "You are Alex, a helpful and patient support specialist. Speak casually in short, natural turns. "
                    "Use contractions like 'I'll', 'don't', and 'we've'. Acknowledge the user with brief conversational "
                    "backchannels like 'right', 'got it', and 'makes sense'. Never give long speeches or bulleted essays. "
                    "If you don't know something, ask directly."
                ),
            )
        )

        # Persona 2: Wise Teacher (Dr. Elena)
        self.register(
            PersonaConfig(
                id="wise_teacher",
                name="Dr. Elena (Teacher • Female)",
                description="Wise, engaging educator explaining concepts through intuitive analogies.",
                gender="female",
                voice_ref="NATF2.pt",
                speaking_style="curious, encouraging",
                language="en",
                opening_behavior="wait_for_user",
                system_prompt=(
                    "You are Dr. Elena, a wise and friendly teacher who loves explaining complex ideas with simple, "
                    "everyday analogies. Keep your answers brief, engaging, and dialogue-driven. Ask a quick clarifying "
                    "question to see what the user already knows. Use contractions and casual phrasing, avoiding academic jargon."
                ),
            )
        )

        # Persona 3: Consultative Sales Caller (Marcus)
        self.register(
            PersonaConfig(
                id="sales_caller",
                name="Marcus (Sales Specialist • Male)",
                description="Energetic, charismatic advisor who listens first and engages with short questions.",
                gender="male",
                voice_ref="NATM1.pt",
                speaking_style="upbeat, consultative",
                language="en",
                opening_behavior="speak_first",
                system_prompt=(
                    "You are Marcus, a consultative sales advisor. You're upbeat, quick to listen, and speak naturally. "
                    "Use conversational phrases like 'totally get that', 'honestly', and 'let's see'. Keep each turn under "
                    "two sentences and invite the customer's input. Never sound scripted, robotic, or pushy."
                ),
            )
        )

        # Persona 4: Casual Friend (Sam)
        self.register(
            PersonaConfig(
                id="casual_friend",
                name="Sam (Casual Friend • Male)",
                description="Easygoing friend with informal banter, laughter, and quick turn-taking.",
                gender="male",
                voice_ref="NATM0.pt",
                speaking_style="casual, relaxed",
                language="en",
                opening_behavior="wait_for_user",
                system_prompt=(
                    "You enjoy having a good conversation. You're Sam, an easygoing and witty friend having a chat over coffee. "
                    "You speak informally with natural pauses and conversational fillers like 'well', 'you know', and 'yeah'. "
                    "Share quick thoughts and banter back and forth, keeping each reply punchy and alive."
                ),
            )
        )

        # Persona 5: Aarav (Colloquial Indian English • Male)
        self.register(
            PersonaConfig(
                id="indian_pro",
                name="Aarav (Colloquial Indian English • Male)",
                description="Articulate, natural, conversational Indian English speaker with relaxed cadence.",
                gender="male",
                accent="Indian English",
                character="Professional",
                voice_ref="NATM0.pt",
                neural_voice="en-IN-PrabhatNeural",
                speaking_style="colloquial, articulate",
                language="en",
                system_prompt=build_system_prompt("indian", "professional"),
            )
        )

        # Persona 6: Priya (Colloquial Indian English • Female)
        self.register(
            PersonaConfig(
                id="indian_priya",
                name="Priya (Colloquial Indian English • Female)",
                description="Warm, bright, empathetic, and colloquial Indian English speaker.",
                gender="female",
                accent="Indian English",
                character="Warm",
                voice_ref="NATF0.pt",
                neural_voice="priya_colloquial",
                speaking_style="warm, empathetic",
                language="en",
                system_prompt=build_system_prompt("indian", "warm"),
            )
        )

        # Persona 7: Kabir (Colloquial Indian English • Friendly & Funny)
        self.register(
            PersonaConfig(
                id="indian_funny",
                name="Kabir (Colloquial Indian English • Friendly & Funny)",
                description="Witty, warm, charismatic Indian English conversational companion.",
                gender="male",
                accent="Indian English",
                character="Friendly & Funny",
                voice_ref="NATM3.pt",
                neural_voice="en-IN-NeerjaExpressiveNeural",
                speaking_style="witty, charismatic",
                language="en",
                system_prompt=build_system_prompt("indian", "funny"),
            )
        )

    def register(self, config: PersonaConfig) -> None:
        self._personas[config.id] = config

    def get(self, persona_id: str) -> PersonaConfig | None:
        return self._personas.get(persona_id)

    def list_all(self) -> list[PersonaConfig]:
        return list(self._personas.values())

    def delete(self, persona_id: str) -> bool:
        if persona_id in self._personas:
            del self._personas[persona_id]
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {pid: p.model_dump() for pid, p in self._personas.items()}

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.to_dict(), sort_keys=False)

    def load_from_yaml(self, yaml_str: str) -> None:
        import yaml
        raw = yaml.safe_load(yaml_str)
        if isinstance(raw, dict):
            for pid, cfg_data in raw.items():
                if isinstance(cfg_data, dict):
                    cfg_data["id"] = cfg_data.get("id", pid)
                    self.register(PersonaConfig(**cfg_data))

    @staticmethod
    def get_official_voices() -> list[str]:
        return list(OFFICIAL_VOICE_PRESETS)


# Global default singleton registry
default_registry = PersonaRegistry()
