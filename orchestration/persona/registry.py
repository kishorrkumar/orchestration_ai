"""
Persona and Voice conditioning registry for PersonaPlex.

Manages:
- 18 official PersonaPlex voice presets (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
- Text prompt wrapping with '<system> ... <system>' tags
- Declarative PersonaConfig models
- Persona catalog with standard pre-packaged roles
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from .prompts import build_system_prompt, MASTER_VOICE_AGENT_SYSTEM_PROMPT

# 18 official PersonaPlex voice preset IDs
OFFICIAL_VOICE_PRESETS = [
    # Natural Female
    "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
    # Natural Male
    "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
    # Variety Female
    "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
    # Variety Male
    "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
]


def wrap_with_system_tags(text: str) -> str:
    """
    Format text prompt with the system delimiters expected by the PersonaPlex LM:
    '<system> {text} <system>'
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


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
    description: str = Field(default="", description="Description of the role")
    accent: str = Field(default="Indian English", description="Accent (Indian English)")
    character: str = Field(default="Professional", description="Character (Professional, Friendly & Funny)")
    voice_prompt: str = Field(default="NATM0.pt", description="Voice embedding (.pt) or audio reference (.wav)")
    neural_voice: str = Field(default="en-IN-PrabhatNeural", description="Open-source neural TTS voice model (e.g. en-IN-PrabhatNeural, en-IN-NeerjaExpressiveNeural)")
    text_prompt: str = Field(..., description="Behavioral instructions and persona facts")
    audio_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    text_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k_audio: int = Field(default=250, ge=1)
    top_k_text: int = Field(default=25, ge=1)
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic generation")

    def get_formatted_text_prompt(self) -> str:
        """Returns the text prompt properly delimited with <system> tags."""
        return wrap_with_system_tags(self.text_prompt)

    def get_normalized_voice_prompt(self) -> str:
        """Returns the normalized voice prompt filename."""
        return normalize_voice_name(self.voice_prompt)


class PersonaRegistry:
    """Thread-safe registry of agent personas and voice profiles."""

    def __init__(self) -> None:
        self._personas: Dict[str, PersonaConfig] = {}
        self._register_default_personas()

    def _register_default_personas(self) -> None:
        # Option 1: Indian English • Professional (Aarav)
        self.register(
            PersonaConfig(
                id="indian_pro",
                name="Aarav (Indian English • Professional)",
                description="Articulate, polite, structured, business-oriented Indian English speaker.",
                accent="Indian English",
                character="Professional",
                voice_prompt="NATM0.pt",
                neural_voice="en-IN-PrabhatNeural",
                text_prompt=build_system_prompt("indian", "professional"),
            )
        )
        # Option 2: Indian English • Friendly & Funny (Rohan)
        self.register(
            PersonaConfig(
                id="indian_funny",
                name="Rohan (Indian English • Friendly & Funny)",
                description="Witty, warm, charismatic, cheerful Indian English conversational companion.",
                accent="Indian English",
                character="Friendly & Funny",
                voice_prompt="NATM3.pt",
                neural_voice="en-IN-NeerjaExpressiveNeural",
                text_prompt=build_system_prompt("indian", "funny"),
            )
        )

    def register(self, config: PersonaConfig) -> None:
        self._personas[config.id] = config

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        return self._personas.get(persona_id)

    def list_all(self) -> List[PersonaConfig]:
        return list(self._personas.values())

    def delete(self, persona_id: str) -> bool:
        if persona_id in self._personas:
            del self._personas[persona_id]
            return True
        return False

    @staticmethod
    def get_official_voices() -> List[str]:
        return list(OFFICIAL_VOICE_PRESETS)


# Global singleton registry
default_registry = PersonaRegistry()
