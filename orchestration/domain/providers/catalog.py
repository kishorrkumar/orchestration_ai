"""
Domain Models and Validation Schemas for Provider Catalog.
All provider capabilities are data-driven via YAML manifests.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field


class ProviderKind(enum.StrEnum):
    STT = "stt"
    LLM = "llm"
    TTS = "tts"


class AuthFieldSpec(BaseModel):
    name: str = Field(..., description="Field identifier (e.g., api_key, base_url, region)")
    label: str = Field(..., description="Human-readable label for UI")
    field_type: Literal["password", "text", "select"] = Field("password", description="UI input type")
    required: bool = Field(True, description="Whether this field is mandatory")
    placeholder: str | None = Field(None, description="Input placeholder text")
    description: str | None = Field(None, description="Help text under field")
    options: List[str] | None = Field(None, description="Options if field_type is select")
    default_value: str | None = Field(None, description="Default value if not provided")


class ModelDescriptor(BaseModel):
    id: str = Field(..., description="Model identifier used in API calls")
    name: str = Field(..., description="Display name for model")
    description: str | None = Field(None, description="Short summary of strengths")
    context_window: int | None = Field(None, description="Max context tokens")
    latency_profile: str | None = Field(None, description="e.g., ultra-fast, standard")
    last_verified: str | None = Field(None, description="ISO Date string when model was verified")


class VoiceDescriptor(BaseModel):
    id: str = Field(..., description="Voice identifier used in synthesis calls")
    name: str = Field(..., description="Human display name")
    gender: str | None = Field(None, description="female | male | neutral")
    language: str = Field("en-US", description="BCP-47 language tag")
    preview_url: str | None = Field(None, description="Audio sample URL for preview")


class ProviderManifest(BaseModel):
    id: str = Field(..., description="Unique provider slug (e.g. deepgram, openai)")
    kind: ProviderKind = Field(..., description="stt | llm | tts")
    display_name: str = Field(..., description="Human-readable provider name")
    description: str = Field(..., description="One-line summary of strengths & languages")
    docs_url: str = Field(..., description="Official documentation URL")
    auth_fields: List[AuthFieldSpec] = Field(default_factory=list, description="Fields required for authentication")
    supported_languages: List[str] = Field(default_factory=lambda: ["en-US"], description="Supported BCP-47 language codes")
    native_sample_rates: List[int] = Field(default_factory=lambda: [16000], description="Supported audio sample rates")
    streaming: bool = Field(True, description="Whether streaming WebSocket/HTTP is supported")
    native_eot: bool = Field(False, description="Whether STT has native end-of-turn detection (e.g. Deepgram Flux)")
    default_model: str = Field(..., description="Recommended default model ID")
    default_voice: str | None = Field(None, description="Recommended default voice ID for TTS")
    model_list_type: Literal["static", "live_api"] = Field("static", description="Whether models are discovered dynamically")
    models: List[ModelDescriptor] = Field(default_factory=list, description="Available models")
    voices: List[VoiceDescriptor] = Field(default_factory=list, description="Preset voices for TTS")
    pipecat_service: str = Field(..., description="Python class path of Pipecat service adapter")


class ProviderCatalog:
    """Loads, validates, and queries provider YAML manifests."""

    def __init__(self, catalog_dir: Path | str | None = None) -> None:
        if catalog_dir is None:
            # Default to orchestration/providers/catalog
            self.catalog_dir = Path(__file__).resolve().parent.parent.parent / "providers" / "catalog"
        else:
            self.catalog_dir = Path(catalog_dir)
        self._providers: Dict[str, ProviderManifest] = {}
        self.reload()

    def reload(self) -> None:
        """Scan and reload all *.yaml files in the catalog directory."""
        self._providers.clear()
        if not self.catalog_dir.exists():
            return

        for yaml_path in sorted(self.catalog_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                if not data:
                    continue
                manifest = ProviderManifest.model_validate(data)
                self._providers[manifest.id] = manifest
            except Exception as e:
                raise ValueError(f"Failed to load provider manifest {yaml_path.name}: {e}") from e

    def get(self, provider_id: str) -> ProviderManifest | None:
        return self._providers.get(provider_id)

    def list_all(self) -> List[ProviderManifest]:
        return list(self._providers.values())

    def list_by_kind(self, kind: ProviderKind) -> List[ProviderManifest]:
        return [p for p in self._providers.values() if p.kind == kind]
