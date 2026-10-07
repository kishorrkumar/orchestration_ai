"""Pydantic v2 DTO schemas for Agent REST endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VoicePresetResponse(BaseModel):
    id: str
    name: str
    gender: str
    speaking_style: str
    accent: str
    recommended_for: str


class CreateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, max_length=128, json_schema_extra={"example": "Friendly Caller"})
    voice_id: str = Field(..., json_schema_extra={"example": "natural_calm"})
    greeting: str = Field(..., json_schema_extra={"example": "Hi {{customer_name}}, thanks for calling! How can I help you today?"})
    agent_speaks_first: bool = Field(default=True, json_schema_extra={"example": True})
    system_prompt: str = Field(..., min_length=1, json_schema_extra={"example": "You are {{agent_name}}, a warm and helpful voice assistant."})
    ending: str = Field(..., json_schema_extra={"example": "Alright, thanks so much for your time. Take care, bye!"})
    end_call_timeout_sec: float = Field(default=2.0, ge=0.5, le=10.0, json_schema_extra={"example": 2.0})
    silence_timeout_sec: float = Field(default=12.0, ge=3.0, le=60.0, json_schema_extra={"example": 12.0})
    max_duration_sec: float = Field(default=600.0, ge=10.0, le=3600.0, json_schema_extra={"example": 600.0})
    timezone_str: str = Field(default="UTC", json_schema_extra={"example": "America/New_York"})
    engine: str = Field(default="personaplex_s2s", json_schema_extra={"example": "personaplex_s2s"})
    language: str = Field(default="en", json_schema_extra={"example": "en"})
    pipeline_json: str = Field(default="{}", json_schema_extra={"example": "{}"})


class UpdateAgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    voice_id: str | None = None
    greeting: str | None = None
    agent_speaks_first: bool | None = None
    system_prompt: str | None = None
    ending: str | None = None
    end_call_timeout_sec: float | None = Field(default=None, ge=0.5, le=10.0)
    silence_timeout_sec: float | None = Field(default=None, ge=3.0, le=60.0)
    max_duration_sec: float | None = Field(default=None, ge=10.0, le=3600.0)
    timezone_str: str | None = None
    engine: str | None = None
    language: str | None = None
    pipeline_json: str | None = None
    change_note: str | None = Field(default=None, max_length=256)


class PublishVersionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    change_note: str | None = Field(default=None, max_length=256, json_schema_extra={"example": "Production release v2"})


class AgentResponse(BaseModel):
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
    status: str
    current_version: int
    published_version: int | None
    created_at: datetime
    updated_at: datetime


class AgentVersionResponse(BaseModel):
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
    engine: str = "personaplex_s2s"
    language: str = "en"
    pipeline_json: str = "{}"
    compiled_prompt: str
    token_count: int
    created_at: datetime
    change_note: str | None
