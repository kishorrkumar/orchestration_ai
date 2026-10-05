"""
Pydantic v2 Request/Response Schemas for Lean S2S Voice Agent Platform.
"""

from __future__ import annotations

import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Agent display name")
    voice_id: str = Field("NATF0.pt", description="One of the 18 PersonaPlex presets")
    greeting_text: str = Field("", description="First spoken greeting")
    greeting_mode: str = Field("agent_first", description="'agent_first' or 'user_first'")
    system_prompt: str = Field("", description="Natural prose system prompt instructions")
    ending_text: str = Field("", description="Goodbye phrase closing the call")
    end_silence_sec: int = Field(20, ge=5, le=120, description="Silence timeout before hangup")
    max_duration_sec: int = Field(600, ge=30, le=3600, description="Maximum call duration in seconds")
    timezone: str = Field("Asia/Kolkata", description="IANA timezone identifier")
    auto_publish: bool = Field(True, description="Immediately create published version 1")


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    voice_id: Optional[str] = None
    greeting_text: Optional[str] = None
    greeting_mode: Optional[str] = None
    system_prompt: Optional[str] = None
    ending_text: Optional[str] = None
    end_silence_sec: Optional[int] = Field(None, ge=5, le=120)
    max_duration_sec: Optional[int] = Field(None, ge=30, le=3600)
    timezone: Optional[str] = None


class AgentPublishRequest(BaseModel):
    change_note: str = Field("", max_length=255, description="Brief note describing changes")


class VersionResponse(BaseModel):
    id: str
    agent_id: str
    version_no: int
    voice_id: str
    greeting_text: str
    greeting_mode: str
    system_prompt: str
    ending_text: str
    end_silence_sec: int
    max_duration_sec: int
    timezone: str
    compiled_token_count: int
    change_note: str
    created_at: datetime.datetime


class AgentResponse(BaseModel):
    id: str
    name: str
    status: str
    published_version_id: Optional[str]
    current_version_no: int
    draft_voice_id: str
    draft_greeting_text: str
    draft_greeting_mode: str
    draft_system_prompt: str
    draft_ending_text: str
    draft_end_silence_sec: int
    draft_max_duration_sec: int
    draft_timezone: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    versions_count: int = 0


class CompilePromptRequest(BaseModel):
    system_prompt: str
    greeting_text: str = ""
    greeting_mode: str = "agent_first"
    ending_text: str = ""
    agent_name: str = "Assistant"
    timezone: str = "Asia/Kolkata"
    variables: Optional[dict[str, Any]] = None


class CompilePromptResponse(BaseModel):
    compiled_text: str
    token_count: int
    ideal_limit: int
    hard_limit: int
    can_publish: bool
    warnings: List[str]
    unrendered_variables: List[str]
    time_line: str
    day_part: str


class TimezoneOption(BaseModel):
    name: str
    current_time: str
    weekday: str
    day_part: str
