"""Pydantic v2 schemas for prompt compiling and linting."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LintWarningResponse(BaseModel):
    rule: str
    message: str
    severity: str
    match: str | None = None


class CompilePromptRequest(BaseModel):
    system_prompt: str = Field(..., json_schema_extra={"example": "You are a helpful voice assistant."})
    greeting: str = Field(default="", json_schema_extra={"example": "Hello, how can I help you today?"})
    ending: str = Field(default="", json_schema_extra={"example": "Goodbye, take care!"})
    agent_name: str = Field(default="Agent", json_schema_extra={"example": "Elena"})
    customer_name: str = Field(default="Friend", json_schema_extra={"example": "Alex"})
    company: str = Field(default="PersonaPlex", json_schema_extra={"example": "Acme Health"})
    timezone_str: str = Field(default="UTC", json_schema_extra={"example": "America/New_York"})


class CompilePromptResponse(BaseModel):
    compiled_text: str
    wrapped_text: str
    token_count: int
    hard_limit: int
    recommended_limit: int
    is_within_hard_limit: bool
    is_fast_start: bool
    local_time_line: str
    warnings: list[LintWarningResponse]
