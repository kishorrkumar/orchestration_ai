"""Command and DTO objects for Agent application use cases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CreateAgentCommand:
    name: str
    voice_id: str
    greeting: str
    system_prompt: str
    ending: str
    agent_speaks_first: bool = True
    end_call_timeout_sec: float = 2.0
    silence_timeout_sec: float = 12.0
    max_duration_sec: float = 600.0
    timezone_str: str = "UTC"


@dataclass
class UpdateAgentCommand:
    name: str | None = None
    voice_id: str | None = None
    greeting: str | None = None
    agent_speaks_first: bool | None = None
    system_prompt: str | None = None
    ending: str | None = None
    end_call_timeout_sec: float | None = None
    silence_timeout_sec: float | None = None
    max_duration_sec: float | None = None
    timezone_str: str | None = None
    change_note: str | None = None


@dataclass
class PublishVersionCommand:
    change_note: str | None = None
