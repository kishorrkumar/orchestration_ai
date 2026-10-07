"""
Typed, single-source-of-truth settings module for PersonaPlex Voice Agent Platform.
Backed by pydantic-settings, loading environment variables, .env, and agent.yaml.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("orchestration.settings")

# Root constants guaranteed bit-exact with NVIDIA PersonaPlex / Kyutai Moshi upstream
MODEL_SAMPLE_RATE: int = 24000
MODEL_FRAME_SAMPLES: int = 1920
MODEL_FRAME_RATE_HZ: float = 12.5
MODEL_FRAME_MS: float = 80.0


class AgentConfig(BaseModel):
    """Configuration for the single conversational agent loaded from agent.yaml."""
    name: str = "Alex"
    timezone: str = "Asia/Kolkata"
    voice_prompt: str = "NATF2.pt"
    greeting_mode: str = "agent_first"
    greeting_text: str = "Hello! This is Alex from {{company}}. How can I help you today?"
    ending_text: str = "Thank you for calling. Have a great day, goodbye!"
    system_prompt: str = (
        "You are Alex, a helpful and friendly voice assistant at {{company}}. "
        "You are speaking with {{caller_name}} over the phone. "
        "You speak in short, natural sentences, the way real people converse. "
        "You listen carefully, answer concisely, and verify understanding before moving to the next point."
    )
    variables: dict[str, str] = Field(
        default_factory=lambda: {
            "company": "BrightNet",
            "caller_name": "the caller",
            "customer_name": "the customer",
        }
    )
    # Generation settings
    audio_temperature: float = 0.8
    text_temperature: float = 0.7
    audio_topk: int = 250
    text_topk: int = 25
    seed: int | None = -1

    # Session limits
    max_duration_sec: int = 600
    end_silence_sec: int = 20


def find_agent_yaml_path(custom_path: str | Path | None = None) -> Path:
    """Finds agent.yaml, falling back to agent.example.yaml if agent.yaml does not exist."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        Path("agent.yaml"),
        repo_root / "agent.yaml",
        Path("agent.example.yaml"),
        repo_root / "agent.example.yaml",
    ])

    for p in candidates:
        if p.is_file():
            return p

    return repo_root / "agent.example.yaml"


def load_agent_yaml(yaml_path: Path) -> AgentConfig:
    """Loads agent configuration from YAML file."""
    if not yaml_path.is_file():
        logger.warning(f"Agent YAML file not found at {yaml_path}, using built-in defaults.")
        return AgentConfig()

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Unpack nested structures if present
        gen = data.get("generation", {})
        sess = data.get("session", {})

        agent_kwargs: dict[str, Any] = {
            "name": data.get("name", "Alex"),
            "timezone": data.get("timezone", "Asia/Kolkata"),
            "voice_prompt": data.get("voice_prompt", "NATF2.pt"),
            "greeting_mode": data.get("greeting_mode", "agent_first"),
            "greeting_text": data.get("greeting_text", ""),
            "ending_text": data.get("ending_text", ""),
            "system_prompt": data.get("system_prompt", ""),
            "variables": data.get("variables", {}),
            "audio_temperature": float(gen.get("audio_temperature", data.get("audio_temperature", 0.8))),
            "text_temperature": float(gen.get("text_temperature", data.get("text_temperature", 0.7))),
            "audio_topk": int(gen.get("audio_topk", data.get("audio_topk", 250))),
            "text_topk": int(gen.get("text_topk", data.get("text_topk", 25))),
            "seed": gen.get("seed", data.get("seed", -1)),
            "max_duration_sec": int(sess.get("max_duration_sec", data.get("max_duration_sec", 600))),
            "end_silence_sec": int(sess.get("end_silence_sec", data.get("end_silence_sec", 20))),
        }
        return AgentConfig(**agent_kwargs)
    except Exception as e:
        logger.error(f"Error loading agent YAML from {yaml_path}: {e}")
        return AgentConfig()


class Settings(BaseSettings):
    """
    Central typed settings module for the PersonaPlex real-time voice orchestrator.
    Loads from .env, environment variables, and agent.yaml.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Model & DSP Invariants
    SAMPLE_RATE: int = Field(default=MODEL_SAMPLE_RATE, description="Model Mimi sample rate (Hz)")
    FRAME_SAMPLES: int = Field(default=MODEL_FRAME_SAMPLES, description="Audio samples per 12.5Hz frame")
    FRAME_RATE_HZ: float = Field(default=MODEL_FRAME_RATE_HZ, description="Audio frame rate in Hz")
    FRAME_MS: float = Field(default=MODEL_FRAME_MS, description="Duration per frame in milliseconds")

    # 2. Client & Transport Formats
    CLIENT_SAMPLE_RATE: int = Field(default=16000, description="Web client browser audio context sample rate (Hz)")
    CLIENT_CAPTURE_RATE: int = Field(default=16000, description="Browser mic capture sample rate (Hz)")
    CLIENT_CODEC: str = Field(default="pcm16", description="Client audio codec: 'pcm16' or 'g711_ulaw'")
    WIRE_ENCODING: str = Field(default="ogg_opus", description="Wire encoding between Gateway and PersonaPlex worker")

    # 3. Network & Endpoints
    WORKER_HOST: str = Field(default="127.0.0.1", description="PersonaPlex S2S worker host")
    WORKER_PORT: int = Field(default=8998, description="PersonaPlex S2S worker port")
    GATEWAY_HOST: str = Field(default="0.0.0.0", description="Gateway bind host")
    GATEWAY_PORT: int = Field(default=8000, description="Gateway bind port")
    AUTH_TOKEN: str | None = Field(default=None, description="Optional bearer token for REST and WebSocket authentication")

    # 4. Latency & Buffers
    JITTER_BUFFER_MS: int = Field(default=120, description="Target playback jitter buffer size in milliseconds")
    CONNECT_TIMEOUT_SEC: float = Field(default=15.0, description="TCP/WebSocket connect timeout to worker in seconds")
    HANDSHAKE_TIMEOUT_SEC: float = Field(default=60.0, description="Handshake/priming wait timeout in seconds")

    # 5. Pre-Warming & Workers
    ENABLE_PREWARM: bool = Field(default=True, description="Pre-warm and prime an idle worker on page load")
    PREWARM_IDLE_TIMEOUT_SEC: float = Field(default=120.0, description="Idle release timeout for pre-warmed worker")

    # 6. Configuration Paths
    AGENT_YAML_PATH: str = Field(default="agent.yaml", description="Path to agent configuration YAML file")

    # Active Agent Configuration instance
    agent: AgentConfig = Field(default_factory=AgentConfig)

    def reload_agent(self, yaml_path: str | Path | None = None) -> None:
        """Reloads agent configuration from disk."""
        target_path = find_agent_yaml_path(yaml_path or self.AGENT_YAML_PATH)
        self.agent = load_agent_yaml(target_path)
        self.AGENT_YAML_PATH = str(target_path)


def get_settings(yaml_override: str | Path | None = None) -> Settings:
    """Instantiates Settings and attaches loaded agent config."""
    s = Settings()
    yaml_path = find_agent_yaml_path(yaml_override or s.AGENT_YAML_PATH)
    s.agent = load_agent_yaml(yaml_path)
    s.AGENT_YAML_PATH = str(yaml_path)
    return s


def validate_startup_settings(s: Settings) -> None:
    """
    Validates all critical system invariants. Fails fast with clear ValueError if invalid.
    """
    # 1. Protocol Invariants
    if s.SAMPLE_RATE != 24000:
        raise ValueError(f"SAMPLE_RATE must be exactly 24000 (Mimi codec invariant). Got {s.SAMPLE_RATE}")
    if s.FRAME_SAMPLES != 1920:
        raise ValueError(f"FRAME_SAMPLES must be exactly 1920 (80ms at 24kHz). Got {s.FRAME_SAMPLES}")
    if abs(s.FRAME_RATE_HZ - 12.5) > 1e-4:
        raise ValueError(f"FRAME_RATE_HZ must be exactly 12.5. Got {s.FRAME_RATE_HZ}")
    if abs(s.FRAME_MS - 80.0) > 1e-4:
        raise ValueError(f"FRAME_MS must be exactly 80.0. Got {s.FRAME_MS}")
    if s.CLIENT_SAMPLE_RATE not in (16000, 8000, 24000):
        raise ValueError(f"CLIENT_SAMPLE_RATE must be 16000 (web) or 8000 (telephony). Got {s.CLIENT_SAMPLE_RATE}")

    # 2. Agent Configuration
    if not s.agent.system_prompt.strip():
        raise ValueError("Agent system_prompt cannot be empty in agent.yaml")
    if not s.agent.voice_prompt.strip():
        raise ValueError("Agent voice_prompt cannot be empty in agent.yaml")

    # 3. Check for unresolved template variables in agent.yaml
    full_prompt_text = f"{s.agent.system_prompt} {s.agent.greeting_text} {s.agent.ending_text}"
    placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", full_prompt_text))
    defined_vars = set(s.agent.variables.keys())

    # Standard dynamic variables provided by the system
    system_vars = {"caller_name", "customer_name", "phone_number", "agent_name", "current_time", "weekday", "date", "day_part", "time_line"}
    unresolved = placeholders - defined_vars - system_vars
    if unresolved:
        raise ValueError(
            f"Startup validation failed: Unresolved template variable(s) {sorted(list(unresolved))} in agent.yaml. "
            f"Define defaults under 'variables:' in agent.yaml."
        )


# Global singleton instance
app_settings = get_settings()
