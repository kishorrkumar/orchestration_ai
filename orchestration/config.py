"""
Central configuration layer for PersonaPlex Orchestration Layer.

Supports loading from:
1. Environment variables (highest priority)
2. YAML configuration file (config.yaml)
3. Built-in production defaults
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    host: str = Field(default="0.0.0.0", description="Gateway host bind address")
    port: int = Field(default=8000, description="Gateway port")
    workers: List[str] = Field(
        default_factory=lambda: ["127.0.0.1:8998"],
        description="List of backend worker endpoints (host:port or id:host:port)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for gateway authentication"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum requests/sessions per minute per IP"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )
    cloned_voices_dir: str = Field(
        default="data/cloned_voices",
        description="Persistent disk directory for cloned voice conditioning artifacts"
    )
    personas_dir: str = Field(
        default="personas",
        description="Directory for custom persona definitions"
    )


class PersonaPlexConfig(BaseModel):
    hf_repo: str = Field(
        default="nvidia/personaplex-7b-v1",
        description="Hugging Face model repository"
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face auth token for model weights access"
    )
    hf_home: Optional[str] = Field(
        default=None,
        description="Custom directory for caching HF models on persistent disk"
    )
    device: str = Field(default="cuda", description="Inference device (cuda or cpu)")
    cpu_offload: bool = Field(
        default=False,
        description="Enable CPU offloading via accelerate for low-VRAM GPUs"
    )
    voice_prompt_dir: Optional[str] = Field(
        default=None,
        description="Path to pre-extracted voices directory containing .pt files"
    )
    default_voice: str = Field(
        default="NATF2.pt",
        description="Default voice preset embedding"
    )
    max_system_prompt_tokens: int = Field(
        default=350,
        description="Maximum allowed tokens for system prompt to bound initialization latency"
    )
    min_voice_prompt_duration_sec: float = Field(
        default=3.0,
        description="Minimum duration for voice cloning reference audio"
    )
    max_voice_prompt_duration_sec: float = Field(
        default=15.0,
        description="Maximum duration for voice cloning reference audio"
    )


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=24000, description="Mimi codec sample rate (Hz)")
    frame_rate: float = Field(default=12.5, description="Audio frame rate (Hz)")
    frame_size: int = Field(default=1920, description="Samples per frame (24000 / 12.5)")
    frame_duration_ms: float = Field(default=80.0, description="Duration per frame in ms")
    jitter_buffer_frames: int = Field(default=2, description="Target jitter buffer size in frames")
    loudness_lufs_target: float = Field(default=-24.0, description="Target LUFS for audio normalization")


class LatencyBudget(BaseModel):
    target_ttfa_ms: float = Field(
        default=300.0,
        description="Target p50 time-to-first-audio after user stops speaking (ms)"
    )
    target_frame_processing_ms: float = Field(
        default=80.0,
        description="Maximum frame inference step budget (ms)"
    )


class AppConfig(BaseModel):
    environment: str = Field(default="production", description="Environment: production, staging, development")
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    personaplex: PersonaPlexConfig = Field(default_factory=PersonaPlexConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    latency: LatencyBudget = Field(default_factory=LatencyBudget)


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load AppConfig by cascading:
    1. Built-in defaults
    2. config.yaml (if present)
    3. Environment variable overrides (e.g. HF_TOKEN, GATEWAY_PORT, etc.)
    """
    data: dict = {}
    
    # Check default config.yaml paths
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths.extend([
        Path("config.yaml"),
        Path("config.yml"),
        Path(__file__).resolve().parent.parent / "config.yaml",
    ])
    
    for p in search_paths:
        if p.exists() and p.is_file():
            data = _load_yaml(p)
            break

    # Build sub-configs
    gw_data = data.get("gateway", {})
    pp_data = data.get("personaplex", {})
    audio_data = data.get("audio", {})
    latency_data = data.get("latency", {})

    # Apply environment variable overrides
    if "GATEWAY_HOST" in os.environ:
        gw_data["host"] = os.environ["GATEWAY_HOST"]
    if "GATEWAY_PORT" in os.environ:
        gw_data["port"] = int(os.environ["GATEWAY_PORT"])
    if "GATEWAY_API_KEY" in os.environ:
        gw_data["api_key"] = os.environ["GATEWAY_API_KEY"] or None
    if "WORKER_ENDPOINTS" in os.environ:
        gw_data["workers"] = [w.strip() for w in os.environ["WORKER_ENDPOINTS"].split(",") if w.strip()]
    if "CLONED_VOICES_DIR" in os.environ:
        gw_data["cloned_voices_dir"] = os.environ["CLONED_VOICES_DIR"]

    if "HF_TOKEN" in os.environ:
        pp_data["hf_token"] = os.environ["HF_TOKEN"]
    if "HF_HOME" in os.environ:
        pp_data["hf_home"] = os.environ["HF_HOME"]
    if "PERSONAPLEX_DEVICE" in os.environ:
        pp_data["device"] = os.environ["PERSONAPLEX_DEVICE"]
    if "PERSONAPLEX_VOICE_DIR" in os.environ:
        pp_data["voice_prompt_dir"] = os.environ["PERSONAPLEX_VOICE_DIR"]

    env = os.environ.get("ENVIRONMENT", data.get("environment", "production"))

    return AppConfig(
        environment=env,
        gateway=GatewayConfig(**gw_data),
        personaplex=PersonaPlexConfig(**pp_data),
        audio=AudioConfig(**audio_data),
        latency=LatencyBudget(**latency_data),
    )


# Global default configuration instance
settings = load_config()
