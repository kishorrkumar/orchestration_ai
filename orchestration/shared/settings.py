"""
12-factor application settings configuration using pydantic-settings.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server binding
    host: str = Field(default="127.0.0.1", description="HTTP/WS server bind host")
    port: int = Field(default=8000, description="HTTP/WS server port")
    environment: str = Field(default="development", description="Runtime environment (development, production, test)")
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///data/platform.db",
        description="Async SQLAlchemy database URL (sqlite+aiosqlite or postgresql+asyncpg)",
    )

    # PersonaPlex S2S Worker
    personaplex_worker_url: str = Field(
        default="ws://127.0.0.1:8998",
        description="WebSocket URL of active PersonaPlex 7B inference worker",
    )
    personaplex_fallback_worker_url: str = Field(
        default="ws://127.0.0.1:8999",
        description="WebSocket URL of fallback worker",
    )
    auto_mock_worker: bool = Field(
        default=True,
        description="Automatically boot mock worker if no GPU worker responds",
    )

    # Prompts & Tokenizer
    tokenizer_path: str = Field(
        default="models/tokenizer_spm_32k_3.model",
        description="Path to SentencePiece 32k Moshi/PersonaPlex tokenizer",
    )
    max_prompt_tokens: int = Field(default=350, description="Hard token cutoff for system prompts")
    recommended_prompt_tokens: int = Field(default=150, description="Recommended token budget for fast startup")

    # Audio Engine
    client_sample_rate: int = Field(default=16000, description="Canonical Web client PCM sample rate")
    model_sample_rate: int = Field(default=24000, description="Internal PersonaPlex 7B Mimi codec sample rate")
    telephony_sample_rate: int = Field(default=8000, description="Telephony G.711 sample rate")

    # Timezone
    default_timezone: str = Field(default="UTC", description="Fallback IANA timezone")

    # Rate Limiting & Auth
    api_key: str | None = Field(default=None, description="Optional bearer token / API key")
    rate_limit_per_minute: int = Field(default=60, description="Max API requests per minute per IP")

    # Phase 1: Session Lifecycle & Timeouts
    idle_soft_silence_sec: float = Field(default=10.0, description="Inactivity seconds before conversational soft nudge")
    idle_hard_silence_sec: float = Field(default=25.0, description="Inactivity seconds before terminating call")
    max_call_duration_sec: float = Field(default=900.0, description="Max session duration cap in seconds (15m)")
    pool_acquire_timeout_sec: float = Field(default=8.0, description="Max seconds to wait in queue for an idle GPU worker")
    pool_max_wait_queue: int = Field(default=5, description="Max queued requests waiting for a worker before fast reject")
    worker_retry_attempts: int = Field(default=3, description="Retry attempts for transient worker failures")
    worker_retry_backoff_sec: float = Field(default=0.5, description="Initial backoff for worker retries")
    enable_cpu_offload: bool = Field(default=False, description="Flag for upstream CPU offload support")
    enable_structured_logging: bool = Field(default=True, description="Emit JSON formatted logs")
    enable_standby_worker: bool = Field(default=True, description="Pre-prime warm worker in background")


# Global singleton settings instance
settings = Settings()
