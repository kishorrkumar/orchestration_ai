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


# Global singleton settings instance
settings = Settings()
