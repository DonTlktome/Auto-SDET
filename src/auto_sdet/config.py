"""
Configuration management using pydantic-settings.
"""
from typing import Literal, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


LLMProvider = Literal["deepseek", "openai", "anthropic", "ollama"]


class Settings(BaseSettings):
    """Application-wide settings, loaded from .env file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider Selection ──────────────────────────────
    llm_provider: LLMProvider = "deepseek"

    # ── DeepSeek ────────────────────────────────────────────
    deepseek_api_key: Optional[SecretStr] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-fast"

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: Optional[SecretStr] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.5-mini"

    # ── Anthropic / Claude ──────────────────────────────────
    anthropic_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-sonnet-4-7"

    # ── Ollama (local, no API key required) ─────────────────
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3.5-coder:7b"

    # ── E2B Sandbox ─────────────────────────────────────────
    e2b_api_key: SecretStr
    e2b_sandbox_timeout: int = 60

    # ── Agent Behavior ──────────────────────────────────────
    max_retries: int = 3

    # ── Episodic Memory ─────────────────────────────────────
    # memory_enabled=False: Reflector neither retrieves nor emits trajectories,
    # MemoryManager is a no-op (benchmark config B).
    memory_enabled: bool = True
    memory_dir: str = ".auto_sdet_memory"
    # legacy    — query = first 300 chars of pytest output; document =
    #             "<classification>: <root_cause first sentence> [in file]";
    #             L2-derived similarity ≥ 0.4; trajectory stored before re-run.
    # signature — both query and document are the normalized error signature
    #             extracted from pytest output; cosine similarity with a
    #             calibrated threshold; outcome written after the re-run.
    memory_variant: Literal["legacy", "signature"] = "legacy"
    memory_cosine_threshold: Optional[float] = None   # signature variant only

    # ── Experiment tracing ──────────────────────────────────
    # JSONL file receiving retrieval / execution / memory-op events.
    trace_file: Optional[str] = None


def get_settings() -> Settings:
    """Factory function to create Settings instance."""
    return Settings()
