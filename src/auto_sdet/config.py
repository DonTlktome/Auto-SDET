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


def get_settings() -> Settings:
    """Factory function to create Settings instance."""
    return Settings()
