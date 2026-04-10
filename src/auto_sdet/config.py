"""
Configuration management using pydantic-settings.
"""
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from .env file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # 环境变量名不区分大小写
        extra="ignore",             # 忽略 .env 中未定义的字段
    )

    # ── DeepSeek LLM ────────────────────────────────────────
    deepseek_api_key: SecretStr                              # 必填，无默认值
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ── E2B Sandbox ─────────────────────────────────────────
    e2b_api_key: SecretStr                                    # 必填
    e2b_sandbox_timeout: int = 60                             # 秒

    # ── Agent Behavior ──────────────────────────────────────
    max_retries: int = 3                                      # 最大自愈重试次数


def get_settings() -> Settings:
    """Factory function to create Settings instance."""
    return Settings()
