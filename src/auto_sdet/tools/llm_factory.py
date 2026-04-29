"""
LLM Provider abstraction — returns a configured chat model instance
based on the active provider, hiding vendor-specific initialization.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from auto_sdet.config import get_settings


# ── Default sampling parameters used by both Generator and Reflector ──
# Both nodes need deterministic output, so temperature=0 is enforced.
# 4096 tokens was insufficient for large test files (e.g. 400+ line tests
# get truncated mid-function and break pytest collection). Modern V4-class
# models support 8K-16K output without issue.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 16384


def get_llm(for_tool_use: bool = False) -> BaseChatModel:
    """
    Build a chat model instance for the configured LLM provider.

    Args:
        for_tool_use: If True, configure the model for safe multi-turn
            tool-calling. For DeepSeek V4 thinking models this means
            disabling thinking, since `reasoning_content` must be
            round-tripped in tool calls (DeepSeek docs) and current
            LangChain clients don't preserve that field across turns.
            For single-turn callers (e.g. Reflector), set False to keep
            thinking enabled and benefit from deeper reasoning.

    Raises:
        ValueError: required API key for the selected provider is missing.
        ImportError: optional provider package (e.g. langchain-anthropic)
            is not installed.
    """
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as e:
            raise ImportError(
                "langchain-deepseek is required for the DeepSeek provider. "
                "Install it with: pip install langchain-deepseek"
            ) from e
        # Per DeepSeek docs: when tool calling is involved, reasoning_content
        # must be round-tripped or the API returns 400. LangChain clients
        # don't preserve that field, so we disable thinking for tool-use callers.
        # For single-turn callers, keep thinking enabled to benefit from V4 reasoning.
        thinking_mode = "disabled" if for_tool_use else "enabled"
        return ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key.get_secret_value(),
            api_base=settings.deepseek_base_url,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            extra_body={"thinking": {"type": thinking_mode}},
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API on /v1; no real key needed
        # but ChatOpenAI requires the field to be set, so use a placeholder.
        return ChatOpenAI(
            model=settings.ollama_model,
            api_key="ollama-local",
            base_url=settings.ollama_base_url,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Install it with: pip install langchain-anthropic"
            ) from e
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_provider_label() -> str:
    """Human-readable label for the active provider, used in CLI logs."""
    settings = get_settings()
    provider = settings.llm_provider
    model = {
        "deepseek": settings.deepseek_model,
        "openai": settings.openai_model,
        "anthropic": settings.anthropic_model,
        "ollama": settings.ollama_model,
    }.get(provider, "unknown")
    return f"{provider}:{model}"
