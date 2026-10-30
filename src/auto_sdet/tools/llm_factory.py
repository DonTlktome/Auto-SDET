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


def get_llm(multi_turn_tool_calling: bool = False, disable_thinking: bool = False) -> BaseChatModel:
    """
    Build a chat model instance for the configured LLM provider.

    Args:
        multi_turn_tool_calling: True only for callers that do an iterative
            `bind_tools` loop (Generator's read_file/list_directory dispatch).
            DeepSeek V4 thinking + multi-turn tool calling requires
            reasoning_content to be round-tripped between turns; LangChain's
            ChatDeepSeek does not preserve that field, so we disable thinking
            for these callers to avoid the resulting 400 errors.

            Set to False for:
              - single-turn `with_structured_output(method="json_mode")` callers
                (Reflector, Evaluator, MemoryManager) — these use `response_format`
                rather than `tool_choice`, which deepseek-reasoner supports fine
              - any plain `invoke()` caller with no tool calling
        disable_thinking: force thinking off for a single-turn caller. Used by
            the Reflector's recovery request after a structured-output failure
            (thinking can exhaust max_tokens before the JSON is complete).

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
        # DeepSeek V4 thinking + multi-turn tool calling is incompatible
        # under current LangChain — it doesn't round-trip reasoning_content
        # across turns and the API returns 400 on the second turn. Single-turn
        # structured-output callers don't hit this because they only need one
        # round-trip; they should keep thinking enabled.
        thinking_mode = "disabled" if (multi_turn_tool_calling or disable_thinking) else "enabled"
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


def describe_llm_config() -> dict:
    """Provider / model / sampling / per-node thinking, for experiment metadata."""
    settings = get_settings()
    provider = settings.llm_provider
    thinking = None
    if provider == "deepseek":
        # Mirrors get_llm(): only multi-turn bind_tools callers disable thinking.
        thinking = {
            "generator": "disabled",
            "evaluator": "enabled",
            "reflector": "enabled",
            "memory_manager": "enabled",
            "reflector_recovery": "disabled",
        }
    return {
        "provider": provider,
        "model": get_provider_label().split(":", 1)[1],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "thinking": thinking,
        "reflector_structured_failure_policy": (
            "one recovery request with thinking disabled; if that also fails the "
            "attempt counts as a failed repair and the loop continues (no saturation)"
        ),
    }


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
