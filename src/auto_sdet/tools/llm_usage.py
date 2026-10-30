"""
LLM usage tracker — a LangChain callback that records wall time and token
usage of every chat-model call made during one agent run.

Attached once at `app.invoke(..., config={"callbacks": [tracker]})`; LangGraph
propagates it into every node, and the `langgraph_node` metadata on each call
lets us attribute time and tokens per node. The MemoryManager's background
thread copies the caller's contextvars, so its calls are tracked too (they run
off the critical path — summaries keep them separable via `by_node`).
"""
from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class LLMUsageTracker(BaseCallbackHandler):
    """Thread-safe accumulator of per-call LLM latency and token usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starts: dict[UUID, tuple[float, str]] = {}
        self.calls: list[dict[str, Any]] = []

    # ── Callback hooks ──────────────────────────────────

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kwargs) -> None:
        node = (metadata or {}).get("langgraph_node", "unknown")
        with self._lock:
            self._starts[run_id] = (time.perf_counter(), node)

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, metadata=None, **kwargs) -> None:
        self.on_chat_model_start(serialized, prompts, run_id=run_id, metadata=metadata)

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        self._finish(run_id, usage=_extract_usage(response), error=None)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        self._finish(run_id, usage={}, error=f"{type(error).__name__}: {error}"[:200])

    # ── Aggregation ─────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Totals plus a per-node breakdown, ready to drop into a results JSON."""
        with self._lock:
            calls = list(self.calls)

        def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "calls": len(rows),
                "errors": sum(1 for r in rows if r["error"]),
                "time_s": round(sum(r["duration_s"] for r in rows), 1),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                "completion_tokens": sum(r["completion_tokens"] for r in rows),
                "reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
                "cache_hit_tokens": sum(r["cache_hit_tokens"] for r in rows),
                "cache_miss_tokens": sum(r["cache_miss_tokens"] for r in rows),
            }

        by_node: dict[str, list[dict[str, Any]]] = {}
        for r in calls:
            by_node.setdefault(r["node"], []).append(r)
        return {**_agg(calls), "by_node": {n: _agg(rows) for n, rows in sorted(by_node.items())}}

    # ── Internal ────────────────────────────────────────

    def _finish(self, run_id: UUID, usage: dict[str, int], error: str | None) -> None:
        with self._lock:
            start, node = self._starts.pop(run_id, (None, "unknown"))
            if start is None:
                return
            self.calls.append({
                "node": node,
                "duration_s": time.perf_counter() - start,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "cache_hit_tokens": usage.get("cache_hit_tokens", 0),
                "cache_miss_tokens": usage.get("cache_miss_tokens", 0),
                "error": error,
            })


def _extract_usage(response) -> dict[str, int]:
    """
    Normalize token usage from an LLMResult.

    OpenAI-compatible providers (DeepSeek included) put the raw `usage` dict in
    `llm_output["token_usage"]`; DeepSeek adds prompt_cache_hit/miss_tokens.
    Falls back to the message's `usage_metadata` for other providers.
    """
    raw = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if raw:
        details = raw.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": raw.get("prompt_tokens") or 0,
            "completion_tokens": raw.get("completion_tokens") or 0,
            "reasoning_tokens": (details.get("reasoning_tokens") if isinstance(details, dict) else 0) or 0,
            "cache_hit_tokens": raw.get("prompt_cache_hit_tokens") or 0,
            "cache_miss_tokens": raw.get("prompt_cache_miss_tokens") or 0,
        }
    try:
        meta = response.generations[0][0].message.usage_metadata or {}
    except (AttributeError, IndexError):
        return {}
    return {
        "prompt_tokens": meta.get("input_tokens", 0),
        "completion_tokens": meta.get("output_tokens", 0),
        "reasoning_tokens": (meta.get("output_token_details") or {}).get("reasoning", 0),
        "cache_hit_tokens": (meta.get("input_token_details") or {}).get("cache_read", 0),
        "cache_miss_tokens": 0,
    }
