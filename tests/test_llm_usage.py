"""
Tests for LLMUsageTracker — per-node LLM time / token accounting.
"""
from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from auto_sdet.tools.llm_usage import LLMUsageTracker


def _deepseek_result(prompt: int, completion: int, hit: int, miss: int, reasoning: int) -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={"token_usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": miss,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        }},
    )


def test_tokens_and_calls_aggregated_per_node():
    tracker = LLMUsageTracker()
    for node, usage in [("generator", (100, 50, 0, 100, 0)), ("reflector", (200, 80, 150, 50, 30)),
                        ("reflector", (300, 90, 250, 50, 40))]:
        rid = uuid4()
        tracker.on_chat_model_start({}, [[]], run_id=rid, metadata={"langgraph_node": node})
        tracker.on_llm_end(_deepseek_result(*usage), run_id=rid)

    s = tracker.summary()
    assert s["calls"] == 3
    assert s["prompt_tokens"] == 600
    assert s["completion_tokens"] == 220
    assert s["cache_hit_tokens"] == 400
    assert s["reasoning_tokens"] == 70
    assert s["by_node"]["reflector"]["calls"] == 2
    assert s["by_node"]["generator"]["prompt_tokens"] == 100


def test_errors_counted_and_unknown_node_tolerated():
    tracker = LLMUsageTracker()
    rid = uuid4()
    tracker.on_chat_model_start({}, [[]], run_id=rid)
    tracker.on_llm_error(RuntimeError("400 bad request"), run_id=rid)

    s = tracker.summary()
    assert s["calls"] == 1
    assert s["errors"] == 1
    assert "unknown" in s["by_node"]


def test_usage_metadata_fallback():
    tracker = LLMUsageTracker()
    rid = uuid4()
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
    tracker.on_chat_model_start({}, [[]], run_id=rid, metadata={"langgraph_node": "evaluator"})
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)

    s = tracker.summary()
    assert s["prompt_tokens"] == 10
    assert s["completion_tokens"] == 4


def test_end_without_start_ignored():
    tracker = LLMUsageTracker()
    tracker.on_llm_end(_deepseek_result(1, 1, 0, 1, 0), run_id=uuid4())
    assert tracker.summary()["calls"] == 0
