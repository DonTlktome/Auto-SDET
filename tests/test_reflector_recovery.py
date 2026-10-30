"""
Tests for the Reflector's structured-output failure handling:
truncation / parse failure → one thinking-off recovery request → otherwise a
failed attempt that does NOT saturate retry_count.
"""
from __future__ import annotations

import pytest

from auto_sdet.graph.nodes import reflector
from auto_sdet.models.schemas import ExecutionResult, ReflectionResult

TRUNCATED = "Could not parse response content as the length limit was reached - CompletionUsage(...)"
FIX = ReflectionResult(
    error_classification="AssertionError", root_cause="wrong expected value",
    fix_strategy="minimal_patch", fixed_code="def test_x():\n    assert 1 == 1\n",
)


class _FakeLLM:
    """Stands in for a chat model: `.with_structured_output().invoke()` follows a script."""

    def __init__(self, outcome):
        self._outcome = outcome

    def with_structured_output(self, schema, method):
        return self

    def invoke(self, messages):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.fixture
def run_reflector(monkeypatch):
    """Run reflector_node with scripted outcomes for the thinking-on / thinking-off calls."""
    monkeypatch.setenv("MEMORY_ENABLED", "false")

    def _run(thinking_on, thinking_off=None, state_extra=None):
        calls = []

        def fake_get_llm(multi_turn_tool_calling=False, disable_thinking=False):
            calls.append("off" if disable_thinking else "on")
            return _FakeLLM(thinking_off if disable_thinking else thinking_on)

        monkeypatch.setattr(reflector, "get_llm", fake_get_llm)
        state = {
            "source_path": "/x/mod.py", "source_code": "def f(): pass\n", "test_code": "def test_x(): pass\n",
            "module_id": "pkg/mod.py", "retry_count": 0, "max_retries": 3, "error_history": ["boom"],
            "execution_result": ExecutionResult(exit_code=1, stdout="E   AssertionError"),
            **(state_extra or {}),
        }
        return reflector.reflector_node(state), calls

    return _run


def test_success_first_try_uses_thinking_only(run_reflector):
    out, calls = run_reflector(FIX)
    assert calls == ["on"]
    assert out["retry_count"] == 1
    assert out["test_code"] == FIX.fixed_code
    assert out["reflector_attempts"] == 1
    assert out["reflector_failed_attempts"] == 0
    assert out["structured_output_failures"] == {}


def test_truncation_recovered_with_thinking_off(run_reflector):
    out, calls = run_reflector(ValueError(TRUNCATED), FIX)
    assert calls == ["on", "off"]
    assert out["test_code"] == FIX.fixed_code
    assert out["retry_count"] == 1
    assert out["reflector_failed_attempts"] == 0
    assert out["structured_output_failures"] == {"truncated": 1, "recovered_with_thinking_off": 1}


def test_unrecovered_failure_is_one_failed_attempt_not_saturation(run_reflector):
    out, calls = run_reflector(ValueError(TRUNCATED), ValueError("Invalid json output: {"))
    assert calls == ["on", "off"]
    assert out["retry_count"] == 1                 # NOT max_retries
    assert out["status"] == "executing"
    assert out["test_code"] == "def test_x(): pass\n"
    assert out["reflector_attempts"] == 1
    assert out["reflector_failed_attempts"] == 1
    assert out["structured_output_failures"] == {"truncated": 1, "parse_error": 1}


def test_empty_result_counted(run_reflector):
    out, _ = run_reflector(None, FIX)
    assert out["structured_output_failures"] == {"empty": 1, "recovered_with_thinking_off": 1}


def test_counters_accumulate_across_attempts(run_reflector):
    out, _ = run_reflector(ValueError(TRUNCATED), FIX, state_extra={
        "retry_count": 1, "reflector_attempts": 1, "reflector_failed_attempts": 1,
        "structured_output_failures": {"truncated": 1, "parse_error": 1},
    })
    assert out["retry_count"] == 2
    assert out["reflector_attempts"] == 2
    assert out["reflector_failed_attempts"] == 1
    assert out["structured_output_failures"] == {"truncated": 2, "parse_error": 1, "recovered_with_thinking_off": 1}
