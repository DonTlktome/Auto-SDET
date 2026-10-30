"""
Tests for experiment tracing: JSONL events and scored retrieval candidates.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from auto_sdet.graph.nodes.reflector import _trigger_of
from auto_sdet.tools import memory_store
from auto_sdet.tools.memory_store import MemoryStore, build_trajectory
from auto_sdet.tools.run_trace import MAX_TEXT_CHARS, clip, trace_event


# ══════════════════════════════════════════════════════════════════
# trace_event / clip
# ══════════════════════════════════════════════════════════════════

def test_trace_event_appends_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "run" / "trace.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    trace_event("execution", module="maths/fibonacci.py", retry=1, passed=True)
    trace_event("reflection", module="sorts/quick_sort.py", retry=2)

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [e["kind"] for e in events] == ["execution", "reflection"]
    assert events[0]["module"] == "maths/fibonacci.py"
    assert "ts" in events[0]


def test_trace_event_noop_without_trace_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACE_FILE", raising=False)
    monkeypatch.chdir(tmp_path)          # no .env here either
    trace_event("execution", module="x")  # must not raise
    assert list(tmp_path.iterdir()) == []


def test_clip_marks_truncation():
    text = "x" * (MAX_TEXT_CHARS + 10)
    clipped = clip(text)
    assert clipped.startswith("x" * MAX_TEXT_CHARS)
    assert "clipped 10 chars" in clipped
    assert clip("short") == "short"
    assert clip(None) is None


@pytest.mark.parametrize("exit_code, expected", [
    (1, "pytest"), (2, "pytest"), (-100, "evaluator_reject"), (-101, "ruff_fatal"),
])
def test_trigger_of(exit_code, expected):
    assert _trigger_of(SimpleNamespace(exit_code=exit_code)) == expected


def test_trigger_of_none():
    assert _trigger_of(None) == "unknown"


# ══════════════════════════════════════════════════════════════════
# retrieve_with_candidates
# ══════════════════════════════════════════════════════════════════

def _traj(classification, cause, target, module):
    return build_trajectory(
        error_classification=classification, root_cause=cause, fix_strategy="minimal_patch",
        fix_summary="-", target_file=target, source_module=module,
    )


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.add_trajectory(_traj("AssertionError", "Expected ValueError for negative input not raised",
                           "factorial.py", "maths/factorial.py"))
    s.add_trajectory(_traj("FixtureError", "The test uses the mocker fixture from pytest-mock which is not installed",
                           "stack.py", "data_structures/stacks/stack.py"))
    s.add_trajectory(_traj("ImportError", "No module named numpy in sandbox",
                           "sigmoid.py", "maths/sigmoid.py"))
    return s


def test_candidates_scored_and_kept_flags_match_selection(store):
    kept, candidates = store.retrieve_with_candidates("fixture 'mocker' not found", k=1)
    assert len(kept) == 1
    assert [c["id"] for c in candidates if c["kept"]] == [kept[0].trajectory_id]
    assert all(0.0 < c["similarity"] <= 1.0 for c in candidates)
    assert {c["source_module"] for c in candidates} <= {
        "maths/factorial.py", "data_structures/stacks/stack.py", "maths/sigmoid.py",
    }


def test_retrieve_similar_unchanged_by_candidate_tracking(store):
    query = "AssertionError: DID NOT RAISE ValueError [in factorial.py]"
    kept, _ = store.retrieve_with_candidates(query, k=3)
    assert [t.trajectory_id for t in store.retrieve_similar(query, k=3)] == [t.trajectory_id for t in kept]


def test_threshold_filtered_candidates_reported_but_not_kept(store, monkeypatch):
    monkeypatch.setattr(memory_store, "SIMILARITY_THRESHOLD", 0.99)
    kept, candidates = store.retrieve_with_candidates("completely unrelated text", k=3)
    assert kept == []
    assert candidates and not any(c["kept"] for c in candidates)


def test_source_module_round_trips(store):
    kept, _ = store.retrieve_with_candidates("No module named numpy", k=1)
    assert kept[0].source_module == "maths/sigmoid.py"
