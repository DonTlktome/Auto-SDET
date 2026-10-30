"""
Tests for the `signature` memory variant (benchmark config C):
cosine store, verified outcomes, and the Executor → MemoryManager hand-off.
"""
from __future__ import annotations

import pytest

from auto_sdet.graph.nodes import executor as executor_module
from auto_sdet.graph.nodes import memory_manager
from auto_sdet.models.schemas import ExecutionResult
from auto_sdet.tools.memory_store import MemoryStore, build_trajectory, get_memory_store


def _pending(outcome: str = "pending"):
    return build_trajectory(
        error_classification="AssertionError", root_cause="Expected ValueError. Not raised.",
        fix_strategy="minimal_patch", fix_summary="-", target_file="factorial.py",
        outcome=outcome, source_module="maths/factorial.py",
    )


# ══════════════════════════════════════════════════════════════════
# Cosine store
# ══════════════════════════════════════════════════════════════════

def test_cosine_store_requires_threshold(tmp_path):
    with pytest.raises(ValueError):
        MemoryStore(tmp_path, space="cosine")


def test_cosine_similarity_and_threshold(tmp_path):
    store = MemoryStore(tmp_path / "c", space="cosine", similarity_threshold=0.6)
    traj = _pending("fix_applied").model_copy(
        update={"error_signature": "Failed: DID NOT RAISE <class 'ValueError'>"})
    store.add_trajectory(traj)

    kept, cands = store.retrieve_with_candidates("Failed: DID NOT RAISE <class 'ValueError'>", k=3)
    assert [t.trajectory_id for t in kept] == [traj.trajectory_id]
    assert cands[0]["similarity"] == pytest.approx(1.0, abs=1e-3)

    kept, cands = store.retrieve_with_candidates("ModuleNotFoundError: No module named 'numpy'", k=3)
    assert kept == []
    assert cands[0]["similarity"] < 0.6


def test_space_mismatch_detected(tmp_path):
    MemoryStore(tmp_path / "s").add_trajectory(_pending("fix_applied"))          # l2 store
    with pytest.raises(ValueError):
        MemoryStore(tmp_path / "s", space="cosine", similarity_threshold=0.5)._get_collection()


def test_get_memory_store_signature_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_VARIANT", "signature")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("MEMORY_COSINE_THRESHOLD", "0.7")
    store = get_memory_store()
    assert store._space == "cosine" and store._threshold == 0.7


def test_get_memory_store_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    assert get_memory_store() is None


# ══════════════════════════════════════════════════════════════════
# Deferred, verified commit
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def spawned(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_manager, "_spawn_background", lambda t: calls.append(t))
    return calls


def test_pending_trajectory_held_by_memory_manager(spawned):
    pending = _pending()
    assert memory_manager.memory_manager_node({"pending_trajectory": pending}) == {}
    assert spawned == []


def test_legacy_trajectory_committed_immediately(spawned):
    out = memory_manager.memory_manager_node({"pending_trajectory": _pending("fix_applied")})
    assert out == {"pending_trajectory": None}
    assert len(spawned) == 1


@pytest.mark.parametrize("passed, outcome", [(True, "fix_applied"), (False, "fix_failed")])
def test_commit_verified_sets_real_outcome(spawned, passed, outcome):
    memory_manager.commit_verified_trajectory(_pending(), passed=passed)
    assert spawned[0].outcome == outcome


class _FakeSandbox:
    def __init__(self, result):
        self._result = result

    def __call__(self, *a, **k):
        return self

    def execute_test(self, **kwargs):
        return self._result


def _state(pending):
    return {"source_path": "/x/factorial.py", "source_code": "", "test_code": "", "retry_count": 1,
            "module_id": "maths/factorial.py", "pending_trajectory": pending}


@pytest.mark.parametrize("exit_code, outcome", [(0, "fix_applied"), (1, "fix_failed")])
def test_executor_commits_held_trajectory_with_pytest_result(monkeypatch, spawned, exit_code, outcome):
    monkeypatch.setattr(executor_module, "SandboxExecutor", _FakeSandbox(ExecutionResult(exit_code=exit_code)))
    out = executor_module.executor_node(_state(_pending()))
    assert out["pending_trajectory"] is None
    assert [t.outcome for t in spawned] == [outcome]


def test_executor_drops_held_trajectory_on_infra_failure(monkeypatch, spawned):
    infra = ExecutionResult(exit_code=-1, infra_failure=True, infra_failures=3)
    monkeypatch.setattr(executor_module, "SandboxExecutor", _FakeSandbox(infra))
    out = executor_module.executor_node(_state(_pending()))
    assert out["pending_trajectory"] is None
    assert spawned == []


def test_executor_leaves_legacy_flow_alone(monkeypatch, spawned):
    monkeypatch.setattr(executor_module, "SandboxExecutor", _FakeSandbox(ExecutionResult(exit_code=0)))
    out = executor_module.executor_node(_state(None))
    assert "pending_trajectory" not in out
    assert spawned == []
