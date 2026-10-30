"""
Tests for flushing MemoryManager background writes before run_agent returns.
"""
from __future__ import annotations

import threading

from auto_sdet.graph.nodes import memory_manager


def _register(target) -> threading.Thread:
    t = threading.Thread(target=target, daemon=True)
    with memory_manager._pending_lock:
        memory_manager._pending_threads.append(t)
    t.start()
    return t


def test_flush_waits_for_background_write():
    written = []
    release = threading.Event()

    def slow_write():
        release.wait(timeout=5)
        written.append("trajectory")

    _register(slow_write)
    threading.Timer(0.2, release.set).start()

    assert memory_manager.wait_for_pending_memory_ops(timeout=5) == 0
    assert written == ["trajectory"]
    assert memory_manager._pending_threads == []


def test_flush_reports_ops_still_running_after_timeout():
    release = threading.Event()
    _register(lambda: release.wait(timeout=5))
    try:
        assert memory_manager.wait_for_pending_memory_ops(timeout=0.05) == 1
    finally:
        release.set()
        assert memory_manager.wait_for_pending_memory_ops(timeout=5) == 0


def test_flush_with_nothing_pending_is_noop():
    assert memory_manager.wait_for_pending_memory_ops(timeout=0.01) == 0
