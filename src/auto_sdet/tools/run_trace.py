"""
Experiment event trace — append-only JSONL written when `TRACE_FILE` is set.

Three event kinds, all keyed by `module` (path relative to the benchmark
root) and `retry` (the attempt number the event belongs to):

  reflection  — Reflector: the error it saw, the memory query, every retrieved
                candidate with its similarity / source module / whether it was
                injected into the prompt, and the test code before / after.
  execution   — Executor: exit code, pass / fail, infra failures, output.
                The execution with the same (module, retry) as a reflection
                tells whether that fix worked.
  memory_op   — MemoryManager: the CRUD decision applied to a trajectory.

Tracing is observability only: it never changes agent behavior, and any
failure to write is swallowed.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cap on long text fields (pytest output, test files) per event.
MAX_TEXT_CHARS = 20000

_lock = threading.Lock()


def _trace_path() -> Path | None:
    try:
        from auto_sdet.config import get_settings
        path = get_settings().trace_file
    except Exception:
        return None
    return Path(path) if path else None


def clip(text: str | None, limit: int = MAX_TEXT_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n…[clipped {len(text) - limit} chars]"


def trace_event(kind: str, **fields: Any) -> None:
    """Append one event to the trace file (no-op when tracing is off)."""
    path = _trace_path()
    if path is None:
        return
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "kind": kind, **fields}
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.warning(f"trace_event({kind}) failed: {e}")
