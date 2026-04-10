"""
Core data models (Pydantic v2) shared across all nodes.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════
# Tool I/O Models  (Pydantic BaseModel — 带验证)
# ══════════════════════════════════════════════════════════

class FileContent(BaseModel):
    """Represents a file read via MCP filesystem server."""
    path: str = Field(..., description="Absolute or relative path of the file")
    content: str = Field(..., description="Full text content of the file")
    mime_type: str = Field(default="text/plain", description="MIME type")


class ExecutionResult(BaseModel):
    """Result returned from E2B sandbox execution."""
    stdout: str = Field(default="", description="Standard output from pytest")
    stderr: str = Field(default="", description="Standard error / traceback")
    exit_code: int = Field(..., description="0 = all tests passed, non-zero = failure")
    duration_ms: int = Field(default=0, description="Execution wall time in milliseconds")
    sandbox_id: str = Field(default="", description="E2B sandbox instance ID for tracing")


# ══════════════════════════════════════════════════════════
# LangGraph Agent State  (TypedDict — 节点间共享状态)
# ══════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    """The shared state that flows between LangGraph nodes."""

    # ── Input context (set by CLI / Generator) ──────────
    source_code: str                      # Target source file content
    source_path: str                      # Target source file path (for E2B upload)
    context_files: dict[str, str]         # {filename: content} of sibling .py files

    # ── Generated test (set by Generator / Reflector) ───
    test_code: str                        # Current version of generated test code

    # ── Execution feedback (set by Executor) ────────────
    execution_result: Optional[ExecutionResult]

    # ── Loop control ────────────────────────────────────
    retry_count: int                      # Current retry number (starts at 0)
    max_retries: int                      # Upper bound for retries (from config)
    error_history: list[str]              # Accumulated stderr logs for Reflector CoT

    # ── Observability ───────────────────────────────────
    status: Literal[
        "generating",
        "executing",
        "reflecting",
        "done",
        "failed",
    ]
