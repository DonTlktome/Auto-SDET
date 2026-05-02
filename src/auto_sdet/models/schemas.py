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


class SymbolSpec(BaseModel):
    """
    A top-level public symbol (function or class) extracted from a source file.

    Used by the parallel-generation flow: the Splitter node fans out
    one Send per SymbolSpec to a Generator worker, which writes tests
    just for that symbol.
    """
    name: str = Field(..., description="Symbol identifier, e.g. 'add' or 'Calculator'")
    kind: Literal["function", "class"] = Field(..., description="What kind of symbol")
    source: str = Field(..., description="Full source code of the symbol (preserves comments and formatting)")
    signature: str = Field(..., description="Signature representation (def/class header + docstring summary, no body)")
    docstring: Optional[str] = Field(default=None, description="First non-empty line of the docstring, if any")


class Weakness(BaseModel):
    """
    A single weakness in a generated test file, surfaced by the Evaluator.

    `actionable` is the anti-coverage-gaming flag: only weaknesses that can
    be fixed WITHOUT introducing fake mocks / vacuous assertions count
    toward the overall_score. "main() not tested" should be marked
    actionable=False because forcing a test would require mocking input(),
    which is testing-the-mock.
    """
    issue: str = Field(..., description="Brief description of the problem")
    severity: Literal["minor", "moderate", "blocking"] = Field(...)
    actionable: bool = Field(
        ...,
        description=(
            "True only if this weakness can be fixed without coverage gaming "
            "(no fake mocks for untestable code). Non-actionable weaknesses "
            "are reported but do NOT lower the score."
        ),
    )
    suggested_fix: Optional[str] = Field(default=None)


class EvaluationResult(BaseModel):
    """
    Structured quality assessment of a generated test file, produced by the
    LLM-as-Judge Evaluator node.

    Anti-coverage-gaming design:
      - `testable_coverage_score` measures coverage of CODE THAT SHOULD BE
        TESTED, not raw line coverage. CLI entrypoints, interactive input
        loops, and pure-print helpers are excluded from the denominator
        (and explicitly listed in `skipped_intentionally`).
      - This prevents the Evaluator from punishing the Generator for
        skipping legitimately-untestable code, which would push the agent
        to write fake mocks just to bump coverage numbers.
    """
    # ── Quality dimensions (0.0 – 1.0 each) ─────────────────
    testable_coverage_score: float = Field(..., ge=0.0, le=1.0,
        description="Coverage of testable code only — CLI / IO / print helpers excluded")
    assertion_quality: float = Field(..., ge=0.0, le=1.0,
        description="Are assertions specific (== / raises / type) vs vacuous (is not None)?")
    mock_correctness: float = Field(..., ge=0.0, le=1.0,
        description="patch target paths and fixture names — correct and stdlib-compatible?")
    code_consistency: float = Field(..., ge=0.0, le=1.0,
        description="Within-file consistency of imports / naming / mock style")
    overall_score: float = Field(..., ge=0.0, le=1.0,
        description="Weighted overall score; the router compares this against the threshold")

    # ── Explanatory fields ──────────────────────────────────
    skipped_intentionally: list[str] = Field(default_factory=list,
        description=(
            "Code regions deliberately not tested + reason. Example: "
            "'main() — contains input() loop, not unit-testable'. "
            "REQUIRED so they don't get re-flagged as missing coverage."
        ),
    )
    strengths: list[str] = Field(default_factory=list,
        description="Up to 3 things the test file does well")
    weaknesses: list[Weakness] = Field(default_factory=list,
        description="Issues found. Only those with actionable=True count toward the score.")

    recommended_action: Literal["pass_to_executor", "regenerate", "manual_review"] = Field(...)


class ReflectionResult(BaseModel):
    """
    Structured output from the Reflector node.

    Replaces free-form text + regex extraction with a strict schema.
    The LLM is forced to emit each field as a separate JSON value, which:
      1. Eliminates the markdown-fence parsing failure mode entirely
      2. Surfaces error_classification + root_cause + fix_strategy as
         first-class fields — usable for downstream analytics / logging
      3. Lets us validate `affected_lines` and `fix_strategy` at runtime
         instead of trusting prose
    """
    error_classification: Literal[
        "ImportError",
        "AssertionError",
        "TypeError",
        "AttributeError",
        "SyntaxError",
        "FixtureError",
        "MockError",
        "Other",
    ] = Field(..., description="Most relevant pytest error category for the failure")

    root_cause: str = Field(
        ...,
        description="One-sentence explanation of why the test failed (NOT how to fix it)",
    )

    affected_lines: list[int] = Field(
        default_factory=list,
        description="Line numbers in the test file that need to change (1-indexed)",
    )

    fix_strategy: Literal["minimal_patch", "full_rewrite"] = Field(
        ...,
        description=(
            "minimal_patch — change only the affected lines; "
            "full_rewrite — rewrite larger portions when minimal isn't viable"
        ),
    )

    fixed_code: str = Field(
        ...,
        description=(
            "The COMPLETE corrected test file content. "
            "Must be valid Python, must include all imports, must contain every test "
            "from the original file unless deletion is the actual fix."
        ),
    )


class ExecutionResult(BaseModel):
    """Result returned from E2B sandbox execution."""
    stdout: str = Field(default="", description="Standard output from pytest")
    stderr: str = Field(default="", description="Standard error / traceback")
    exit_code: int = Field(..., description="0 = all tests passed, non-zero = failure")
    duration_ms: int = Field(default=0, description="Execution wall time in milliseconds")
    sandbox_id: str = Field(default="", description="E2B sandbox instance ID for tracing")
    coverage_pct: Optional[int] = Field(default=None, description="Line coverage percentage (0-100), None if unavailable")


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

    # ── LLM-as-Judge Evaluator output (set by evaluator_node) ──────
    evaluation_result: Optional[EvaluationResult]
    # How many times the Evaluator has rejected the current test_code.
    # Used as an anti-loop guard: after 2 rejections the router
    # forces the flow to Executor regardless of evaluation score.
    evaluator_reject_count: int

    # ── Execution feedback (set by Executor) ────────────
    execution_result: Optional[ExecutionResult]

    # ── Loop control ────────────────────────────────────
    retry_count: int                      # Current retry number (starts at 0)
    max_retries: int                      # Upper bound for retries (from config)
    error_history: list[str]              # Accumulated stderr logs for Reflector CoT

    # ── Observability ───────────────────────────────────
    status: Literal[
        "generating",
        "evaluating",
        "executing",
        "reflecting",
        "done",
        "failed",
    ]
