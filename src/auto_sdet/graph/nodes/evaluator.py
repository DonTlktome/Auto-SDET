"""
Evaluator Node — LLM-as-Judge quality assessment of generated test code.

Sits between Generator and Executor. Scores the test file across 4
dimensions and decides whether to:
  - pass through to Executor (score ≥ 0.5)
  - route back to Reflector with feedback (score < 0.5)

When routing to Reflector, the evaluation feedback is synthesized into a
pseudo-stderr and pushed onto error_history, so the existing Reflector
implementation can consume it without code changes.

Tool calling note: with_structured_output uses tool_choice internally,
which is incompatible with DeepSeek thinking mode. for_tool_use=True
disables thinking for this call.
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.markup import escape

from auto_sdet.models.schemas import AgentState, EvaluationResult, ExecutionResult
from auto_sdet.prompts.evaluator import build_evaluator_prompt
from auto_sdet.tools.llm_factory import get_llm

logger = logging.getLogger(__name__)
console = Console()

# Thresholds used by route_after_evaluator. Centralized here so the
# evaluator's terminal output can label scores using the same boundaries.
PASS_THRESHOLD = 0.75
WARN_THRESHOLD = 0.50


def evaluator_node(state: AgentState) -> dict:
    """LLM-as-Judge: score the generated test file, decide next step."""
    console.print(
        "[bold blue]🧐 [Evaluator][/]  Scoring generated test file..."
    )

    system_prompt, user_prompt = build_evaluator_prompt(
        source_path=state["source_path"],
        source_code=state["source_code"],
        test_code=state["test_code"],
    )

    # tool_choice path → must disable thinking
    llm = get_llm(for_tool_use=True)
    structured_llm = llm.with_structured_output(EvaluationResult)

    try:
        result: EvaluationResult = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        # Evaluator failure: fall through to Executor rather than block the
        # whole pipeline. The real test run is the source of truth anyway.
        logger.error(f"Evaluator structured output failed: {e}")
        console.print(
            f"[bold yellow]⚠ [Evaluator][/]  Failed to score "
            f"({escape(str(e))[:80]}…) — falling through to Executor"
        )
        return {
            "evaluation_result": None,
            "status": "executing",
        }

    # ── Pretty-print scores + breakdown ─────────────────
    score_label = (
        "[green]PASS[/]" if result.overall_score >= PASS_THRESHOLD
        else "[yellow]WARN[/]" if result.overall_score >= WARN_THRESHOLD
        else "[red]REJECT[/]"
    )
    console.print(
        f"[bold blue]🧐 [Evaluator][/]  Overall: [bold]{result.overall_score:.2f}[/] "
        f"({score_label})"
    )
    console.print(
        f"[dim blue]  ├─ testable coverage: {result.testable_coverage_score:.2f}[/]\n"
        f"[dim blue]  ├─ assertion quality: {result.assertion_quality:.2f}[/]\n"
        f"[dim blue]  ├─ mock correctness:  {result.mock_correctness:.2f}[/]\n"
        f"[dim blue]  └─ code consistency:  {result.code_consistency:.2f}[/]"
    )

    if result.skipped_intentionally:
        console.print("[dim cyan]  Intentionally skipped (anti-gaming):[/]")
        for s in result.skipped_intentionally[:3]:
            console.print(f"[dim cyan]    • {escape(s)}[/]")

    actionable_weaknesses = [w for w in result.weaknesses if w.actionable]
    if actionable_weaknesses:
        console.print("[dim yellow]  Actionable weaknesses:[/]")
        for w in actionable_weaknesses[:5]:
            console.print(
                f"[dim yellow]    • [{w.severity}] {escape(w.issue)}[/]"
            )

    # ── Decide: pass through or feed back to Reflector ──
    update: dict = {"evaluation_result": result}
    reject_count = state.get("evaluator_reject_count", 0)

    # Anti-loop guard: after 2 rejections, stop second-guessing the LLM
    # and let the real Executor decide.
    if reject_count >= 2:
        console.print(
            "[bold yellow]⚠ [Evaluator][/]  "
            f"Already rejected {reject_count} time(s) — passing through to Executor"
        )
        update["status"] = "executing"
        return update

    if result.overall_score >= WARN_THRESHOLD:
        # 0.5–0.75 → still send to Executor, don't waste a Reflector cycle
        update["status"] = "executing"
        return update

    # Below WARN_THRESHOLD: synthesize feedback as pseudo-stderr so the
    # existing Reflector node can consume it through its usual interface.
    feedback_str = _format_evaluation_as_pseudo_stderr(result)
    pseudo_result = ExecutionResult(
        stdout="",
        stderr=feedback_str,
        exit_code=-100,        # sentinel: "rejected by evaluator", not a real pytest exit
        duration_ms=0,
        sandbox_id="evaluator",
        coverage_pct=None,
    )
    error_history = list(state.get("error_history", []))
    error_history.append(feedback_str)

    console.print(
        "[bold magenta]🧐 [Evaluator][/]  "
        f"Score below {WARN_THRESHOLD} — routing to Reflector with feedback"
    )

    update.update({
        "execution_result": pseudo_result,
        "error_history": error_history,
        "evaluator_reject_count": reject_count + 1,
        "status": "reflecting",
    })
    return update


def _format_evaluation_as_pseudo_stderr(result: EvaluationResult) -> str:
    """
    Render an EvaluationResult as a synthetic stderr-like block so the
    Reflector node (which expects stderr-style input) can consume it.

    Keeps the format close to a pytest report so the Reflector's CoT
    error-classification step still hits a familiar shape.
    """
    lines = [
        "==================== EVALUATOR REJECTION ====================",
        f"overall_score = {result.overall_score:.2f} (below threshold)",
        "",
        f"testable_coverage_score: {result.testable_coverage_score:.2f}",
        f"assertion_quality:       {result.assertion_quality:.2f}",
        f"mock_correctness:        {result.mock_correctness:.2f}",
        f"code_consistency:        {result.code_consistency:.2f}",
        "",
    ]

    actionable = [w for w in result.weaknesses if w.actionable]
    if actionable:
        lines.append("ACTIONABLE WEAKNESSES (must be addressed):")
        for w in actionable:
            lines.append(f"  [{w.severity}] {w.issue}")
            if w.suggested_fix:
                lines.append(f"    → suggested: {w.suggested_fix}")
        lines.append("")

    if result.skipped_intentionally:
        lines.append(
            "INTENTIONALLY SKIPPED (do NOT add tests for these — would be coverage gaming):"
        )
        for s in result.skipped_intentionally:
            lines.append(f"  - {s}")
        lines.append("")

    lines.append("==================== END EVALUATOR REJECTION ====================")
    return "\n".join(lines)