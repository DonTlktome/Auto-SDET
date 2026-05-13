"""
Evaluator Node — LLM-as-Judge quality assessment of generated test code.

Sits between Generator and Executor. Scores the test file across 4
dimensions and decides whether to:
  - pass through to Executor (score ≥ 0.5)
  - route back to Reflector with feedback (score < 0.5)

When routing to Reflector, the evaluation feedback is synthesized into a
pseudo-stderr and pushed onto error_history, so the existing Reflector
implementation can consume it without code changes.

Protocol note: with_structured_output is called with method="json_mode",
which uses `response_format` instead of `tool_choice`. deepseek-reasoner
rejects `tool_choice` but accepts `response_format`, so this path keeps
thinking enabled and we get both structured output AND V4 reasoning.
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.markup import escape

from auto_sdet.models.schemas import AgentState, EvaluationResult, ExecutionResult
from auto_sdet.prompts.evaluator import build_evaluator_prompt
from auto_sdet.tools.llm_factory import get_llm
from auto_sdet.tools.ruff_checker import check_code, format_issues_for_prompt

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

    # ── Step 0: Static-analysis sidecar signal (ruff) ─────
    # Independent of the LLM judge — catches the things same-model self-eval
    # is known to miss (syntax / unused imports / undefined names). Fatal
    # findings (parser errors, undefined symbols) short-circuit straight to
    # Reflector since the code provably can't run — no point asking the LLM
    # judge or paying for a sandbox cycle.
    ruff_issues = check_code(state["test_code"])
    fatal_issues = [i for i in ruff_issues if i.is_fatal]

    if fatal_issues:
        console.print(
            f"[bold red]🧐 [Evaluator][/]  "
            f"ruff caught {len(fatal_issues)} fatal issue(s) — bypassing LLM judge, "
            f"routing direct to Reflector"
        )
        for issue in fatal_issues[:5]:
            console.print(
                f"[dim red]  L{issue.line} {issue.code}: {escape(issue.message)}[/]"
            )

        # Synthesize stderr-like feedback exactly the way the LLM-judge
        # rejection path does, so Reflector consumes a uniform format.
        feedback_str = _format_ruff_fatal_as_pseudo_stderr(fatal_issues)
        pseudo_result = ExecutionResult(
            stdout="", stderr=feedback_str,
            exit_code=-101,        # distinct sentinel from -100 (LLM judge reject)
            sandbox_id="ruff",
        )
        error_history = list(state.get("error_history", []))
        error_history.append(feedback_str)
        return {
            "evaluation_result": None,
            "execution_result": pseudo_result,
            "error_history": error_history,
            "evaluator_reject_count": state.get("evaluator_reject_count", 0) + 1,
            "status": "reflecting",
        }

    # Non-fatal ruff issues get folded into the LLM-judge prompt as a sidecar
    # fact set — the LLM weighs them as one input among many.
    ruff_report = format_issues_for_prompt(ruff_issues)

    system_prompt, user_prompt = build_evaluator_prompt(
        source_path=state["source_path"],
        source_code=state["source_code"],
        test_code=state["test_code"],
        ruff_report=ruff_report,
    )

    # Single-turn json_mode: deepseek-reasoner supports `response_format`
    # alongside thinking, unlike the default function_calling method which
    # uses tool_choice (rejected). Thinking ON helps the judge weigh trade-offs
    # between the 4 quality dimensions more carefully.
    llm = get_llm(multi_turn_tool_calling=False)
    structured_llm = llm.with_structured_output(EvaluationResult, method="json_mode")

    # Same two failure modes as Reflector: invoke raises, or invoke returns None.
    # Both mean "no usable score" — fall through to Executor since the real
    # pytest run is the source of truth anyway, no point blocking on a soft check.
    result: EvaluationResult | None = None
    failure_reason: str | None = None
    try:
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        failure_reason = f"invoke raised: {e}"
    if result is None and failure_reason is None:
        failure_reason = "invoke returned None"

    if result is None:
        logger.error(f"Evaluator structured output unavailable: {failure_reason}")
        console.print(
            f"[bold yellow]⚠ [Evaluator][/]  Failed to score "
            f"({escape(failure_reason or 'unknown')[:80]}…) — falling through to Executor"
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


def _format_ruff_fatal_as_pseudo_stderr(fatal_issues) -> str:
    """
    Render ruff fatal findings as a stderr-like block (same shape as the
    LLM-judge rejection) so Reflector consumes a uniform error format.
    """
    lines = [
        "==================== STATIC ANALYSIS (ruff) — FATAL ====================",
        "The generated test code cannot run — ruff detected blocking issues.",
        "",
    ]
    for issue in fatal_issues:
        lines.append(f"  L{issue.line} {issue.code}: {issue.message}")
    lines.append("")
    lines.append("Fix these before the code can even be parsed/imported by pytest.")
    lines.append("==================== END STATIC ANALYSIS ====================")
    return "\n".join(lines)