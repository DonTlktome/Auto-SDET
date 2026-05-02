"""
Reflector Node — analyzes test failures via CoT and generates patches.

Output is constrained to a `ReflectionResult` Pydantic schema via
`llm.with_structured_output`. This eliminates the markdown-fence parsing
fragility that the old free-text mode suffered from, and surfaces
diagnostic fields (error classification, root cause, fix strategy) as
first-class data instead of buried prose.
"""
from __future__ import annotations

import difflib
import logging

from rich.markup import escape
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from auto_sdet.models.schemas import AgentState, ReflectionResult
from auto_sdet.tools.llm_factory import get_llm
from auto_sdet.prompts.reflector import build_reflector_prompt

logger = logging.getLogger(__name__)
console = Console()


def reflector_node(state: AgentState) -> dict:
    """Reflector Node: analyze failure via CoT, return patched test code via structured output."""
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    console.print(
        f"[bold magenta]🧠 [Reflector][/]  "
        f"Analyzing error (attempt {retry_count + 1}/{max_retries})..."
    )

    # ── Step 1: Collect error context ───────────────────
    execution_result = state.get("execution_result")
    latest_error = (execution_result.stdout + execution_result.stderr).strip() if execution_result else "Unknown error"
    error_history = state.get("error_history", [])

    # ── Step 2: Build CoT prompt ────────────────────────
    system_prompt, user_prompt = build_reflector_prompt(
        source_path=state["source_path"],
        source_code=state["source_code"],
        test_code=state["test_code"],
        latest_error=latest_error,
        error_history=error_history,
        retry_count=retry_count + 1,
        max_retries=max_retries,
    )

    # ── Step 3: Call LLM with structured output ─────────
    # `with_structured_output` binds a single internal tool whose schema is
    # ReflectionResult, then forces the LLM to call it via `tool_choice`.
    # DeepSeek thinking-mode models (deepseek-reasoner / v4 with thinking on)
    # do NOT support tool_choice → 400 error. So we must disable thinking
    # here for the same reason the Generator's Tool Use loop does.
    # The 4-step CoT is now driven explicitly by the prompt (Step 1 → 4),
    # not by the model's internal reasoning_content.
    console.print("[bold magenta]🧠 [Reflector][/]  Generating fix via CoT (structured)...")

    llm = get_llm(for_tool_use=True)   # tool_choice path requires thinking off
    structured_llm = llm.with_structured_output(ReflectionResult)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        result: ReflectionResult = structured_llm.invoke(messages)
    except Exception as e:
        # If structured output fails (provider rejects tool_choice, returns
        # malformed JSON, etc.), there's no point looping — same input would
        # fail the same way. Saturate retry_count to terminate immediately
        # via the existing route_after_executor path.
        logger.error(f"Structured output parse failed: {e}")
        console.print(
            f"[bold red]✗ [Reflector][/]  Structured output failed: {escape(str(e))}"
        )
        console.print(
            "[bold red]✗ [Reflector][/]  Aborting self-healing loop "
            "(no point retrying with the same broken contract)"
        )
        return {
            "test_code": state["test_code"],
            "retry_count": max_retries,        # saturate → next router pass goes to end_failed
            "status": "failed",
        }

    patched_test_code = result.fixed_code

    # ── Step 4: Surface diagnostic fields to the user ───
    # These weren't visible in free-text mode; now they're first-class.
    console.print(
        f"[dim magenta]  ├─ classification: {result.error_classification}[/]\n"
        f"[dim magenta]  ├─ root cause:    {escape(result.root_cause)}[/]\n"
        f"[dim magenta]  ├─ strategy:      {result.fix_strategy}[/]\n"
        f"[dim magenta]  └─ affected lines: {result.affected_lines or '(none reported)'}[/]"
    )

    # ── Step 5: Show diff ───────────────────────────────
    original_lines = state["test_code"].splitlines(keepends=True)
    patched_lines = patched_test_code.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        original_lines, patched_lines,
        fromfile="before", tofile="after", n=2,
    ))

    if not diff:
        console.print(
            "[bold yellow]⚠ [Reflector][/]  "
            "Patch is identical to original — LLM may be stuck"
        )
    else:
        console.print("[bold magenta]🧠 [Reflector][/]  Changes made:")
        for line in diff:
            line = line.rstrip("\n")
            if line.startswith("+") and not line.startswith("+++"):
                console.print(f"[green]{escape(line)}[/]")
            elif line.startswith("-") and not line.startswith("---"):
                console.print(f"[red]{escape(line)}[/]")
            elif line.startswith("@@"):
                console.print(f"[cyan]{escape(line)}[/]")

    console.print("[bold green]✓ [Reflector][/]  Patch applied, re-executing...")

    # ── Step 6: Return updated state ────────────────────
    return {
        "test_code": patched_test_code,
        "retry_count": retry_count + 1,
        "status": "executing",
    }