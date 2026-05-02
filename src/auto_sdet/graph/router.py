"""
Conditional Edge Routers.

  - route_after_evaluator: pass to Executor or fall back to Reflector,
    based on the status field set by the Evaluator
  - route_after_executor: success / retry / give up
"""
from __future__ import annotations

import logging

from rich.console import Console

from auto_sdet.models.schemas import AgentState

logger = logging.getLogger(__name__)
console = Console()


def route_after_evaluator(state: AgentState) -> str:
    """
    Read the status the Evaluator set and dispatch.

    The Evaluator has all the context (score thresholds, reject counter,
    actionable weaknesses) so it makes the decision and just communicates
    it via state["status"]. The router's job is only to map status values
    to graph nodes — it stays a pure function with no business logic.
    """
    status = state.get("status", "executing")
    if status == "reflecting":
        return "reflector"
    return "executor"


def route_after_executor(state: AgentState) -> str:
    """
    Routing function called after Executor Node.

    Returns "end_success", "reflect", or "end_failed".
    """
    execution_result = state.get("execution_result")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # ── Case 1: Tests passed ────────────────────────────
    if execution_result and execution_result.exit_code == 0:
        console.print("[bold green]→ [Router][/]  Tests passed! Routing to END.")
        return "end_success"

    # ── Case 2: Tests failed, retries available ─────────
    if retry_count < max_retries:
        console.print(
            f"[bold magenta]→ [Router][/]  Tests failed. "
            f"Routing to Reflector ({retry_count + 1}/{max_retries})"
        )
        return "reflect"

    # ── Case 3: Tests failed, no more retries ───────────
    console.print(
        f"[bold red]→ [Router][/]  Max retries ({max_retries}) reached. "
        f"Routing to END (failed)."
    )
    return "end_failed"
