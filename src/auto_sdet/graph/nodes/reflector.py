"""
Reflector Node — analyzes test failures via CoT and generates patches.
"""
from __future__ import annotations

import difflib
import logging

from rich.markup import escape
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.graph.nodes.generator import extract_python_code
from auto_sdet.tools.llm_factory import get_llm
from auto_sdet.prompts.reflector import build_reflector_prompt

logger = logging.getLogger(__name__)
console = Console()


def reflector_node(state: AgentState) -> dict:
    """Reflector Node: analyze failure via CoT, call LLM, return patched test code."""
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

    # ── Step 3: Call LLM ────────────────────────────────
    console.print("[bold magenta]🧠 [Reflector][/]  Generating fix via CoT...")

    llm = get_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # ── Step 4: Extract patched code ────────────────────
    patched_test_code = extract_python_code(response.content)

    # ── Show diff ───────────────────────────────────────
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

    # ── Step 5: Return updated state ────────────────────
    return {
        "test_code": patched_test_code,
        "retry_count": retry_count + 1,
        "status": "executing",        # Next: back to Executor
    }
