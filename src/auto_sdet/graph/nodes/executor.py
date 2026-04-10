"""
Executor Node — runs generated tests in E2B sandbox, captures results.
"""
from __future__ import annotations

import logging

from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.tools.e2b_sandbox import SandboxExecutor

logger = logging.getLogger(__name__)
console = Console()


def executor_node(state: AgentState) -> dict:
    """Executor Node: run tests in E2B sandbox and capture results."""
    console.print("[bold yellow]⚡ [Executor][/]  Creating E2B sandbox...")

    executor = SandboxExecutor()

    # Run test in sandbox
    console.print("[bold yellow]⚡ [Executor][/]  Running pytest in sandbox...")

    result = executor.execute_test(
        source_path=state["source_path"],
        source_code=state["source_code"],
        test_code=state["test_code"],
        context_files=state.get("context_files"),
    )

    console.print(
        f"[bold yellow]⚡ [Executor][/]  "
        f"Sandbox {result.sandbox_id} | exit_code={result.exit_code} | "
        f"{result.duration_ms}ms"
    )

    # ── Collect output into error_history ───────────────
    error_history = list(state.get("error_history", []))
    full_output = (result.stdout + result.stderr).strip()
    if result.exit_code != 0 and full_output:
        error_history.append(full_output)

    # ── Log result ──────────────────────────────────────
    if result.exit_code == 0:
        console.print("[bold green]✓ [Executor][/]  All tests passed!")
    else:
        # Skip pytest header lines, show the actual error section
        error_start = full_output.find("ERRORS")
        if error_start == -1:
            error_start = full_output.find("ImportError")
        if error_start == -1:
            error_start = full_output.find("ERROR")
        output_preview = full_output[error_start:error_start + 500] if error_start != -1 else full_output[:500]
        console.print(f"[bold red]✗ [Executor][/]  Tests failed:\n[dim]{output_preview}[/]")

    return {
        "execution_result": result,
        "error_history": error_history,
        "status": "executing",
    }
