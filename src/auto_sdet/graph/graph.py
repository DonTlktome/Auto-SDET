"""
LangGraph StateGraph assembly — the core FSM orchestration.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.graph.nodes.generator import generator_node
from auto_sdet.graph.nodes.executor import executor_node
from auto_sdet.graph.nodes.reflector import reflector_node
from auto_sdet.graph.router import route_after_executor

console = Console()


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph FSM.

    Graph topology:

        [Generator] → [Executor] ──→ END (success)
                          │
                          ├──→ [Reflector] → [Executor] (retry loop)
                          │
                          └──→ END (max retries exceeded)
    """
    # ── Create graph ────────────────────────────────────
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────
    graph.add_node("generator", generator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)

    # ── Set entry point ─────────────────────────────────
    graph.set_entry_point("generator")

    # ── Deterministic edges ─────────────────────────────
    graph.add_edge("generator", "executor")    # 生成后 → 执行
    graph.add_edge("reflector", "executor")    # 修复后 → 重新执行

    # ── Conditional edges (the decision point) ──────────
    graph.add_conditional_edges(
        source="executor",
        path=route_after_executor,
        path_map={
            "end_success": END,        # exit_code == 0
            "reflect": "reflector",     # exit_code != 0 + retries left
            "end_failed": END,          # exit_code != 0 + no retries left
        },
    )

    # ── Compile ─────────────────────────────────────────
    return graph.compile()


def run_agent(
    target_path: Path,
    output_path: Path,
    max_retries: int = 3,
    verbose: bool = False,
) -> dict:
    """Main entry point: build graph and execute."""
    # ── Build and compile graph ─────────────────────────
    app = build_graph()

    # ── Prepare initial state ───────────────────────────
    initial_state: AgentState = {
        "source_path": str(target_path.resolve()),
        "source_code": "",           # Will be filled by Generator
        "context_files": {},         # Will be filled by Generator
        "test_code": "",
        "execution_result": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "error_history": [],
        "status": "generating",
    }

    # ── Execute graph ───────────────────────────────────
    console.print("\n[dim]Starting LangGraph FSM...[/]\n")

    final_state = app.invoke(initial_state)

    # ── Determine final status ──────────────────────────
    execution_result = final_state.get("execution_result")

    if execution_result and execution_result.exit_code == 0:
        final_state["status"] = "done"

        # Write test file to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_state["test_code"], encoding="utf-8")

        console.print(f"\n[bold green]✨ Test file saved to:[/] {output_path}")
    else:
        final_state["status"] = "failed"

        # Still save the last attempt for debugging
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if execution_result:
            # Extract the meaningful error section (after the pytest header lines)
            full_output = execution_result.stdout + execution_result.stderr
            error_section = full_output[full_output.find("ERRORS"):] or full_output
            last_error = error_section[:300].strip()
        else:
            last_error = "Unknown"
        output_path.write_text(
            f"# AUTO-SDET: Test generation FAILED after "
            f"{final_state.get('retry_count', 0)} retries\n"
            f"# Last error: {last_error}\n\n"
            f"{final_state.get('test_code', '')}",
            encoding="utf-8",
        )

    return final_state
