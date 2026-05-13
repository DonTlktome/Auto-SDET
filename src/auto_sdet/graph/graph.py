"""
LangGraph StateGraph assembly — the core FSM orchestration.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.graph.nodes.generator import generator_node
from auto_sdet.graph.nodes.evaluator import evaluator_node
from auto_sdet.graph.nodes.executor import executor_node
from auto_sdet.graph.nodes.reflector import reflector_node
from auto_sdet.graph.nodes.memory_manager import memory_manager_node
from auto_sdet.graph.router import route_after_executor, route_after_evaluator

console = Console()


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph FSM with asymmetric quality gating.

    Graph topology:

        [Generator]
            ↓
        [Evaluator] ─score≥0.5→ [Executor] ──pass──→ END (success)
            │                        │
            │                        └──fail──→ [Reflector]
            │                                       │
            │                                  [MemoryManager]    ← Memory-R1 CRUD
            │                                       │
            └─score<0.5──→ [Reflector]       (direct to Executor — retry path
                                              skips Evaluator since pytest will
                                              decide truth anyway)
                                                    ↓
                                                [Executor]

    Termination guarantees:
      - retry_count ≤ max_retries     (Executor → Reflector loop)
      - evaluator_reject_count ≤ 2    (kept for first-gen anti-loop safety)
      - Both counters strictly increase, so the FSM is bounded.

    Asymmetric quality gating (since 2026-05-06):
      First-gen output goes through Evaluator (catches obviously bad output
      before paying the sandbox cost). Retry path skips Evaluator and goes
      straight to Executor, because:
        1. pytest is ground truth — an LLM judge can't beat real execution
        2. Evaluator reject rate on retry path was empirically < 5% but
           every retry paid ~80s for it (ROI calculation)
        3. Reflector's fix is already conditioned on real stderr; the soft
           quality check adds noise more than signal

    Memory architecture: Memory-R1 (Lu 2025) style CRUD.
      Reflector emits a candidate trajectory into state.pending_trajectory,
      MemoryManager runs the {ADD, UPDATE, DELETE, NOOP} controller in a
      background thread (fire-and-forget) — memory ops never block the agent.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("memory_manager", memory_manager_node)

    # ── Entry point ─────────────────────────────────────
    graph.set_entry_point("generator")

    # ── Deterministic edges ─────────────────────────────
    # Asymmetric routing: first-gen goes through Evaluator, retry path skips it.
    #   Generator       → Evaluator     (first-gen quality gate)
    #   Reflector       → MemoryManager (sink the trajectory)
    #   MemoryManager   → Executor      (retry path bypasses Evaluator)
    graph.add_edge("generator", "evaluator")
    graph.add_edge("reflector", "memory_manager")
    graph.add_edge("memory_manager", "executor")

    # ── Evaluator → Executor (pass) or Reflector (reject) ─
    graph.add_conditional_edges(
        source="evaluator",
        path=route_after_evaluator,
        path_map={
            "executor": "executor",
            "reflector": "reflector",
        },
    )

    # ── Executor → success / retry / fail ──────────────
    graph.add_conditional_edges(
        source="executor",
        path=route_after_executor,
        path_map={
            "end_success": END,
            "reflect": "reflector",
            "end_failed": END,
        },
    )

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
        "source_code": "",            # Generator populates via Tool Use
        "context_files": {},
        "test_code": "",
        "evaluation_result": None,
        "evaluator_reject_count": 0,
        "pending_trajectory": None,   # Reflector → MemoryManager handoff
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
