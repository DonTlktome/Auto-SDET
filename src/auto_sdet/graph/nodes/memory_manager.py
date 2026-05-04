"""
Memory Manager Node — Memory-R1 style CRUD controller for episodic memory.

Sits between Reflector and Evaluator in the closed-loop quality gate.
Decides whether the trajectory just produced by the Reflector should:
  - be ADDED to memory (novel experience)
  - UPDATE an existing similar but inferior trajectory
  - DELETE (soft) an existing trajectory now proven wrong
  - NOOP (no change — memory already covers this)

Why this exists (vs. Reflexion-style append-only):
  Append-only memory monotonically grows. After many runs, retrieval
  returns near-duplicates and the signal-to-noise ratio collapses.
  Memory-R1 (Lu et al., 2025) introduces the {ADD, UPDATE, DELETE, NOOP}
  operation set to keep memory lean and current.

Safety guards (defense against LLM error):
  - DELETE is soft (mark deprecated, data preserved)
  - DELETE auto-downgrades to NOOP if confidence < 'high'
  - Manager output validation failure → caller defaults to ADD
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.markup import escape

from auto_sdet.models.schemas import AgentState, MemoryOperation, MemoryTrajectory
from auto_sdet.prompts.memory_manager import build_memory_manager_prompt
from auto_sdet.tools.llm_factory import get_llm
from auto_sdet.tools.memory_store import MemoryStore

logger = logging.getLogger(__name__)
console = Console()

# Number of neighbors retrieved to inform the CRUD decision.
# Balance: enough context to spot duplicates / contradictions, but not so
# many that the prompt balloons.
NEIGHBOR_K = 3


def memory_manager_node(state: AgentState) -> dict:
    """
    LLM-driven CRUD controller for the trajectory just produced by Reflector.

    Reads from state:
      - pending_trajectory: the MemoryTrajectory the Reflector just emitted

    Returns no state changes (writes only to the persistent memory store).
    Always passes through to the next node — memory ops never block the agent.
    """
    pending: MemoryTrajectory | None = state.get("pending_trajectory")
    if pending is None:
        # Nothing to manage (Reflector wasn't run, or it failed before emitting)
        return {}

    memory = MemoryStore()

    # ── Step 1: Retrieve neighbors (include deprecated, so Manager can see
    # what's been wrong before and avoid suggesting we re-DELETE them) ──
    neighbors = memory.retrieve_similar(
        pending.error_signature,
        k=NEIGHBOR_K,
        include_deprecated=True,
    )

    console.print(
        f"[bold blue]🗂  [MemoryManager][/]  "
        f"Deciding CRUD for new trajectory ({len(neighbors)} neighbor(s) retrieved)..."
    )

    # ── Step 2: Ask the LLM to decide the operation ──
    system_prompt, user_prompt = build_memory_manager_prompt(pending, neighbors)
    llm = get_llm(for_tool_use=True)   # tool_choice path: thinking off
    structured_llm = llm.with_structured_output(MemoryOperation)

    try:
        op: MemoryOperation = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        # Manager failure: default to ADD (conservative — keep the data).
        # Memory writes never block the agent, so we proceed regardless.
        logger.warning(f"MemoryManager structured output failed: {e}")
        console.print(
            f"[bold yellow]⚠ [MemoryManager][/]  "
            f"Decision LLM failed ({escape(str(e))[:60]}…) — defaulting to ADD"
        )
        memory.add_trajectory(pending)
        return {"pending_trajectory": None}

    # ── Step 3: Validate + apply the operation ──
    op = _enforce_safety_guards(op, neighbors)
    _apply_operation(op, pending, neighbors, memory)

    return {"pending_trajectory": None}


# ──────────────────────────────────────────────────────────────────
# Safety + application
# ──────────────────────────────────────────────────────────────────

def _enforce_safety_guards(
    op: MemoryOperation,
    neighbors: list[MemoryTrajectory],
) -> MemoryOperation:
    """
    Apply safety downgrades before executing the operation.

    Rules:
      1. DELETE with confidence < high  →  downgrade to NOOP
      2. UPDATE / DELETE with target_id not in neighbors  →  downgrade to NOOP
         (defends against the LLM hallucinating a trajectory id)
      3. ADD / NOOP with a target_trajectory_id set  →  drop the spurious id
    """
    neighbor_ids = {n.trajectory_id for n in neighbors}

    # Rule 1: high-risk DELETE without high confidence → NOOP
    if op.operation == "DELETE" and op.confidence != "high":
        console.print(
            f"[dim yellow]🗂  [MemoryManager][/]  "
            f"DELETE downgraded to NOOP (confidence='{op.confidence}', "
            f"need 'high')"
        )
        return op.model_copy(update={
            "operation": "NOOP",
            "target_trajectory_id": None,
            "reasoning": op.reasoning + " [SAFETY DOWNGRADE: DELETE→NOOP, low confidence]",
        })

    # Rule 2: UPDATE/DELETE pointing at a non-neighbor → hallucination, downgrade
    if op.operation in ("UPDATE", "DELETE"):
        if not op.target_trajectory_id or op.target_trajectory_id not in neighbor_ids:
            console.print(
                f"[dim yellow]🗂  [MemoryManager][/]  "
                f"{op.operation} target id not in neighbor set — downgraded to NOOP"
            )
            return op.model_copy(update={
                "operation": "NOOP",
                "target_trajectory_id": None,
                "reasoning": op.reasoning + " [SAFETY DOWNGRADE: target id hallucinated]",
            })

    # Rule 3: ADD/NOOP shouldn't carry a target id
    if op.operation in ("ADD", "NOOP") and op.target_trajectory_id:
        return op.model_copy(update={"target_trajectory_id": None})

    return op


def _apply_operation(
    op: MemoryOperation,
    pending: MemoryTrajectory,
    neighbors: list[MemoryTrajectory],
    memory: MemoryStore,
) -> None:
    """Execute the validated operation against the memory store."""
    if op.operation == "ADD":
        memory.add_trajectory(pending)
        console.print(
            f"[bold green]🗂  [MemoryManager][/]  ADD: persisted as new trajectory "
            f"(memory now: {memory.count()} live)\n"
            f"[dim]   reason: {escape(op.reasoning)}[/]"
        )

    elif op.operation == "UPDATE":
        ok = memory.update_trajectory(op.target_trajectory_id, pending)
        verb = "UPDATE" if ok else "UPDATE failed"
        console.print(
            f"[bold cyan]🗂  [MemoryManager][/]  {verb}: replaced "
            f"trajectory {op.target_trajectory_id[:8]}…\n"
            f"[dim]   reason: {escape(op.reasoning)}[/]"
        )

    elif op.operation == "DELETE":
        ok = memory.mark_deprecated(op.target_trajectory_id, op.reasoning)
        verb = "DELETE (soft)" if ok else "DELETE failed"
        console.print(
            f"[bold red]🗂  [MemoryManager][/]  {verb}: deprecated "
            f"trajectory {op.target_trajectory_id[:8]}…\n"
            f"[dim]   reason: {escape(op.reasoning)}[/]"
        )

    elif op.operation == "NOOP":
        console.print(
            f"[dim]🗂  [MemoryManager][/]  NOOP: memory already covers this pattern\n"
            f"[dim]   reason: {escape(op.reasoning)}[/]"
        )