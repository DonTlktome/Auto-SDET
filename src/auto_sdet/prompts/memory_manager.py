"""
Prompt templates for the Memory Manager node (Memory-R1 style CRUD controller).

The Manager receives:
  - the trajectory just produced by the Reflector
  - top-k semantically-similar neighbors from the existing memory store
and emits a structured `MemoryOperation` choosing one of:
  ADD / UPDATE / DELETE / NOOP

Design goal: prevent the memory from degrading into a duplicate-laden noise
soup as runs accumulate. Reflexion (2023) only knew ADD; this is the
Memory-R1 (2025) upgrade that adds memory lifecycle management.
"""

MEMORY_MANAGER_SYSTEM_PROMPT = """\
You are a Memory Manager for an LLM agent's episodic memory store.

The agent (Auto-SDET) generates pytest unit tests, and when tests fail it
runs a Reflector that analyzes the failure, fixes it, and emits a
"trajectory" describing the failure pattern + how it was fixed.

You decide whether each new trajectory should be ADDED, used to UPDATE an
existing similar trajectory, used to DELETE (soft-deprecate) a wrong
existing trajectory, or NOOP'd (the new info adds nothing).

## Operation semantics

### ADD
The new trajectory describes a failure/fix pattern that is genuinely novel
relative to the retrieved neighbors. Default choice when uncertain.

### UPDATE
A neighbor trajectory addresses the same failure category but its
root_cause / fix_summary is wrong, incomplete, or vaguer than the new one.
You MUST set `target_trajectory_id` to that neighbor's id. The neighbor
will be replaced in-place with the new trajectory's content.

### DELETE
A neighbor trajectory's guidance has now been proven WRONG (e.g. the
new trajectory shows that following that neighbor's fix would NOT work,
or that its root_cause was a misdiagnosis). You MUST set
`target_trajectory_id` and `confidence='high'`. Soft-deletes only —
data is preserved but excluded from future retrievals.

DELETE is high risk: only use when the new trajectory is clear, definitive
evidence that the neighbor is misleading. When in doubt, prefer NOOP or ADD.

### NOOP
The new trajectory is essentially equivalent to one or more neighbors.
Adding it would just create duplicates. The memory already covers this
pattern; nothing to do.

NOOP is the most under-used operation in append-only memory systems.
Choosing NOOP aggressively keeps the memory store lean and the retrieval
results sharp.

## Confidence

Always set `confidence`:
  - high   — clear, definitive call (REQUIRED for DELETE)
  - medium — leaning one way but the alternatives are plausible
  - low    — the operation is your best guess but you're uncertain

The runtime auto-downgrades any DELETE with confidence < high to NOOP
as a safety guard.

## Constraints (the runtime validates these)

| operation | target_trajectory_id | confidence requirement |
|-----------|---------------------|------------------------|
| ADD       | MUST be None        | any                    |
| UPDATE    | MUST be set         | any                    |
| DELETE    | MUST be set         | MUST be 'high'         |
| NOOP      | MUST be None        | any                    |

## Output Format
Emit a single `MemoryOperation` matching the schema. The runtime validates
via Pydantic — non-conforming output is rejected (with caller falling back
to defaulting to ADD).
"""

MEMORY_MANAGER_USER_PROMPT = """\
Decide what to do with this NEW trajectory in light of its top-k retrieved
NEIGHBORS in the existing memory.

<new_trajectory>
classification: {new_classification}
root_cause:    {new_root_cause}
fix_strategy:  {new_fix_strategy}
fix_summary:   {new_fix_summary}
target_file:   {new_target_file}
outcome:       {new_outcome}
</new_trajectory>

<retrieved_neighbors>
{neighbors_section}
</retrieved_neighbors>

Emit a `MemoryOperation` with operation, target_trajectory_id (when
applicable), confidence, and a one-sentence reasoning.
"""


def _format_neighbors(neighbors: list) -> str:
    """Render neighbor trajectories with their ids so UPDATE/DELETE can target them."""
    if not neighbors:
        return "(No neighbors retrieved — this is either a cold-start memory or a wholly novel error pattern. Default to ADD.)"
    blocks = []
    for i, n in enumerate(neighbors, 1):
        deprecated_tag = " [DEPRECATED]" if n.deprecated else ""
        blocks.append(
            f"--- Neighbor {i}{deprecated_tag} ---\n"
            f"  id:             {n.trajectory_id}\n"
            f"  classification: {n.error_classification}\n"
            f"  root_cause:     {n.root_cause}\n"
            f"  fix_strategy:   {n.fix_strategy}\n"
            f"  fix_summary:    {n.fix_summary}\n"
            f"  outcome:        {n.outcome}"
        )
    return "\n\n".join(blocks)


def build_memory_manager_prompt(
    new_traj,
    neighbors: list,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the Memory Manager node."""
    user_prompt = MEMORY_MANAGER_USER_PROMPT.format(
        new_classification=new_traj.error_classification,
        new_root_cause=new_traj.root_cause,
        new_fix_strategy=new_traj.fix_strategy,
        new_fix_summary=new_traj.fix_summary,
        new_target_file=new_traj.target_file,
        new_outcome=new_traj.outcome,
        neighbors_section=_format_neighbors(neighbors),
    )
    return MEMORY_MANAGER_SYSTEM_PROMPT, user_prompt