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

from pathlib import Path

from auto_sdet.config import get_settings
from auto_sdet.models.schemas import AgentState, ReflectionResult
from auto_sdet.tools.error_signature import extract_error_signature
from auto_sdet.tools.llm_factory import get_llm
from auto_sdet.tools.memory_store import build_trajectory, get_memory_store
from auto_sdet.tools.run_trace import clip, trace_event
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

    # ── Step 2: Reflexion Memory — retrieve similar past cases ──────
    # legacy:    query = first 300 chars of the output + filename. In
    #            practice that is mostly the pytest session header.
    # signature: query = normalized error signature extracted from the
    #            output — the same representation the documents are stored in.
    memory = get_memory_store()
    signature_variant = get_settings().memory_variant == "signature"
    target_file = Path(state["source_path"]).name
    module_id = state.get("module_id") or target_file
    if signature_variant:
        query_text = extract_error_signature(latest_error)
    else:
        query_text = f"{latest_error[:300]} [in {target_file}]"
    similar_cases, candidates = (
        memory.retrieve_with_candidates(query_text, k=3) if memory else ([], [])
    )

    # Experiment trace (observability only). `retry` is the attempt this fix
    # is for — the execution event with the same (module, retry) says whether
    # it worked.
    trace = {
        "module": module_id,
        "retry": retry_count + 1,
        "trigger": _trigger_of(execution_result),
        "memory_enabled": memory is not None,
        "latest_error": clip(latest_error),
        "query": query_text if memory else None,
        "candidates": candidates,
        "injected": [
            c.model_dump(include={
                "trajectory_id", "source_module", "target_file", "error_signature",
                "error_classification", "root_cause", "fix_strategy", "fix_summary", "outcome",
            })
            for c in similar_cases
        ],
        "test_before": clip(state["test_code"]),
    }

    if memory is None:
        console.print("[dim cyan]🧠 [Reflector][/]  Episodic memory disabled — skipping retrieval")
    elif similar_cases:
        console.print(
            f"[dim cyan]🧠 [Reflector][/]  Retrieved {len(similar_cases)} similar past case(s) "
            f"from episodic memory (total: {memory.count()})"
        )
    else:
        console.print(
            f"[dim cyan]🧠 [Reflector][/]  No similar past cases in memory "
            f"(total memory size: {memory.count()})"
        )

    # ── Step 3: Build CoT prompt with memory context ─────
    system_prompt, user_prompt = build_reflector_prompt(
        source_path=state["source_path"],
        source_code=state["source_code"],
        test_code=state["test_code"],
        latest_error=latest_error,
        error_history=error_history,
        retry_count=retry_count + 1,
        max_retries=max_retries,
        similar_cases=similar_cases,
    )

    # ── Step 3: Call LLM with structured output ─────────
    # `with_structured_output(method="json_mode")` uses `response_format` rather
    # than `tool_choice`, which is what deepseek-reasoner accepts alongside
    # thinking. (The default function_calling method uses tool_choice and
    # 400s on deepseek-reasoner — verified empirically; see DeepSeek docs at
    # https://api-docs.deepseek.com/zh-cn/guides/thinking_mode.)
    # Thinking ON gives Reflector real reasoning depth for "diagnose + patch"
    # tasks; that recovers the one-shot pass rate lost in earlier benchmark
    # runs that ran this node with thinking disabled.
    console.print("[bold magenta]🧠 [Reflector][/]  Generating fix via CoT (structured + thinking)...")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Structured output can fail in two ways:
    #   1. invoke() raises — most often the response hit max_tokens because
    #      thinking consumed the budget, so the JSON is cut off; or the JSON
    #      is malformed / fails schema validation
    #   2. invoke() returns None (LLM produced empty content)
    # Recovery: ask once more with thinking DISABLED. Without reasoning tokens
    # the JSON fits in the budget; the prompt is unchanged.
    failures: list[dict] = []
    llm = get_llm(multi_turn_tool_calling=False)   # single-turn → keep thinking ON
    result = _invoke_structured(llm, messages, failures, thinking=True)
    if result is None:
        console.print(
            f"[bold yellow]⚠ [Reflector][/]  Structured output failed "
            f"({failures[-1]['kind']}) — retrying once with thinking disabled"
        )
        fallback_llm = get_llm(multi_turn_tool_calling=False, disable_thinking=True)
        result = _invoke_structured(fallback_llm, messages, failures, thinking=False)

    counters = _attempt_counters(state, failures, fix_produced=result is not None)

    if result is None:
        # One failed repair attempt, not a reason to give up: the router
        # decides whether retries remain. The test code is unchanged, so the
        # Executor re-runs the same file and the loop continues normally.
        logger.error(f"Structured output unavailable after recovery: {failures}")
        console.print(
            f"[bold red]✗ [Reflector][/]  No usable fix this attempt "
            f"({', '.join(f['kind'] for f in failures)}) — counted as a failed attempt"
        )
        trace_event("reflection", **trace, fix_produced=False, structured_failures=failures)
        return {
            "test_code": state["test_code"],
            "retry_count": retry_count + 1,
            "status": "executing",
            **counters,
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

    # ── Step 6: Build pending trajectory (Memory Manager will commit) ────
    # We no longer write to memory directly. The Memory Manager (next node)
    # decides ADD / UPDATE / DELETE / NOOP based on what's already in store.
    # This is the Memory-R1 (Lu 2025) upgrade over Reflexion's append-only
    # model — see graph/nodes/memory_manager.py for rationale.
    # With memory disabled nothing is emitted, so MemoryManager is a no-op.
    # signature variant: the document is the same normalized signature used as
    # the query (so the root_cause is never embedded or truncated), and the
    # outcome stays "pending" until the Executor has re-run pytest.
    pending_trajectory = None
    if memory is not None:
        try:
            pending_trajectory = build_trajectory(
                error_classification=result.error_classification,
                root_cause=result.root_cause,
                fix_strategy=result.fix_strategy,
                fix_summary=_extract_fix_summary(diff),
                target_file=target_file,
                outcome="pending" if signature_variant else "fix_applied",
                source_module=module_id,
            )
            if signature_variant:
                pending_trajectory = pending_trajectory.model_copy(update={"error_signature": query_text})
        except Exception as e:
            # Trajectory build failure is non-fatal — Memory Manager will
            # see pending_trajectory=None and skip its CRUD logic.
            logger.warning(f"Failed to build pending trajectory: {e}")

    trace_event(
        "reflection",
        **trace,
        fix_produced=True,
        classification=result.error_classification,
        root_cause=result.root_cause,
        fix_strategy=result.fix_strategy,
        test_after=clip(patched_test_code),
        pending_trajectory_id=pending_trajectory.trajectory_id if pending_trajectory else None,
        structured_failures=failures,
    )

    # ── Step 7: Return updated state ────────────────────
    return {
        "test_code": patched_test_code,
        "retry_count": retry_count + 1,
        "status": "executing",
        "pending_trajectory": pending_trajectory,
        **counters,
    }


def _invoke_structured(llm, messages, failures: list[dict], thinking: bool) -> ReflectionResult | None:
    """Invoke the json_mode structured LLM; on failure record why and return None."""
    structured_llm = llm.with_structured_output(ReflectionResult, method="json_mode")
    try:
        result = structured_llm.invoke(messages)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        kind = "truncated" if "length limit" in detail or "LengthFinishReason" in detail else "parse_error"
        failures.append({"thinking": thinking, "kind": kind, "detail": detail[:300]})
        return None
    if result is None:
        failures.append({"thinking": thinking, "kind": "empty", "detail": "invoke returned None"})
    return result


def _attempt_counters(state: AgentState, failures: list[dict], fix_produced: bool) -> dict:
    """Running per-module totals of repair attempts and structured-output failures."""
    so_failures = dict(state.get("structured_output_failures") or {})
    for f in failures:
        so_failures[f["kind"]] = so_failures.get(f["kind"], 0) + 1
    if failures and fix_produced:
        so_failures["recovered_with_thinking_off"] = so_failures.get("recovered_with_thinking_off", 0) + 1
    return {
        "reflector_attempts": state.get("reflector_attempts", 0) + 1,
        "reflector_failed_attempts": state.get("reflector_failed_attempts", 0) + (0 if fix_produced else 1),
        "structured_output_failures": so_failures,
    }


def _trigger_of(execution_result) -> str:
    """What sent us to the Reflector: a real pytest run, the LLM judge, or ruff."""
    if execution_result is None:
        return "unknown"
    return {-100: "evaluator_reject", -101: "ruff_fatal"}.get(execution_result.exit_code, "pytest")


def _extract_fix_summary(diff: list[str], max_chars: int = 300) -> str:
    """
    Build a compact summary of the diff for storage in memory.

    Stores the first few +/- lines (not the full file) so the memory record
    stays small and the summary itself is human-readable when retrieved.
    """
    if not diff:
        return "(no changes detected — possible identical patch)"

    interesting = [
        line.rstrip("\n")
        for line in diff
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    ]
    summary = "; ".join(interesting[:6])   # cap at 6 changed lines
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "…"
    return summary or "(diff content not summarizable)"