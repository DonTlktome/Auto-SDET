"""
Generator Worker — single-symbol test generator used in parallel mode.

A worker is invoked once per Send dispatched by the Splitter conditional
edge. It receives one SymbolSpec plus a sibling-signature index in its
state, calls the LLM exactly once, and emits a partial test file that
the Combiner will later merge with siblings' partials.

Key differences from the original Generator node:
  - No Tool Use / multi-turn loop (all context is pre-packaged)
  - One LLM call total → simpler, cheaper, fully parallelizable
  - Returns `partial_test_codes: [code]` so the operator.add reducer
    accumulates outputs across concurrent workers
"""
from __future__ import annotations

import logging

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.tools.llm_factory import get_llm, get_provider_label
from auto_sdet.graph.nodes.generator import extract_python_code
from auto_sdet.prompts.generator_worker import build_worker_prompt

logger = logging.getLogger(__name__)
console = Console()


def generator_worker_node(state: AgentState) -> dict:
    """
    Generate tests for a single symbol. Single LLM call, no tools.

    Reads from state (populated by the Splitter Send):
      - current_symbol: SymbolSpec — the symbol to test
      - sibling_signatures: str    — context block of other symbols' signatures
      - source_path: str           — for the prompt's "Module path" line
    """
    symbol = state.get("current_symbol")
    siblings = state.get("sibling_signatures", "")
    source_path = state.get("source_path", "")

    if not symbol:
        # Defensive: should never happen if router is correct, but keep
        # the reducer healthy by emitting an empty partial.
        logger.error("generator_worker invoked without current_symbol")
        return {"partial_test_codes": [""]}

    console.print(
        f"[bold cyan]✨ [Worker:{symbol.name}][/]  "
        f"Calling LLM ({get_provider_label()})..."
    )

    system_prompt, user_prompt = build_worker_prompt(
        symbol_name=symbol.name,
        symbol_kind=symbol.kind,
        symbol_source=symbol.source,
        sibling_signatures=siblings,
        source_path=source_path,
    )

    # Workers run in parallel; the slowest one bounds total wall time.
    # V4 thinking adds 30-60s per worker for marginal quality gain on
    # mode-heavy single-symbol test generation. Disabling it here makes
    # the parallel fan-out actually fast. Reflector keeps thinking on
    # because debugging benefits more from V4's reasoning.
    llm = get_llm(for_tool_use=True)   # for_tool_use=True ⇒ thinking disabled
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    test_code = extract_python_code(response.content)
    line_count = len(test_code.splitlines())

    console.print(
        f"[bold green]✓ [Worker:{symbol.name}][/]  "
        f"Generated {line_count} lines"
    )

    # Single-element list — operator.add reducer in AgentState will
    # concatenate all workers' contributions into one list.
    return {"partial_test_codes": [test_code]}