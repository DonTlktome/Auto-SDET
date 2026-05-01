"""
Splitter Node — entry point of the graph. Reads the source file and
extracts top-level public symbols (functions / classes).

Routing logic (lives in graph/router.py):
- 0 or 1 symbol → fall back to single-shot Generator (with Tool Use)
- 2+ symbols   → fan out via Send to one generator_worker per symbol
"""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.tools.ast_extractor import extract_top_level_symbols
from auto_sdet.tools.mcp_context import MCPContextManager

logger = logging.getLogger(__name__)
console = Console()


def splitter_node(state: AgentState) -> dict:
    """
    Read source file, extract top-level symbols, decide whether to fan out.

    The actual fan-out / fall-back routing happens in the conditional edge
    (`route_after_splitter`); this node just populates `symbols` and
    `source_code` so the router can read them.
    """
    source_path = state["source_path"]
    console.print(f"[bold blue]🔀 [Splitter][/]  Analyzing {Path(source_path).name}...")

    # Read source via MCP (no LLM, just file I/O)
    mcp = MCPContextManager()
    try:
        source_code = mcp.read_file(source_path).content
    except Exception as e:
        logger.error(f"Splitter could not read source: {e}")
        # Empty source → 0 symbols → router will fall back to single-shot
        return {
            "source_code": "",
            "symbols": [],
            "status": "splitting",
        }

    symbols = extract_top_level_symbols(source_code)

    if not symbols:
        console.print(
            f"[bold blue]🔀 [Splitter][/]  No public symbols extracted "
            f"(empty / syntax error / private-only) — falling back to single-shot Generator"
        )
    elif len(symbols) == 1:
        console.print(
            f"[bold blue]🔀 [Splitter][/]  1 symbol found ({symbols[0].name}) "
            f"— single-shot Generator is more efficient than fan-out"
        )
    else:
        names = ", ".join(s.name for s in symbols)
        console.print(
            f"[bold blue]🔀 [Splitter][/]  {len(symbols)} symbols found: [bold]{names}[/]\n"
            f"          Fanning out to {len(symbols)} parallel workers"
        )

    return {
        "source_code": source_code,
        "symbols": symbols,
        "status": "splitting",
    }