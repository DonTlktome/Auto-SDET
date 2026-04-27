"""
Generator Node — reads context via MCP + generates test code via LLM.
"""
from __future__ import annotations

import re
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.tools.mcp_context import MCPContextManager
from auto_sdet.prompts.generator import build_generator_prompt
from auto_sdet.config import get_settings

logger = logging.getLogger(__name__)
console = Console()


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response (```python ... ``` blocks)."""
    # Try to match ```python ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the longest match (most likely the complete test file)
        return max(matches, key=len).strip()

    # Fallback: if no code block found, assume entire response is code
    logger.warning("No code block found in LLM response, using entire response")
    return text.strip()


def generator_node(state: AgentState) -> dict:
    """
    Generator Node: read context via MCP, call LLM, return generated test code.
    """
    console.print("[bold cyan]✨ [Generator][/]  Reading file context (MCP)...")

    settings = get_settings()

    # ── Step 1: Gather context via MCP ──────────────────
    mcp = MCPContextManager()
    source_code, context_files = mcp.gather_context(
        target_path=state["source_path"]
    )

    console.print(
        f"[bold cyan]✨ [Generator][/]  "
        f"Context loaded: {len(context_files)} dependency file(s)"
    )

    # ── Step 2: Build prompt ────────────────────────────
    system_prompt, user_prompt = build_generator_prompt(
        source_path=state["source_path"],
        source_code=source_code,
        context_files=context_files,
    )

    # ── Step 3: Call LLM ────────────────────────────────
    console.print("[bold cyan]✨ [Generator][/]  Calling DeepSeek-V4...")

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        temperature=0.0,           # Deterministic output for code generation
        max_tokens=4096,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # ── Step 4: Extract code ────────────────────────────
    test_code = extract_python_code(response.content)

    line_count = len(test_code.strip().splitlines())
    console.print(
        f"[bold green]✓ [Generator][/]  "
        f"Generated test_{state['source_path'].split('/')[-1]} ({line_count} lines)"
    )

    # ── Step 5: Return state update ─────────────────────
    return {
        "source_code": source_code,
        "context_files": context_files,
        "test_code": test_code,
        "status": "executing",      # Next: Executor
    }
