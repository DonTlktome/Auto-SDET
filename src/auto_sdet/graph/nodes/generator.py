"""
Generator Node — uses LLM Tool Use (tool_call / function calling) to gather
context, then emits the final pytest test file.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from rich.console import Console

from auto_sdet.models.schemas import AgentState
from auto_sdet.tools.mcp_context import MCPContextManager
from auto_sdet.tools.mcp_tools import GENERATOR_TOOLS, execute_tool_call
from auto_sdet.tools.llm_factory import get_llm, get_provider_label
from auto_sdet.prompts.generator import GENERATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
console = Console()

# Hard cap on tool-calling iterations. Keeps token cost bounded if the LLM
# decides to read every file in sight. 6 is enough for typical 1-3 deps.
MAX_TOOL_ITERATIONS = 6


def extract_python_code(text: str) -> str:
    """
    Extract Python code from an LLM response.

    Strategy:
    1. Prefer ```python ... ``` fenced code blocks (largest one wins).
    2. If no fence is found, try to salvage by trimming prose before the
       first plausible Python construct (import / def / class / from).
       This handles V4-style responses that mix narration with code.
    3. As a last resort return the raw text.
    """
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return max(matches, key=len).strip()

    logger.warning("No code block found in LLM response, attempting prose-strip fallback")

    # Find the first line that looks like Python source and drop everything before.
    code_start_pattern = re.compile(
        r"^(import |from |def |class |@|#!|#.*coding)",
        re.MULTILINE,
    )
    match = code_start_pattern.search(text)
    if match:
        return text[match.start():].strip()

    return text.strip()


def _build_initial_user_message(source_path: str) -> str:
    return (
        f"Generate a complete pytest test file for the source file at:\n"
        f"  {source_path}\n\n"
        f"Use the available tools to read the target file and any internal "
        f"dependencies you need. When you have enough context, emit the "
        f"final test code as a single ```python ... ``` block."
    )


def generator_node(state: AgentState) -> dict:
    """
    Generator Node: drives a tool-calling loop where the LLM autonomously
    reads files via MCP tools, then emits the final test code.
    """
    source_path = state["source_path"]
    console.print(f"[bold cyan]✨ [Generator][/]  Calling LLM ({get_provider_label()}) with tools...")

    # ── Bind tools to the LLM ─────────────────────────────
    # for_tool_use=True disables DeepSeek V4 thinking to avoid the
    # reasoning_content round-trip 400 error in multi-turn tool calls.
    llm = get_llm(for_tool_use=True)
    llm_with_tools = llm.bind_tools(GENERATOR_TOOLS)

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=_build_initial_user_message(source_path)),
    ]

    # ── Tool-calling loop ─────────────────────────────────
    # Track which files the LLM ended up reading so we can echo them out
    # to the user and persist them for the Executor sandbox uploader.
    files_read: dict[str, str] = {}
    # Cache of tool results keyed by (name, frozen-args). Repeating the same
    # call should not re-execute or burn tokens reading the same file twice.
    call_cache: dict[tuple[str, str], str] = {}
    final_response = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            final_response = response
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            cache_key = (tool_name, repr(sorted(tool_args.items())))

            if cache_key in call_cache:
                tool_result = call_cache[cache_key]
                console.print(
                    f"[dim yellow]  → tool_call (cached): {tool_name}({tool_args})[/]"
                )
            else:
                console.print(
                    f"[dim cyan]  → tool_call: {tool_name}({tool_args})[/]"
                )
                tool_result = execute_tool_call(tool_name, tool_args)
                call_cache[cache_key] = tool_result

            # Track read files so the sandbox uploader has them later
            if tool_name == "read_file":
                requested_path = tool_args.get("path", "")
                if requested_path and not tool_result.startswith("Error"):
                    files_read[requested_path] = tool_result

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
            )
    else:
        # Loop exited without a final response — force one more call
        # without tools so the LLM is compelled to produce code.
        console.print(
            f"[bold yellow]⚠ [Generator][/]  "
            f"Tool-call iteration cap ({MAX_TOOL_ITERATIONS}) reached, forcing final answer"
        )
        final_response = llm.invoke(messages)

    # ── Extract code ──────────────────────────────────────
    test_code = extract_python_code(final_response.content)

    # ── Reconstruct source_code + context_files for downstream nodes ──
    # The Executor expects state["source_code"] (the target file content)
    # and state["context_files"] (filename -> content) so it can upload them.
    source_code = files_read.get(source_path, "")
    if not source_code:
        # Fallback: LLM never read the target file via tool. Read directly.
        try:
            source_code = MCPContextManager().read_file(source_path).content
        except Exception as e:
            logger.warning(f"Fallback read_file failed: {e}")

    context_files: dict[str, str] = {}
    target_path_obj = Path(source_path).resolve()
    for path_str, content in files_read.items():
        if Path(path_str).resolve() == target_path_obj:
            continue
        context_files[Path(path_str).name] = content

    line_count = len(test_code.strip().splitlines())
    console.print(
        f"[bold green]✓ [Generator][/]  "
        f"Generated test_{Path(source_path).name} "
        f"({line_count} lines, {len(files_read)} file(s) read via tools)"
    )

    return {
        "source_code": source_code,
        "context_files": context_files,
        "test_code": test_code,
        "status": "executing",
    }
