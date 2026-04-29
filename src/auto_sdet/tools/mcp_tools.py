"""
LangChain tool wrappers around MCPContextManager.

Exposing file-system access as `@tool`-decorated functions allows the LLM
to invoke them via `tool_call` / function calling, giving the agent
true Tool Use capability rather than fixed Python invocation.
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from auto_sdet.tools.mcp_context import MCPContextManager

logger = logging.getLogger(__name__)

# Single shared MCP manager — read-only operations, safe to reuse.
_mcp = MCPContextManager()

# Cap each file payload to keep tool-result tokens bounded.
# Most well-factored Python source files fit in 30K chars; only sprawling
# multi-implementation files (e.g. TheAlgorithms/fibonacci.py at ~12K) need
# the larger budget. Truncation forces the LLM to fabricate signatures.
MAX_FILE_CHARS = 30000


@tool
def read_file(path: str) -> str:
    """
    Read the full text content of a Python source file at the given absolute path.

    Use this to inspect the target file or any dependency files referenced by
    its imports. Returns the file content as a string. If the file is larger
    than the cap, the content is truncated and a notice is appended.

    Args:
        path: Absolute path to the file. Must exist and be a regular file.
    """
    try:
        content = _mcp.read_file(path).content
        if len(content) > MAX_FILE_CHARS:
            return content[:MAX_FILE_CHARS] + f"\n\n[... truncated, {len(content) - MAX_FILE_CHARS} more chars ...]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def list_directory(path: str, pattern: str = "*.py") -> list[str]:
    """
    List files in a directory matching a glob pattern.

    Use this to discover sibling or dependency files when planning what to
    read with `read_file`. Returns a list of absolute file paths.

    Args:
        path: Absolute path to a directory.
        pattern: Glob pattern, e.g. "*.py" (default) or "test_*.py".
    """
    try:
        return _mcp.list_directory(path, pattern)
    except Exception as e:
        return [f"Error listing directory: {e}"]


# Expose a stable list for binding
GENERATOR_TOOLS = [read_file, list_directory]


def execute_tool_call(name: str, args: dict) -> str:
    """
    Manually dispatch a tool call by name, returning a string result.
    Used inside the Generator's tool-calling loop instead of relying on
    a pre-built executor, so we keep tight control over iteration limits.
    """
    if name == "read_file":
        return read_file.invoke(args)
    if name == "list_directory":
        result = list_directory.invoke(args)
        return "\n".join(result) if isinstance(result, list) else str(result)
    return f"Unknown tool: {name}"
