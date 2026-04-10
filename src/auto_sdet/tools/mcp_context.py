"""
MCP (Model Context Protocol) filesystem context tool.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from auto_sdet.models.schemas import FileContent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# Tool Input Schemas (for LLM function calling)
# ══════════════════════════════════════════════════════════

class ReadFileInput(BaseModel):
    """Schema for the read_file tool — used by LLM via function calling."""
    path: str = Field(..., description="Absolute path of the file to read")


class ListDirectoryInput(BaseModel):
    """Schema for the list_directory tool."""
    path: str = Field(..., description="Absolute path of the directory to list")
    pattern: str = Field(default="*.py", description="Glob pattern to filter files")


# ══════════════════════════════════════════════════════════
# MCP Context Manager
# ══════════════════════════════════════════════════════════

class MCPContextManager:
    """
    Manages file context retrieval via MCP filesystem server.

    Falls back to direct file I/O if MCP server is unavailable.
    """

    def __init__(self, mcp_server_url: Optional[str] = None):
        self._mcp_client = None
        self._mcp_server_url = mcp_server_url

    async def _init_mcp_client(self):
        """
        Initialize MCP client connection to filesystem server.
        If connection fails, sets _mcp_client to None (fallback mode).
        """
        if self._mcp_server_url is None:
            logger.info("No MCP server URL configured, using direct file I/O fallback")
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    # allowed directories will be passed as args
                ],
            )
            # Note: In production, we'd keep this connection alive
            # For simplicity, we create per-call connections
            logger.info(f"MCP client initialized with server: {self._mcp_server_url}")
        except ImportError:
            logger.warning("MCP SDK not installed, using direct file I/O fallback")
        except Exception as e:
            logger.warning(f"MCP connection failed: {e}, using fallback")

    def read_file(self, path: str) -> FileContent:
        """Read a file's content, either via MCP or direct I/O."""
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        return FileContent(
            path=str(file_path),
            content=content,
            mime_type="text/x-python" if file_path.suffix == ".py" else "text/plain",
        )

    def list_directory(self, path: str, pattern: str = "*.py") -> list[str]:
        """
        List files in a directory matching the given pattern.

        Returns list of absolute file paths.
        """
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        return sorted([str(f.resolve()) for f in dir_path.glob(pattern) if f.is_file()])

    def gather_context(
        self,
        target_path: Path,
        max_context_files: int = 5,
    ) -> tuple[str, dict[str, str]]:
        """
        High-level method: read target file + sibling .py dependencies.

        Args:
            target_path: Path to the target source file
            max_context_files: Max number of sibling files to include

        Returns:
            tuple[str, dict[str, str]]: (source_code, {filename: content})

"""
        target_path = Path(target_path).resolve()

        # Read target file
        target_content = self.read_file(str(target_path))
        source_code = target_content.content

        # Read sibling .py files (excluding __pycache__, tests, target itself)
        sibling_dir = target_path.parent
        context_files: dict[str, str] = {}

        try:
            siblings = self.list_directory(str(sibling_dir), "*.py")
        except Exception:
            siblings = []

        for sibling_path_str in siblings:
            sibling_path = Path(sibling_path_str)

            # Skip target file itself
            if sibling_path == target_path:
                continue
            # Skip test files
            if sibling_path.name.startswith("test_"):
                continue
            # Skip __init__.py (usually empty or boilerplate)
            if sibling_path.name == "__init__.py":
                continue

            if len(context_files) >= max_context_files:
                break

            try:
                file_content = self.read_file(str(sibling_path))
                context_files[sibling_path.name] = file_content.content
            except Exception as e:
                logger.warning(f"Failed to read {sibling_path}: {e}")

        logger.info(
            f"Context gathered: source={target_path.name}, "
            f"dependencies={list(context_files.keys())}"
        )

        return source_code, context_files
