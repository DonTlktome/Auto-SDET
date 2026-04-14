"""
E2B Sandbox tool — executes generated tests in an isolated Micro-VM.
"""
from __future__ import annotations

import ast
import re
import time
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from auto_sdet.models.schemas import ExecutionResult
from auto_sdet.config import get_settings

logger = logging.getLogger(__name__)


def _fix_imports(code: str, module_name: str) -> str:
    """Replace full package paths to a module with just its filename."""
    code = re.sub(
        rf"from\s+[\w.]*\.{re.escape(module_name)}\s+import",
        f"from {module_name} import",
        code,
    )
    code = re.sub(
        rf"import\s+[\w.]*\.{re.escape(module_name)}(\s|$)",
        f"import {module_name}\\1",
        code,
    )
    return code


def _collect_internal_deps(source_path: str, source_code: str) -> dict[str, str]:
    """
    Parse imports in source_code, find internal project files, and return
    {filename: content} for each resolved dependency.
    """
    source_file = Path(source_path).resolve()

    # Walk up to find the project src root (contains pyproject.toml or src/)
    project_root = source_file.parent
    for parent in source_file.parents:
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break

    src_root = project_root / "src" if (project_root / "src").exists() else project_root

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    deps: dict[str, str] = {}

    for node in ast.walk(tree):
        module_str = None
        if isinstance(node, ast.ImportFrom) and node.module:
            module_str = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_str = alias.name

        if not module_str:
            continue

        parts = module_str.split(".")
        # Try to resolve as a file under src_root
        candidate = src_root.joinpath(*parts).with_suffix(".py")
        if candidate.exists() and candidate != source_file:
            filename = candidate.stem + ".py"
            try:
                content = candidate.read_text(encoding="utf-8")
                deps[filename] = content
            except Exception as e:
                logger.warning(f"Could not read dependency {candidate}: {e}")

    return deps


# ══════════════════════════════════════════════════════════
# Tool Input Schemas (Pydantic — for potential ReAct usage)
# ══════════════════════════════════════════════════════════

class InstallDependenciesInput(BaseModel):
    """Schema for install_dependencies tool."""
    packages: list[str] = Field(
        default=["pytest"],
        description="List of pip packages to install in sandbox",
    )


class WriteFileInput(BaseModel):
    """Schema for write_file tool."""
    path: str = Field(..., description="Path inside sandbox (e.g., /workspace/target.py)")
    content: str = Field(..., description="File content to write")


class RunPytestInput(BaseModel):
    """Schema for run_pytest tool."""
    test_file: str = Field(..., description="Path to test file inside sandbox")
    verbose: bool = Field(default=True, description="Run pytest with -v flag")


# ══════════════════════════════════════════════════════════
# E2B Sandbox Executor
# ══════════════════════════════════════════════════════════

class SandboxExecutor:
    """Manages the lifecycle of an E2B sandbox for test execution."""

    def __init__(self, timeout: int | None = None):
        self._settings = get_settings()
        self._timeout = timeout or self._settings.e2b_sandbox_timeout

    def execute_test(
        self,
        source_path: str,
        source_code: str,
        test_code: str,
        context_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a test file inside an E2B sandbox."""
        from e2b_code_interpreter import Sandbox
        from e2b.sandbox.commands.command_handle import CommandExitException

        sandbox = None
        start_time = time.time()

        try:
            # ── Step 1: Create sandbox ──────────────────
            sandbox = Sandbox.create(
                timeout=self._timeout,
                api_key=self._settings.e2b_api_key.get_secret_value(),
            )
            sandbox_id = sandbox.sandbox_id
            logger.info(f"Sandbox created: {sandbox_id}")

            # ── Step 2: Upload files ────────────────────
            source_filename = Path(source_path).name
            module_name = Path(source_path).stem   # e.g. "router"
            test_filename = f"test_{source_filename}"

            # Collect all internal project deps imported by the source file
            internal_deps = _collect_internal_deps(source_path, source_code)
            all_files = {**internal_deps, **(context_files or {})}

            # Rewrite full package imports → flat filename imports in test + source
            all_module_names = {Path(f).stem for f in all_files} | {module_name}
            for mod in all_module_names:
                test_code = _fix_imports(test_code, mod)
                source_code = _fix_imports(source_code, mod)
            for fname in list(all_files):
                fixed = all_files[fname]
                for mod in all_module_names:
                    fixed = _fix_imports(fixed, mod)
                all_files[fname] = fixed

            sandbox.files.write(f"/workspace/{source_filename}", source_code)
            sandbox.files.write(f"/workspace/{test_filename}", test_code)

            for filename, content in all_files.items():
                sandbox.files.write(f"/workspace/{filename}", content)

            # ── Step 3: Install dependencies ────────────
            try:
                sandbox.commands.run(
                    "pip install pytest pytest-cov langgraph langchain-openai langchain-core pydantic pydantic-settings",
                    timeout=60,
                )
            except CommandExitException as e:
                logger.warning(f"pip install warning: {e.stderr}")

            # ── Step 4: Run pytest with coverage ────────
            # --cov=<module_name>  targets the source module (no path, no .py)
            # --cov-report=term-missing  prints uncovered line numbers in stdout
            try:
                test_result = sandbox.commands.run(
                    f"cd /workspace && python -m pytest {test_filename} -v"
                    f" --cov={module_name} --cov-report=term-missing",
                    timeout=self._timeout,
                )
                stdout, stderr, exit_code = test_result.stdout, test_result.stderr, test_result.exit_code
            except CommandExitException as e:
                stdout, stderr, exit_code = e.stdout, e.stderr, e.exit_code

            # ── Step 5: Parse coverage percentage ───────
            # pytest-cov output format: "module.py   25   3   88%   45, 67"
            coverage_pct: int | None = None
            match = re.search(
                rf"{re.escape(module_name)}\.py\s+\d+\s+\d+\s+(\d+)%",
                stdout,
            )
            if match:
                coverage_pct = int(match.group(1))

            duration_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                sandbox_id=sandbox_id,
                coverage_pct=coverage_pct,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Sandbox execution failed: {e}")

            return ExecutionResult(
                stdout="",
                stderr=f"Sandbox execution error: {str(e)}",
                exit_code=-1,
                duration_ms=duration_ms,
                sandbox_id=sandbox.sandbox_id if sandbox else "N/A",
            )

        finally:
            # ── Step 6: Always destroy sandbox ──────────
            if sandbox:
                try:
                    sandbox.kill()
                    logger.info(f"Sandbox destroyed: {sandbox.sandbox_id}")
                except Exception as e:
                    logger.warning(f"Failed to destroy sandbox: {e}")
