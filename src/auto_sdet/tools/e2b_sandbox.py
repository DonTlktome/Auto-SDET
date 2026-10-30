"""
E2B Sandbox tool — executes generated tests in an isolated Micro-VM.
"""
from __future__ import annotations

import ast
import re
import sys
import time
import logging
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from auto_sdet.models.schemas import ExecutionResult
from auto_sdet.config import get_settings

logger = logging.getLogger(__name__)

# The only packages the test run itself needs. Everything else installed in
# the sandbox must come from the code under test (see _collect_third_party_deps).
# Production note: a custom E2B template with these pre-installed would remove
# the per-sandbox pip step entirely; we install on demand so benchmark runs
# stay reproducible against the stock code-interpreter template.
BASE_TEST_PACKAGES = ("pytest", "pytest-cov")

# Import name → PyPI distribution name, only where the two differ.
_IMPORT_TO_DIST = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "Crypto": "pycryptodome",
    "attr": "attrs",
}

# Infrastructure failures (sandbox create / upload / pip / sandbox death) are
# retried here and never reach the Reflector — they say nothing about the
# test code, so letting the LLM "fix" them only burns a retry.
INFRA_MAX_ATTEMPTS = 3
INFRA_BACKOFF_S = (5, 15)
PIP_TIMEOUT_S = 120
# Sandbox lifetime must outlive create + upload + pip + the full pytest budget,
# otherwise E2B reaps the VM mid-run and a slow pip shows up as a test failure.
SANDBOX_LIFETIME_MARGIN_S = 180

# pytest's own exit codes (0 ok, 1 failed, 2 interrupted/collection error,
# 3 internal error, 4 usage error, 5 no tests collected).
PYTEST_TIMEOUT_EXIT_CODE = 124


class InfraError(Exception):
    """Sandbox-side failure unrelated to the test code under evaluation."""


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


def _find_src_root(source_path: str) -> Path:
    """Walk up to the project root (contains pyproject.toml) and return its src root."""
    source_file = Path(source_path).resolve()
    project_root = source_file.parent
    for parent in source_file.parents:
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break
    return project_root / "src" if (project_root / "src").exists() else project_root


def _collect_internal_deps(source_path: str, source_code: str) -> dict[str, str]:
    """
    Parse imports in source_code, find internal project files, and return
    {filename: content} for each resolved dependency.
    """
    source_file = Path(source_path).resolve()
    src_root = _find_src_root(source_path)

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


def _collect_third_party_deps(
    codes: Iterable[str],
    local_modules: set[str],
    src_root: Path | None = None,
) -> list[str]:
    """
    AST-walk the code under test and return the pip distributions it needs.

    Excluded: stdlib (`sys.stdlib_module_names`), relative imports, modules
    uploaded into the flat /workspace (`local_modules`), and top-level
    packages that live inside the project (`src_root/<name>` exists).
    """
    top_level: set[str] = set()
    for code in codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level.add(node.module.split(".")[0])

    deps: list[str] = []
    for name in sorted(top_level):
        if name in sys.stdlib_module_names or name in local_modules:
            continue
        if src_root is not None and (
            (src_root / name).is_dir() or (src_root / f"{name}.py").exists()
        ):
            continue
        deps.append(_IMPORT_TO_DIST.get(name, name))
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
        # `_timeout` is the pytest budget; the sandbox itself lives longer
        # (see SANDBOX_LIFETIME_MARGIN_S) so setup time never eats into it.
        self._timeout = timeout or self._settings.e2b_sandbox_timeout

    def execute_test(
        self,
        source_path: str,
        source_code: str,
        test_code: str,
        context_files: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a test file inside a fresh E2B sandbox.

        Infrastructure failures are retried up to INFRA_MAX_ATTEMPTS times with
        a new sandbox each time. If every attempt fails, the result carries
        `infra_failure=True` so the router ends the run instead of handing a
        sandbox error to the Reflector as if it were a test failure.
        """
        start_time = time.time()
        timings = {"sandbox_create_ms": 0, "upload_ms": 0, "pip_install_ms": 0,
                   "pytest_ms": 0, "infra_backoff_ms": 0}
        infra_errors: list[str] = []

        for attempt in range(1, INFRA_MAX_ATTEMPTS + 1):
            try:
                result = self._execute_once(
                    source_path, source_code, test_code, context_files, timings,
                )
                return result.model_copy(update={
                    **timings,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "infra_failures": len(infra_errors),
                    "infra_errors": infra_errors,
                })
            except InfraError as e:
                infra_errors.append(str(e)[:500])
                logger.warning(f"Sandbox infra failure (attempt {attempt}/{INFRA_MAX_ATTEMPTS}): {e}")
                if attempt < INFRA_MAX_ATTEMPTS:
                    backoff = INFRA_BACKOFF_S[min(attempt - 1, len(INFRA_BACKOFF_S) - 1)]
                    time.sleep(backoff)
                    timings["infra_backoff_ms"] += backoff * 1000

        return ExecutionResult(
            stdout="",
            stderr=(
                f"Sandbox infrastructure error after {INFRA_MAX_ATTEMPTS} attempts: "
                f"{infra_errors[-1] if infra_errors else 'unknown'}"
            ),
            exit_code=-1,
            duration_ms=int((time.time() - start_time) * 1000),
            sandbox_id="N/A",
            infra_failure=True,
            infra_failures=len(infra_errors),
            infra_errors=infra_errors,
            **timings,
        )

    def _execute_once(
        self,
        source_path: str,
        source_code: str,
        test_code: str,
        context_files: dict[str, str] | None,
        timings: dict[str, int],
    ) -> ExecutionResult:
        """One sandbox lifecycle. Raises InfraError for sandbox-side failures."""
        from e2b_code_interpreter import Sandbox
        from e2b.exceptions import TimeoutException
        from e2b.sandbox.commands.command_handle import CommandExitException

        def _elapsed_ms(t0: float) -> int:
            return int((time.perf_counter() - t0) * 1000)

        # ── Step 1: Create sandbox ──────────────────
        t0 = time.perf_counter()
        try:
            sandbox = Sandbox.create(
                timeout=self._timeout + SANDBOX_LIFETIME_MARGIN_S,
                api_key=self._settings.e2b_api_key.get_secret_value(),
            )
        except Exception as e:
            raise InfraError(f"sandbox create failed: {e}") from e
        finally:
            timings["sandbox_create_ms"] += _elapsed_ms(t0)
        sandbox_id = sandbox.sandbox_id
        logger.info(f"Sandbox created: {sandbox_id}")

        try:
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

            t0 = time.perf_counter()
            try:
                sandbox.files.write(f"/workspace/{source_filename}", source_code)
                sandbox.files.write(f"/workspace/{test_filename}", test_code)
                for filename, content in all_files.items():
                    sandbox.files.write(f"/workspace/{filename}", content)
            except Exception as e:
                raise InfraError(f"file upload failed: {e}") from e
            finally:
                timings["upload_ms"] += _elapsed_ms(t0)

            # ── Step 3: Install dependencies ────────────
            # pytest + pytest-cov, plus whatever third-party packages the code
            # under test imports. The agent's own dependencies (langgraph etc.)
            # have no business inside the sandbox.
            third_party = _collect_third_party_deps(
                [source_code, *all_files.values()],
                local_modules=all_module_names,
                src_root=_find_src_root(source_path),
            )
            packages = [*BASE_TEST_PACKAGES, *third_party]
            pip_warning: str | None = None
            t0 = time.perf_counter()
            try:
                try:
                    sandbox.commands.run(
                        "pip install -q --disable-pip-version-check " + " ".join(packages),
                        timeout=PIP_TIMEOUT_S,
                    )
                except CommandExitException as e:
                    if not third_party:
                        raise InfraError(f"pip install failed: {e.stderr[-300:]}") from e
                    # A bad third-party guess must not take the base install
                    # down with it; the test will surface the ImportError.
                    pip_warning = f"third-party install failed ({' '.join(third_party)}): {e.stderr[-300:]}"
                    logger.warning(pip_warning)
                    try:
                        sandbox.commands.run(
                            "pip install -q --disable-pip-version-check " + " ".join(BASE_TEST_PACKAGES),
                            timeout=PIP_TIMEOUT_S,
                        )
                    except CommandExitException as e2:
                        raise InfraError(f"pip install failed: {e2.stderr[-300:]}") from e2
            except InfraError:
                raise
            except Exception as e:
                # TimeoutException, transport errors, sandbox reaped mid-install
                raise InfraError(f"pip install error: {e}") from e
            finally:
                timings["pip_install_ms"] += _elapsed_ms(t0)

            # ── Step 4: Run pytest with coverage ────────
            # --cov=<module_name>  targets the source module (no path, no .py)
            # --cov-report=term-missing  prints uncovered line numbers in stdout
            t0 = time.perf_counter()
            try:
                test_result = sandbox.commands.run(
                    f"cd /workspace && python -m pytest {test_filename} -v"
                    f" --cov={module_name} --cov-report=term-missing",
                    timeout=self._timeout,
                )
                stdout, stderr, exit_code = test_result.stdout, test_result.stderr, test_result.exit_code
            except CommandExitException as e:
                stdout, stderr, exit_code = e.stdout, e.stderr, e.exit_code
            except TimeoutException as e:
                # A hanging test (infinite loop, runaway recursion) is the test
                # code's fault; a dead sandbox is not.
                if not _sandbox_alive(sandbox):
                    raise InfraError(f"sandbox died during pytest: {e}") from e
                stdout, stderr = "", (
                    f"pytest did not finish within {self._timeout}s — a test is likely "
                    f"hanging (infinite loop, runaway recursion, or blocking input())."
                )
                exit_code = PYTEST_TIMEOUT_EXIT_CODE
            except Exception as e:
                raise InfraError(f"pytest command error: {e}") from e
            finally:
                timings["pytest_ms"] += _elapsed_ms(t0)

            if "No module named pytest" in (stderr or ""):
                raise InfraError("pytest missing in sandbox after install")

            # ── Step 5: Parse coverage percentage ───────
            # pytest-cov output format: "module.py   25   3   88%   45, 67"
            coverage_pct: int | None = None
            match = re.search(
                rf"{re.escape(module_name)}\.py\s+\d+\s+\d+\s+(\d+)%",
                stdout,
            )
            if match:
                coverage_pct = int(match.group(1))

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                sandbox_id=sandbox_id,
                coverage_pct=coverage_pct,
                pip_packages=packages,
                pip_warning=pip_warning,
            )

        finally:
            # ── Step 6: Always destroy sandbox ──────────
            try:
                sandbox.kill()
                logger.info(f"Sandbox destroyed: {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to destroy sandbox: {e}")


def _sandbox_alive(sandbox) -> bool:
    """Best-effort liveness probe used to tell a hanging test from a dead VM."""
    try:
        return bool(sandbox.is_running(request_timeout=10))
    except Exception:
        return False
