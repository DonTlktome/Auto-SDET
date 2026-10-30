"""
Hand-written tests for the AST + import-rewriting helpers in e2b_sandbox.

Coverage focus:
  - `_fix_imports`: rewrites full package paths to flat filename imports
  - `_collect_internal_deps`: AST-walks source code to resolve project-internal
    file dependencies and load them
"""
from __future__ import annotations

from pathlib import Path

import pytest

from auto_sdet.tools import e2b_sandbox
from auto_sdet.tools.e2b_sandbox import (
    InfraError,
    SandboxExecutor,
    _collect_internal_deps,
    _collect_third_party_deps,
    _fix_imports,
)
from auto_sdet.models.schemas import ExecutionResult


# ══════════════════════════════════════════════════════════════════
# _fix_imports
# ══════════════════════════════════════════════════════════════════

class TestFixImports:
    """Verify regex-based import rewriting for sandbox flat structure."""

    # ── from-import rewrite ─────────────────────────────────────

    def test_from_with_single_parent_package_rewrites(self):
        code = "from auto_sdet.router import x"
        assert _fix_imports(code, "router") == "from router import x"

    def test_from_with_deeply_nested_package_rewrites(self):
        code = "from auto_sdet.graph.nodes.router import x"
        assert _fix_imports(code, "router") == "from router import x"

    def test_from_with_multiple_imports_rewrites(self):
        code = "from auto_sdet.models.schemas import AgentState, ExecutionResult"
        result = _fix_imports(code, "schemas")
        assert result == "from schemas import AgentState, ExecutionResult"

    def test_plain_from_import_unchanged(self):
        """`from module import x` (no parent path) stays as-is."""
        code = "from router import x"
        assert _fix_imports(code, "router") == code

    # ── import-as rewrite ───────────────────────────────────────

    def test_import_as_rewrites_keeping_alias(self):
        code = "import maths.fibonacci as fib_mod"
        result = _fix_imports(code, "fibonacci")
        assert result.startswith("import fibonacci")
        assert "as fib_mod" in result

    def test_import_with_parent_package_rewrites(self):
        code = "import auto_sdet.router"
        # Trailing context is end-of-string, captured by the (\s|$) group
        result = _fix_imports(code, "router")
        assert result.strip() == "import router"

    def test_plain_import_unchanged(self):
        """`import module` (no parent) should not be modified."""
        code = "import router"
        assert _fix_imports(code, "router") == code

    # ── unrelated imports untouched ─────────────────────────────

    def test_other_modules_unchanged(self):
        """Imports of unrelated modules must not be rewritten."""
        code = "from langchain_core.messages import SystemMessage"
        assert _fix_imports(code, "router") == code

    def test_partial_name_match_not_rewritten(self):
        """`from auto_sdet.router_helpers import x` should NOT match `router`."""
        code = "from auto_sdet.router_helpers import x"
        # `router_helpers` is a different module name; the regex requires
        # \.{module_name}\s, so it won't match the prefix.
        assert _fix_imports(code, "router") == code

    # ── multi-line and mixed scenarios ──────────────────────────

    def test_multi_line_mixed_imports(self):
        code = (
            "from auto_sdet.graph.router import route_after_executor\n"
            "from langchain_core.messages import HumanMessage\n"
            "import auto_sdet.models.schemas as s\n"
            "import os\n"
        )
        result = _fix_imports(_fix_imports(code, "router"), "schemas")
        assert "from router import route_after_executor" in result
        assert "from langchain_core.messages import HumanMessage" in result
        assert "import schemas as s" in result
        assert "import os" in result

    def test_idempotent(self):
        """Running _fix_imports twice should be a no-op the second time."""
        code = "from auto_sdet.graph.router import x"
        once = _fix_imports(code, "router")
        twice = _fix_imports(once, "router")
        assert once == twice


# ══════════════════════════════════════════════════════════════════
# _collect_internal_deps
# ══════════════════════════════════════════════════════════════════

class TestCollectInternalDeps:
    """Verify AST-based dependency resolution."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        """
        Construct a minimal mock project structure:

            tmp/
              pyproject.toml
              src/
                pkg/
                  main.py
                  utils.py
                  models/
                    schemas.py
        """
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        src = tmp_path / "src"
        pkg = src / "pkg"
        models = pkg / "models"
        models.mkdir(parents=True)

        (pkg / "utils.py").write_text("def helper(): return 1\n")
        (models / "schemas.py").write_text("class Schema: pass\n")
        return tmp_path

    def test_resolves_single_internal_import(self, project: Path):
        main = project / "src" / "pkg" / "main.py"
        source_code = "from pkg.utils import helper\n"
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert "utils.py" in deps
        assert "def helper" in deps["utils.py"]

    def test_resolves_nested_package_import(self, project: Path):
        main = project / "src" / "pkg" / "main.py"
        source_code = "from pkg.models.schemas import Schema\n"
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert "schemas.py" in deps
        assert "class Schema" in deps["schemas.py"]

    def test_external_imports_not_included(self, project: Path):
        """3rd-party / stdlib imports must not appear in deps."""
        main = project / "src" / "pkg" / "main.py"
        source_code = (
            "import os\n"
            "import json\n"
            "from langchain_core.messages import SystemMessage\n"
        )
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert deps == {}

    def test_self_import_skipped(self, project: Path):
        """A file shouldn't include itself in its own deps."""
        main = project / "src" / "pkg" / "main.py"
        # Reference itself by package path (degenerate case)
        source_code = "# pkg.main is self — should be skipped\n"
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert "main.py" not in deps

    def test_syntax_error_returns_empty(self, project: Path):
        """Unparseable source → AST raises → return {}."""
        main = project / "src" / "pkg" / "main.py"
        source_code = "def broken(:\n    pass\n"   # invalid syntax
        deps = _collect_internal_deps(str(main), source_code)
        assert deps == {}

    def test_empty_source_returns_empty(self, project: Path):
        main = project / "src" / "pkg" / "main.py"
        deps = _collect_internal_deps(str(main), "")
        assert deps == {}

    def test_multiple_imports_all_resolved(self, project: Path):
        main = project / "src" / "pkg" / "main.py"
        source_code = (
            "from pkg.utils import helper\n"
            "from pkg.models.schemas import Schema\n"
        )
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert "utils.py" in deps
        assert "schemas.py" in deps
        assert len(deps) == 2

    def test_import_module_form(self, project: Path):
        """Plain `import pkg.utils` should resolve too."""
        main = project / "src" / "pkg" / "main.py"
        source_code = "import pkg.utils\n"
        main.write_text(source_code)

        deps = _collect_internal_deps(str(main), source_code)
        assert "utils.py" in deps

    def test_no_pyproject_falls_back_to_parent(self, tmp_path: Path):
        """When there's no pyproject.toml, src_root falls back to file's parent."""
        flat_dir = tmp_path / "flat"
        flat_dir.mkdir()
        (flat_dir / "helper.py").write_text("def x(): pass\n")
        main = flat_dir / "main.py"
        main.write_text("from helper import x\n")

        deps = _collect_internal_deps(str(main), main.read_text())
        # Without src/, resolution uses the file's parent dir as src_root
        assert "helper.py" in deps


# ══════════════════════════════════════════════════════════════════
# _collect_third_party_deps
# ══════════════════════════════════════════════════════════════════

class TestCollectThirdPartyDeps:
    """Only real third-party imports of the code under test get pip-installed."""

    def test_stdlib_only_module_needs_nothing(self):
        code = "import doctest\nfrom __future__ import annotations\nfrom math import sqrt\n"
        assert _collect_third_party_deps([code], local_modules=set()) == []

    def test_third_party_import_detected(self):
        code = "import numpy as np\nfrom numpy import ndarray\n"
        assert _collect_third_party_deps([code], local_modules=set()) == ["numpy"]

    def test_import_name_mapped_to_distribution(self):
        code = "import cv2\nfrom PIL import Image\nfrom sklearn.linear_model import Lasso\nimport yaml\n"
        deps = _collect_third_party_deps([code], local_modules=set())
        assert deps == ["pillow", "opencv-python", "scikit-learn", "pyyaml"]

    def test_uploaded_local_modules_excluded(self):
        code = "from stack import Stack\nimport helpers\n"
        assert _collect_third_party_deps([code], local_modules={"stack", "helpers"}) == []

    def test_relative_imports_excluded(self):
        code = "from .stack import Stack\nfrom . import utils\n"
        assert _collect_third_party_deps([code], local_modules=set()) == []

    def test_project_packages_excluded(self, tmp_path: Path):
        (tmp_path / "data_structures").mkdir()
        (tmp_path / "helpers.py").write_text("x = 1\n")
        code = "from data_structures.stacks.stack import Stack\nimport helpers\nimport numpy\n"
        deps = _collect_third_party_deps([code], local_modules=set(), src_root=tmp_path)
        assert deps == ["numpy"]

    def test_agent_own_dependencies_never_injected(self):
        """The old hard-coded langgraph/langchain list must not come back."""
        code = "def add(a, b):\n    return a + b\n"
        assert _collect_third_party_deps([code], local_modules=set()) == []

    def test_multiple_sources_merged_and_deduplicated(self):
        deps = _collect_third_party_deps(
            ["import numpy\n", "import numpy as np\nimport scipy\n"], local_modules=set(),
        )
        assert deps == ["numpy", "scipy"]

    def test_unparseable_source_skipped(self):
        deps = _collect_third_party_deps(["def broken(:\n", "import numpy\n"], local_modules=set())
        assert deps == ["numpy"]


# ══════════════════════════════════════════════════════════════════
# SandboxExecutor infra retry
# ══════════════════════════════════════════════════════════════════

class TestInfraRetry:
    """Infra failures are retried with a fresh sandbox and kept out of test results."""

    @pytest.fixture
    def executor(self, monkeypatch) -> SandboxExecutor:
        monkeypatch.setenv("E2B_API_KEY", "test-key")
        monkeypatch.setattr(e2b_sandbox.time, "sleep", lambda s: None)
        return SandboxExecutor(timeout=60)

    def test_transient_failure_retried_then_succeeds(self, executor, monkeypatch):
        calls = {"n": 0}

        def fake_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise InfraError("sandbox create failed: 502")
            return ExecutionResult(stdout="1 passed", exit_code=0, sandbox_id="sbx")

        monkeypatch.setattr(executor, "_execute_once", fake_once)
        result = executor.execute_test("/x/mod.py", "", "")
        assert result.exit_code == 0
        assert result.infra_failure is False
        assert result.infra_failures == 1
        assert result.infra_backoff_ms == e2b_sandbox.INFRA_BACKOFF_S[0] * 1000

    def test_persistent_failure_flagged_not_reported_as_test_failure(self, executor, monkeypatch):
        def always_fail(*args, **kwargs):
            raise InfraError("sandbox create failed: quota")

        monkeypatch.setattr(executor, "_execute_once", always_fail)
        result = executor.execute_test("/x/mod.py", "", "")
        assert result.infra_failure is True
        assert result.infra_failures == e2b_sandbox.INFRA_MAX_ATTEMPTS
        assert len(result.infra_errors) == e2b_sandbox.INFRA_MAX_ATTEMPTS

    def test_test_failure_is_not_retried(self, executor, monkeypatch):
        calls = {"n": 0}

        def failing_tests(*args, **kwargs):
            calls["n"] += 1
            return ExecutionResult(stdout="1 failed", exit_code=1, sandbox_id="sbx")

        monkeypatch.setattr(executor, "_execute_once", failing_tests)
        result = executor.execute_test("/x/mod.py", "", "")
        assert calls["n"] == 1
        assert result.exit_code == 1
        assert result.infra_failures == 0

