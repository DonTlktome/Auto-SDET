"""
Tests for tools.ruff_checker — the static-analysis sidecar signal that
complements the LLM-as-Judge Evaluator.

These tests assume `ruff` is on PATH (declared as a project dependency in
pyproject.toml). If it isn't, the helpers fall back gracefully (verified
in test_missing_ruff_returns_empty).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from auto_sdet.tools.ruff_checker import (
    RuffIssue,
    check_code,
    format_issues_for_prompt,
    _is_fatal,
)


# ──────────────────────────────────────────────────────────────────
# _is_fatal — pure logic
# ──────────────────────────────────────────────────────────────────

class TestIsFatal:
    def test_undefined_name_is_fatal(self):
        assert _is_fatal("F821") is True

    def test_invalid_syntax_is_fatal(self):
        assert _is_fatal("invalid-syntax") is True

    def test_e9xx_prefix_is_fatal(self):
        assert _is_fatal("E999") is True
        assert _is_fatal("E901") is True

    def test_unused_import_is_not_fatal(self):
        assert _is_fatal("F401") is False

    def test_style_warnings_are_not_fatal(self):
        assert _is_fatal("E501") is False     # line too long

    def test_none_is_not_fatal(self):
        assert _is_fatal(None) is False
        assert _is_fatal("") is False


# ──────────────────────────────────────────────────────────────────
# check_code — end-to-end against the real ruff binary
# ──────────────────────────────────────────────────────────────────

class TestCheckCode:
    def test_clean_code_returns_no_issues(self):
        clean = "def add(a, b):\n    return a + b\n"
        assert check_code(clean) == []

    def test_undefined_name_is_detected_as_fatal(self):
        code = (
            "def test_x():\n"
            "    result = undefined_function(1, 2)\n"
            "    assert result == 3\n"
        )
        issues = check_code(code)
        assert any(i.code == "F821" and i.is_fatal for i in issues)

    def test_syntax_error_is_detected_as_fatal(self):
        broken = "def foo(\n  pass\n"
        issues = check_code(broken)
        fatals = [i for i in issues if i.is_fatal]
        assert len(fatals) >= 1, f"expected fatal issue, got {issues}"

    def test_unused_import_is_detected_as_nonfatal(self):
        code = "import numpy\n\ndef test_x():\n    assert 1 == 1\n"
        issues = check_code(code)
        f401 = [i for i in issues if i.code == "F401"]
        assert len(f401) == 1
        assert f401[0].is_fatal is False

    def test_line_number_is_populated(self):
        code = "import numpy\n\ndef test_x():\n    assert 1 == 1\n"
        issues = check_code(code)
        f401 = next(i for i in issues if i.code == "F401")
        # Unused import is on line 1
        assert f401.line == 1


class TestCheckCodeFallback:
    """check_code should never raise — failures degrade to empty list."""

    def test_missing_ruff_returns_empty(self):
        with patch(
            "auto_sdet.tools.ruff_checker.subprocess.run",
            side_effect=FileNotFoundError("ruff not on PATH"),
        ):
            assert check_code("x = 1") == []

    def test_timeout_returns_empty(self):
        import subprocess
        with patch(
            "auto_sdet.tools.ruff_checker.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ruff", timeout=1),
        ):
            assert check_code("x = 1") == []

    def test_malformed_json_returns_empty(self):
        from unittest.mock import MagicMock
        fake = MagicMock(stdout="not-json{}{", stderr="")
        with patch(
            "auto_sdet.tools.ruff_checker.subprocess.run",
            return_value=fake,
        ):
            assert check_code("x = 1") == []


# ──────────────────────────────────────────────────────────────────
# format_issues_for_prompt — rendering
# ──────────────────────────────────────────────────────────────────

class TestFormatIssuesForPrompt:
    def test_empty_list_renders_clean_message(self):
        out = format_issues_for_prompt([])
        assert "no static-analysis issues" in out

    def test_fatal_issues_get_fatal_tag(self):
        issues = [RuffIssue(code="F821", message="Undefined name `x`", line=3, column=0, is_fatal=True)]
        out = format_issues_for_prompt(issues)
        assert "[FATAL]" in out
        assert "F821" in out
        assert "Undefined name" in out

    def test_lint_issues_get_lint_tag(self):
        issues = [RuffIssue(code="F401", message="`numpy` imported but unused", line=1, column=0, is_fatal=False)]
        out = format_issues_for_prompt(issues)
        assert "[lint]" in out
        assert "F401" in out

    def test_fatal_issues_render_before_nonfatal(self):
        issues = [
            RuffIssue(code="F401", message="lint", line=1, column=0, is_fatal=False),
            RuffIssue(code="F821", message="fatal", line=2, column=0, is_fatal=True),
        ]
        out = format_issues_for_prompt(issues)
        # The fatal entry should appear above the lint entry in the rendered block.
        assert out.index("F821") < out.index("F401")

    def test_caps_at_max_entries(self):
        many = [
            RuffIssue(code="F401", message=f"unused {i}", line=i, column=0, is_fatal=False)
            for i in range(20)
        ]
        out = format_issues_for_prompt(many, max_entries=3)
        assert "and 17 more" in out

    def test_summary_counts_fatals_and_lints(self):
        issues = [
            RuffIssue(code="F821", message="x", line=1, column=0, is_fatal=True),
            RuffIssue(code="F401", message="y", line=2, column=0, is_fatal=False),
            RuffIssue(code="F401", message="z", line=3, column=0, is_fatal=False),
        ]
        out = format_issues_for_prompt(issues)
        assert "1 fatal" in out
        assert "2 lint" in out