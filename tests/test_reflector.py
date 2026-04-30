"""
Tests for the reflector prompt builder module.

Covers:
- build_reflector_prompt with and without error history
- Edge cases: empty strings, single error in history, large retry counts
- Prompt template integrity (format placeholders present)
"""

import pytest

from auto_sdet.prompts.reflector import (
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
    build_reflector_prompt,
)


class TestBuildReflectorPrompt:
    """Tests for the build_reflector_prompt function."""

    # ------------------------------------------------------------------
    # Positive / happy-path tests
    # ------------------------------------------------------------------

    def test_returns_tuple_of_two_strings(self):
        """build_reflector_prompt should return a 2-tuple of strings."""
        result = build_reflector_prompt(
            source_path="/fake/path.py",
            source_code="def foo(): pass",
            test_code="def test_foo(): pass",
            latest_error="AssertionError: x != y",
            error_history=[],
            retry_count=1,
            max_retries=3,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_system_prompt_is_unchanged(self):
        """The system prompt returned should be the module-level constant."""
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="e",
            error_history=[],
            retry_count=0,
            max_retries=5,
        )
        # System prompt is always the raw constant
        sys_prompt, _ = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="e",
            error_history=[],
            retry_count=0,
            max_retries=5,
        )
        assert sys_prompt == REFLECTOR_SYSTEM_PROMPT

    def test_user_prompt_contains_all_placeholders_filled(self):
        """Every {placeholder} in the template should be replaced."""
        source_path = "/proj/src/calc.py"
        source_code = "def add(a, b): return a + b"
        test_code = "def test_add(): assert add(1, 2) == 3"
        latest_error = "AssertionError: assert 4 == 3"
        error_history = ["ImportError: no module named foo"]
        retry_count = 2
        max_retries = 4

        _, user = build_reflector_prompt(
            source_path=source_path,
            source_code=source_code,
            test_code=test_code,
            latest_error=latest_error,
            error_history=error_history,
            retry_count=retry_count,
            max_retries=max_retries,
        )

        # No raw placeholders should remain
        assert "{" not in user
        assert "}" not in user
        # Key values should appear verbatim
        assert source_path in user
        assert source_code in user
        assert test_code in user
        assert latest_error in user
        assert str(retry_count) in user
        assert str(max_retries) in user

    # ------------------------------------------------------------------
    # Error-history formatting
    # ------------------------------------------------------------------

    def test_error_history_empty_produces_first_attempt_message(self):
        """When error_history is empty, the placeholder text is used."""
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="e",
            error_history=[],
            retry_count=1,
            max_retries=3,
        )
        assert "(First attempt — no previous errors)" in user

    def test_error_history_single_entry(self):
        """A single error in history should be prefixed with '--- Attempt 1 ---'."""
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="latest",
            error_history=["SyntaxError: bad input"],
            retry_count=2,
            max_retries=3,
        )
        assert "--- Attempt 1 ---" in user
        assert "SyntaxError: bad input" in user
        assert "(First attempt" not in user

    def test_error_history_multiple_entries(self):
        """Multiple errors should each get their own attempt header."""
        history = [
            "ImportError: X",
            "TypeError: Y",
            "AssertionError: Z",
        ]
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="final",
            error_history=history,
            retry_count=3,
            max_retries=5,
        )
        assert "--- Attempt 1 ---" in user
        assert "--- Attempt 2 ---" in user
        assert "--- Attempt 3 ---" in user
        assert "ImportError: X" in user
        assert "TypeError: Y" in user
        assert "AssertionError: Z" in user

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_all_string_fields_can_be_empty(self):
        """Empty source/test/error strings should not cause crashes."""
        _, user = build_reflector_prompt(
            source_path="",
            source_code="",
            test_code="",
            latest_error="",
            error_history=[],
            retry_count=0,
            max_retries=1,
        )
        assert isinstance(user, str)
        # The empty strings should appear literally in the output
        assert 'path=""' in user or "path=" in user

    def test_retry_count_zero(self):
        """retry_count=0 is valid (first attempt before any retry)."""
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="e",
            error_history=[],
            retry_count=0,
            max_retries=3,
        )
        assert "Current retry: 0 / 3" in user

    def test_retry_count_equals_max_retries(self):
        """Boundary: retry_count == max_retries (last allowed attempt)."""
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="e",
            error_history=["err1"],
            retry_count=5,
            max_retries=5,
        )
        assert "Current retry: 5 / 5" in user

    def test_large_error_history(self):
        """A large error history should not break formatting."""
        history = [f"Error number {i}" for i in range(100)]
        _, user = build_reflector_prompt(
            source_path="p.py",
            source_code="s",
            test_code="t",
            latest_error="final",
            error_history=history,
            retry_count=99,
            max_retries=100,
        )
        assert "--- Attempt 1 ---" in user
        assert "--- Attempt 100 ---" in user
        assert "Error number 0" in user
        assert "Error number 99" in user

    # ------------------------------------------------------------------
    # Template integrity
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "placeholder",
        [
            "{source_path}",
            "{source_code}",
            "{test_code}",
            "{latest_error}",
            "{error_history_section}",
            "{retry_count}",
            "{max_retries}",
        ],
    )
    def test_user_prompt_template_contains_placeholder(self, placeholder):
        """The REFLECTOR_USER_PROMPT constant must contain all expected placeholders."""
        assert placeholder in REFLECTOR_USER_PROMPT, (
            f"Missing placeholder {placeholder} in REFLECTOR_USER_PROMPT"
        )

    def test_system_prompt_is_non_empty(self):
        """The system prompt constant should be a non-empty string."""
        assert isinstance(REFLECTOR_SYSTEM_PROMPT, str)
        assert len(REFLECTOR_SYSTEM_PROMPT) > 0

    def test_user_prompt_is_non_empty(self):
        """The user prompt constant should be a non-empty string."""
        assert isinstance(REFLECTOR_USER_PROMPT, str)
        assert len(REFLECTOR_USER_PROMPT) > 0