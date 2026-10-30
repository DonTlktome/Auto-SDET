"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import builtins
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from palindrome import (
    benchmark_function,
    is_palindrome,
    is_palindrome_recursive,
    is_palindrome_slice,
    is_palindrome_traversal,
    test_data,
)

ALL_IMPLS = [
    is_palindrome,
    is_palindrome_traversal,
    is_palindrome_recursive,
    is_palindrome_slice,
]

IMPL_IDS = [f.__name__ for f in ALL_IMPLS]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------


def test_test_data_is_consistent_with_slicing():
    """The module-level test_data must agree with the naive slice reversal."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_both_outcomes():
    """test_data should exercise both True and False branches."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# Shared behaviour across all implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("text,expected", list(test_data.items()))
def test_all_implementations_match_test_data(func, text, expected):
    """Every implementation must reproduce the expected result for test_data."""
    assert func(text) is expected


@pytest.mark.parametrize("func", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),          # empty string is trivially a palindrome
        ("a", True),         # single character
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd length palindrome
        ("abba", True),      # even length palindrome
        ("abca", False),     # mismatch at the outer pair
        ("abcba", True),     # odd length palindrome, longer
        ("abccbx", False),   # mismatch near the middle
        ("  ", True),        # whitespace is not stripped
        ("a a", True),       # spaces count as characters
        ("a b", False),      # spaces count as characters
        ("Aa", False),       # case sensitive
        ("12321", True),     # digits
        ("12345", False),    # digits, not a palindrome
        ("!@#$%$#@!", True), # punctuation
    ],
)
def test_all_implementations_edge_cases(func, text, expected):
    """Edge cases: empty, single char, case sensitivity, non-letters."""
    assert func(text) is expected


@pytest.mark.parametrize("func", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_return_bool(func):
    """Each implementation must return an actual bool, not a truthy value."""
    assert isinstance(func("level"), bool)
    assert isinstance(func("String"), bool)


@pytest.mark.parametrize("func", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_agree_on_long_input(func):
    """All implementations must agree on a long generated palindrome."""
    half = "abcdefghijklmnopqrstuvwxyz"
    text = half + half[::-1]
    assert func(text) is True
    assert func(text + "z") is False


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------


def test_is_palindrome_happy_path():
    """A classic palindrome returns True."""
    assert is_palindrome("amanaplanacanalpanama") is True


def test_is_palindrome_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome("abcdba") is False


def test_is_palindrome_empty_string():
    """Empty string: loop body never runs, returns True."""
    assert is_palindrome("") is True


def test_is_palindrome_single_character():
    """Single character: start_i == end_i so loop never runs, returns True."""
    assert is_palindrome("A") is True


def test_is_palindrome_two_characters():
    """Two characters exercise exactly one loop iteration."""
    assert is_palindrome("BB") is True
    assert is_palindrome("AB") is False


def test_is_palindrome_does_not_mutate_input():
    """The input string must be unchanged after the call."""
    text = "rotor"
    is_palindrome(text)
    assert text == "rotor"


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


def test_is_palindrome_traversal_happy_path():
    """A palindrome returns True via half traversal."""
    assert is_palindrome_traversal("MALAYALAM") is True


def test_is_palindrome_traversal_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome_traversal("ABC") is False


def test_is_palindrome_traversal_empty_string():
    """Empty string: range(0) is empty, all() is True."""
    assert is_palindrome_traversal("") is True


def test_is_palindrome_traversal_single_character():
    """Single character: len // 2 == 0, so no comparisons are made."""
    assert is_palindrome_traversal("x") is True


def test_is_palindrome_traversal_ignores_middle_character():
    """For odd lengths the middle character is never compared."""
    assert is_palindrome_traversal("aXa") is True
    assert is_palindrome_traversal("aXb") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


def test_is_palindrome_recursive_happy_path():
    """A palindrome returns True through recursive peeling."""
    assert is_palindrome_recursive("level") is True


def test_is_palindrome_recursive_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome_recursive("String") is False


@pytest.mark.parametrize("text", ["", "a", "Z"])
def test_is_palindrome_recursive_base_cases(text):
    """Strings of length <= 1 hit the base case and return True."""
    assert is_palindrome_recursive(text) is True


def test_is_palindrome_recursive_deep_recursion():
    """A long palindrome must not exceed the recursion limit."""
    half = "ab" * 200
    text = half + half[::-1]
    assert text == text[::-1]
    assert is_palindrome_recursive(text) is True
    assert is_palindrome_recursive(text + "z") is False


def test_is_palindrome_recursive_mismatch_at_inner_pair():
    """Mismatch only at the innermost pair still returns False."""
    assert is_palindrome_recursive("abccba") is True
    assert is_palindrome_recursive("abccbx") is False


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


def test_is_palindrome_slice_happy_path():
    """Slice reversal comparison returns True for a palindrome."""
    assert is_palindrome_slice("rotor") is True


def test_is_palindrome_slice_non_palindrome():
    """Slice reversal comparison returns False for a non-palindrome."""
    assert is_palindrome_slice("abcdba") is False


def test_is_palindrome_slice_empty_string():
    """Empty string equals its own reversal."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared correctly."""
    assert is_palindrome_slice("été") is True
    assert is_palindrome_slice("étéx") is False


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result():
    """benchmark_function prints a formatted line containing the function name."""
    buffer = io.StringIO()
    with patch("palindrome.timeit", return_value=0.12345) as mock_timeit:
        with redirect_stdout(buffer):
            benchmark_function("is_palindrome_slice")

    output = buffer.getvalue()
    assert "is_palindrome_slice" in output
    assert "0.12345" in output
    assert "500,000" in output
    mock_timeit.assert_called_once()


def test_benchmark_function_builds_expected_timeit_arguments():
    """benchmark_function must pass the right stmt/setup/number to timeit."""
    with patch("palindrome.timeit", return_value=0.0) as mock_timeit:
        with redirect_stdout(io.StringIO()):
            benchmark_function("is_palindrome")

    _, kwargs = mock_timeit.call_args
    assert kwargs["stmt"] == (
        "all(is_palindrome(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"
    assert kwargs["number"] == 500000


def test_benchmark_function_returns_none():
    """benchmark_function is a side-effect-only helper and returns None."""
    with patch("palindrome.timeit", return_value=0.0):
        with redirect_stdout(io.StringIO()):
            assert benchmark_function("is_palindrome") is None


def test_benchmark_function_output_format():
    """The printed line uses the documented left-aligned 35-char name column."""
    buffer = io.StringIO()
    with patch("palindrome.timeit", return_value=1.5):
        with redirect_stdout(buffer):
            benchmark_function("is_palindrome")

    line = buffer.getvalue().rstrip("\n")
    assert line.startswith("is_palindrome".ljust(35))
    assert line.endswith("finished 500,000 runs in 1.50000 seconds")


def test_benchmark_function_does_not_use_real_timeit():
    """Guard: the real timeit must never run during tests (no long waits)."""
    with patch("palindrome.timeit", return_value=0.0) as mock_timeit:
        with redirect_stdout(io.StringIO()):
            benchmark_function("is_palindrome_recursive")
    assert mock_timeit.called


# ---------------------------------------------------------------------------
# Module import side effects
# ---------------------------------------------------------------------------


def test_module_import_does_not_run_benchmarks():
    """Importing the module must not execute the __main__ benchmark block."""
    import importlib
    import sys

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        sys.modules.pop("palindrome", None)
        importlib.import_module("palindrome")
    assert buffer.getvalue() == ""


def test_module_does_not_print_on_import_of_builtins():
    """Sanity check that builtins.print is untouched by the module import."""
    assert builtins.print is print
