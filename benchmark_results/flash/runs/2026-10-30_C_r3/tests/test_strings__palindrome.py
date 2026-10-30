"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import builtins
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

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

IMPL_IDS = ["iterative", "traversal", "recursive", "slice"]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------


def test_test_data_is_consistent_with_slicing():
    """The module-level test_data must agree with the naive slice reversal."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_both_outcomes():
    """test_data must contain at least one palindrome and one non-palindrome."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# Shared parametrized coverage for all four implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("text,expected", list(test_data.items()))
def test_all_impls_match_test_data(impl, text, expected):
    """Every implementation must reproduce the expected result for test_data."""
    assert impl(text) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),          # empty string boundary
        ("a", True),         # single character boundary
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd-length palindrome
        ("abba", True),      # even-length palindrome
        ("abca", False),     # mismatch at the innermost pair
        ("abcba", True),     # odd-length palindrome, longer
        ("abccbx", False),   # mismatch near the end
        ("  ", True),        # whitespace only
        ("a a", True),       # whitespace inside
        ("a b", False),      # whitespace breaks symmetry
        ("Aa", False),       # case sensitivity
        ("12321", True),     # digits
        ("12345", False),    # digits, non-palindrome
        ("!@#$%$#@!", True), # punctuation
    ],
)
def test_all_impls_edge_cases(impl, text, expected):
    """All implementations must agree on edge cases and boundaries."""
    assert impl(text) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_impls_return_bool_type(impl):
    """Each implementation must return an actual bool, not a truthy value."""
    assert isinstance(impl("level"), bool)
    assert isinstance(impl("lever"), bool)


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_impls_agree_on_long_string(impl):
    """All implementations must agree on a long palindrome and its near-miss."""
    long_pal = "abcdefghijklmnopqrstuvwxyz" + "zyxwvutsrqponmlkjihgfedcba"
    assert impl(long_pal) is True
    assert impl(long_pal + "x") is False


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer iterative)
# ---------------------------------------------------------------------------


def test_is_palindrome_happy_path():
    """A classic palindrome returns True."""
    assert is_palindrome("amanaplanacanalpanama") is True


def test_is_palindrome_returns_false_early_on_mismatch():
    """A mismatch at the first pair short-circuits to False."""
    assert is_palindrome("ab" + "a" * 1000) is False


def test_is_palindrome_empty_string():
    """The empty string is trivially a palindrome."""
    assert is_palindrome("") is True


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


def test_is_palindrome_traversal_happy_path():
    """A classic palindrome returns True via half traversal."""
    assert is_palindrome_traversal("rotor") is True


def test_is_palindrome_traversal_single_char():
    """A one-character string has an empty traversal range and is a palindrome."""
    assert is_palindrome_traversal("Z") is True


def test_is_palindrome_traversal_detects_inner_mismatch():
    """A mismatch inside the traversed first half is detected."""
    assert is_palindrome_traversal("abxca") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


def test_is_palindrome_recursive_happy_path():
    """A classic palindrome returns True recursively."""
    assert is_palindrome_recursive("level") is True


def test_is_palindrome_recursive_base_cases():
    """Length 0 and 1 are base cases returning True."""
    assert is_palindrome_recursive("") is True
    assert is_palindrome_recursive("q") is True


def test_is_palindrome_recursive_false_case():
    """A non-palindrome returns False."""
    assert is_palindrome_recursive("abcdba") is False


def test_is_palindrome_recursive_recurses_on_inner_slice():
    """The recursive implementation must call itself with the inner slice."""
    with patch(
        "palindrome.is_palindrome_recursive",
        side_effect=is_palindrome_recursive,
        wraps=is_palindrome_recursive,
    ) as spy:
        assert is_palindrome_recursive("abba") is True
    assert spy.call_count >= 1


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


def test_is_palindrome_slice_happy_path():
    """A classic palindrome returns True via slicing."""
    assert is_palindrome_slice("MALAYALAM") is True


def test_is_palindrome_slice_empty_string():
    """The empty string equals its own reversal."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome_slice("String") is False


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result():
    """benchmark_function prints a formatted line containing the function name."""
    fake_timeit = MagicMock(return_value=0.12345)
    buffer = io.StringIO()
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(buffer):
        result = benchmark_function("is_palindrome_slice")

    assert result is None
    output = buffer.getvalue()
    assert "is_palindrome_slice" in output
    assert "0.12345" in output
    assert "500,000" in output


def test_benchmark_function_builds_expected_timeit_arguments():
    """benchmark_function must call timeit with the documented stmt/setup/number."""
    fake_timeit = MagicMock(return_value=0.0)
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(io.StringIO()):
        benchmark_function("is_palindrome")

    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["number"] == 500000
    assert "is_palindrome" in kwargs["stmt"]
    assert "test_data" in kwargs["stmt"]
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"


def test_benchmark_function_does_not_execute_real_timing():
    """benchmark_function must not perform real timing work (fully mocked)."""
    fake_timeit = MagicMock(return_value=1.0)
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(io.StringIO()):
        benchmark_function("is_palindrome_recursive")
    assert fake_timeit.call_count == 1


# ---------------------------------------------------------------------------
# Module import side effects
# ---------------------------------------------------------------------------


def test_module_import_does_not_run_benchmarks():
    """Importing the module must not execute the __main__ benchmark block."""
    assert "palindrome" in sys.modules
    # The module-level assert on test_data must have passed at import time.
    assert all((k == k[::-1]) == v for k, v in test_data.items())
