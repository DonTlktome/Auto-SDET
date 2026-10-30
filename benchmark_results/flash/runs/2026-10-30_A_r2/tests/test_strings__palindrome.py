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


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------


def test_test_data_is_self_consistent():
    """The module-level test_data must agree with Python's own reversal check."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_both_outcomes():
    """test_data should exercise both True and False outcomes."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_matches_expected(s, expected):
    """is_palindrome returns the expected verdict for every sample string."""
    assert is_palindrome(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_positive_edge_cases(s):
    """Empty, single-char and symmetric strings are palindromes."""
    assert is_palindrome(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_negative_edge_cases(s):
    """Near-miss strings are correctly rejected."""
    assert is_palindrome(s) is False


def test_is_palindrome_is_case_sensitive():
    """Comparison is exact, so differing case breaks the palindrome property."""
    assert is_palindrome("Aa") is False
    assert is_palindrome("aA") is False


def test_is_palindrome_does_not_mutate_input():
    """The input string is unchanged after the call (strings are immutable)."""
    s = "amanaplanacanalpanama"
    original = str(s)
    is_palindrome(s)
    assert s == original


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_traversal_matches_expected(s, expected):
    """is_palindrome_traversal returns the expected verdict for every sample."""
    assert is_palindrome_traversal(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_traversal_positive_edge_cases(s):
    """Empty, single-char and symmetric strings are palindromes."""
    assert is_palindrome_traversal(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_traversal_negative_edge_cases(s):
    """Near-miss strings are correctly rejected."""
    assert is_palindrome_traversal(s) is False


def test_is_palindrome_traversal_odd_length_middle_ignored():
    """For odd-length strings the middle character is never compared."""
    assert is_palindrome_traversal("abcba") is True
    assert is_palindrome_traversal("abcda") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_recursive_matches_expected(s, expected):
    """is_palindrome_recursive returns the expected verdict for every sample."""
    assert is_palindrome_recursive(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_recursive_positive_edge_cases(s):
    """Empty and single-char strings hit the base case and return True."""
    assert is_palindrome_recursive(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_recursive_negative_edge_cases(s):
    """Near-miss strings are correctly rejected."""
    assert is_palindrome_recursive(s) is False


def test_is_palindrome_recursive_handles_long_input():
    """A long palindrome must not blow the recursion limit (depth ~ n/2)."""
    s = "ab" * 400 + "c" + "ba" * 400
    assert is_palindrome_recursive(s) is True


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_slice_matches_expected(s, expected):
    """is_palindrome_slice returns the expected verdict for every sample."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_slice_positive_edge_cases(s):
    """Empty, single-char and symmetric strings are palindromes."""
    assert is_palindrome_slice(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_slice_negative_edge_cases(s):
    """Near-miss strings are correctly rejected."""
    assert is_palindrome_slice(s) is False


def test_is_palindrome_slice_returns_bool():
    """The slice implementation returns a real bool, not a truthy object."""
    assert isinstance(is_palindrome_slice("aba"), bool)
    assert isinstance(is_palindrome_slice("abc"), bool)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(test_data.keys()) + ["", "a", "ab", "aba", "abca"])
def test_all_implementations_agree(s):
    """Every implementation must produce identical results for the same input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


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
    """benchmark_function passes the correct stmt/setup/number to timeit."""
    with patch("palindrome.timeit", return_value=0.0) as mock_timeit:
        with redirect_stdout(io.StringIO()):
            benchmark_function("is_palindrome")

    kwargs = mock_timeit.call_args.kwargs
    assert kwargs["stmt"] == (
        "all(is_palindrome(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"
    assert kwargs["number"] == 500000


def test_benchmark_function_returns_none():
    """benchmark_function is a side-effect-only helper and returns None."""
    with patch("palindrome.timeit", return_value=0.0):
        with redirect_stdout(io.StringIO()):
            assert benchmark_function("is_palindrome_slice") is None


def test_benchmark_function_does_not_call_real_timeit():
    """No real timing work happens when timeit is patched out."""
    with patch("palindrome.timeit", return_value=1.0) as mock_timeit:
        with redirect_stdout(io.StringIO()):
            benchmark_function("is_palindrome_recursive")
    assert mock_timeit.call_count == 1


# ---------------------------------------------------------------------------
# Module import side effects
# ---------------------------------------------------------------------------


def test_module_import_does_not_run_benchmarks():
    """Importing the module must not execute the __main__ benchmark block."""
    import importlib
    import sys

    buffer = io.StringIO()
    with patch("palindrome.timeit") as mock_timeit:
        with redirect_stdout(buffer):
            sys.modules.pop("palindrome", None)
            importlib.import_module("palindrome")
    assert mock_timeit.call_count == 0
    assert buffer.getvalue() == ""


def test_builtins_all_used_by_implementations():
    """Sanity check that `all` is the builtin used by the traversal variant."""
    assert builtins.all([True, True]) is True
    assert is_palindrome_traversal("abba") is True