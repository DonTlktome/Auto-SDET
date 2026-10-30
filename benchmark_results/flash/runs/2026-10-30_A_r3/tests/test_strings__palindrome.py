"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import builtins
import sys
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

ALL_IMPLEMENTATIONS = [
    is_palindrome,
    is_palindrome_traversal,
    is_palindrome_recursive,
    is_palindrome_slice,
]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------


def test_test_data_is_consistent_with_python_slicing():
    """The module-level test_data must agree with the trivial slice check."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_both_true_and_false_cases():
    """test_data must exercise both positive and negative outcomes."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_matches_expected(s, expected):
    """Two-pointer implementation returns the expected verdict for test_data."""
    assert is_palindrome(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),          # empty string is a palindrome
        ("a", True),         # single character
        ("aa", True),        # even length, minimal
        ("ab", False),       # even length, minimal mismatch
        ("aba", True),       # odd length
        ("abc", False),      # odd length mismatch
        ("abca", False),     # mismatch at the outer pair
        ("abccba", True),    # even length palindrome
        ("abccbx", False),   # mismatch at the innermost pair
        ("  ", True),        # whitespace is not stripped
        ("a a", True),       # spaces are compared literally
        ("a b", False),
        ("12321", True),     # digits
        ("12345", False),
        ("!@#$%$#@!", True), # punctuation
    ],
)
def test_is_palindrome_edge_cases(s, expected):
    """Boundary and edge inputs for the two-pointer implementation."""
    assert is_palindrome(s) is expected


def test_is_palindrome_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy object."""
    assert isinstance(is_palindrome("rotor"), bool)
    assert isinstance(is_palindrome("String"), bool)


def test_is_palindrome_does_not_mutate_input():
    """The input string must be unchanged after the call."""
    s = "amanaplanacanalpanama"
    original = str(s)
    is_palindrome(s)
    assert s == original


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_traversal_matches_expected(s, expected):
    """Half-traversal implementation returns the expected verdict."""
    assert is_palindrome_traversal(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("ab", False),
        ("aba", True),
        ("abc", False),
        ("abccba", True),
        ("abccbx", False),
        ("12321", True),
        ("12345", False),
    ],
)
def test_is_palindrome_traversal_edge_cases(s, expected):
    """Boundary inputs for the half-traversal implementation."""
    assert is_palindrome_traversal(s) is expected


def test_is_palindrome_traversal_returns_bool_type():
    """The traversal implementation must return a bool instance."""
    assert isinstance(is_palindrome_traversal("level"), bool)
    assert isinstance(is_palindrome_traversal("ABC"), bool)


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_recursive_matches_expected(s, expected):
    """Recursive implementation returns the expected verdict."""
    assert is_palindrome_recursive(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),        # base case: len <= 1
        ("a", True),       # base case: len == 1
        ("aa", True),
        ("ab", False),
        ("aba", True),
        ("abc", False),
        ("abccba", True),
        ("abccbx", False),
        ("x" * 200, True), # deep-ish recursion still terminates
    ],
)
def test_is_palindrome_recursive_edge_cases(s, expected):
    """Base cases and boundary inputs for the recursive implementation."""
    assert is_palindrome_recursive(s) is expected


def test_is_palindrome_recursive_returns_bool_type():
    """The recursive implementation must return a bool instance."""
    assert isinstance(is_palindrome_recursive("rotor"), bool)
    assert isinstance(is_palindrome_recursive("String"), bool)


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_slice_matches_expected(s, expected):
    """Slice-reversal implementation returns the expected verdict."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("ab", False),
        ("aba", True),
        ("abc", False),
        ("abccba", True),
        ("abccbx", False),
        ("12321", True),
        ("12345", False),
    ],
)
def test_is_palindrome_slice_edge_cases(s, expected):
    """Boundary inputs for the slice-reversal implementation."""
    assert is_palindrome_slice(s) is expected


def test_is_palindrome_slice_returns_bool_type():
    """The slice implementation must return a bool instance."""
    assert isinstance(is_palindrome_slice("level"), bool)
    assert isinstance(is_palindrome_slice("ABC"), bool)


# ---------------------------------------------------------------------------
# Cross-implementation agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    [
        "",
        "a",
        "ab",
        "aba",
        "abc",
        "abccba",
        "abccbx",
        "MALAYALAM",
        "amanaplanacanalpanama",
        "abcdba",
        "A man a plan a canal panama",
        "12321",
        "!@#$%$#@!",
    ],
)
def test_all_implementations_agree(s):
    """Every implementation must produce identical results for the same input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLEMENTATIONS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_expected_format(capsys):
    """benchmark_function prints the function name and run count."""
    fake_timeit = MagicMock(return_value=0.12345)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome_slice")

    captured = capsys.readouterr()
    assert "is_palindrome_slice" in captured.out
    assert "500,000" in captured.out
    assert "0.12345" in captured.out
    assert "seconds" in captured.out


def test_benchmark_function_calls_timeit_with_expected_arguments():
    """benchmark_function must call timeit with the right stmt/setup/number."""
    fake_timeit = MagicMock(return_value=0.0)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome")

    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["number"] == 500000
    assert "is_palindrome" in kwargs["stmt"]
    assert "test_data" in kwargs["stmt"]
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"


def test_benchmark_function_returns_none(capsys):
    """benchmark_function is a reporting helper and returns None."""
    with patch("palindrome.timeit", MagicMock(return_value=0.5)):
        assert benchmark_function("is_palindrome_recursive") is None
    capsys.readouterr()


@pytest.mark.parametrize("name", ["is_palindrome", "is_palindrome_traversal", "custom_name"])
def test_benchmark_function_embeds_name_in_output(name, capsys):
    """The provided function name must appear in the printed report."""
    with patch("palindrome.timeit", MagicMock(return_value=1.0)):
        benchmark_function(name)
    assert name in capsys.readouterr().out


def test_benchmark_function_does_not_execute_real_timing():
    """Guard: the real timeit must never run during tests (no long benchmarks)."""
    with patch("palindrome.timeit", MagicMock(return_value=0.0)) as fake_timeit:
        benchmark_function("is_palindrome_slice")
    assert fake_timeit.call_count == 1