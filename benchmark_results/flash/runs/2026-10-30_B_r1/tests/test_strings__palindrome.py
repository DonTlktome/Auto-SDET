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
    """test_data should contain at least one palindrome and one non-palindrome."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_matches_test_data(s, expected):
    """is_palindrome must reproduce every expected value in test_data."""
    assert is_palindrome(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),          # empty string boundary
        ("a", True),         # single character boundary
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd length palindrome
        ("abca", False),     # mismatch at the innermost pair
        ("abba", True),      # even length palindrome
        ("abcaa", False),    # mismatch on the first comparison
        ("  ", True),        # whitespace is not stripped
        ("a a", True),       # inner space preserved
        ("a b", False),      # inner space breaks symmetry
        ("Aa", False),       # case sensitive
        ("12321", True),     # digits
        ("12345", False),    # digits, non-palindrome
        ("!@#@!", True),     # punctuation
    ],
)
def test_is_palindrome_edge_cases(s, expected):
    """is_palindrome handles empty, single-char, case and symbol inputs."""
    assert is_palindrome(s) is expected


def test_is_palindrome_returns_bool_type():
    """is_palindrome must return an actual bool, not a truthy value."""
    assert isinstance(is_palindrome("rotor"), bool)
    assert isinstance(is_palindrome("String"), bool)


def test_is_palindrome_long_string():
    """is_palindrome works on a long constructed palindrome."""
    half = "abcdefghij" * 100
    assert is_palindrome(half + half[::-1]) is True
    # Inserting TWO differing characters at the centre breaks symmetry.
    assert is_palindrome(half + "XY" + half[::-1]) is False


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_traversal_matches_test_data(s, expected):
    """is_palindrome_traversal must reproduce every expected value in test_data."""
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
        ("abba", True),
        ("abab", False),
        ("aabbaa", True),
        ("aabcaa", False),
        ("Aa", False),
        ("  ", True),
    ],
)
def test_is_palindrome_traversal_edge_cases(s, expected):
    """is_palindrome_traversal handles empty, odd/even and mixed-case inputs."""
    assert is_palindrome_traversal(s) is expected


def test_is_palindrome_traversal_returns_bool_type():
    """is_palindrome_traversal returns a real bool (all() result)."""
    assert isinstance(is_palindrome_traversal("level"), bool)
    assert isinstance(is_palindrome_traversal("ABC"), bool)


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_recursive_matches_test_data(s, expected):
    """is_palindrome_recursive must reproduce every expected value in test_data."""
    assert is_palindrome_recursive(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),        # base case: len <= 1
        ("z", True),       # base case: single char
        ("zz", True),
        ("zy", False),
        ("zyz", True),
        ("zyyz", True),
        ("zyxz", False),
        ("ab", False),
        ("aba", True),
        ("Aa", False),
        ("1", True),
        ("12", False),
    ],
)
def test_is_palindrome_recursive_edge_cases(s, expected):
    """is_palindrome_recursive handles base cases and mismatches."""
    assert is_palindrome_recursive(s) is expected


def test_is_palindrome_recursive_returns_bool_type():
    """is_palindrome_recursive returns a real bool."""
    assert isinstance(is_palindrome_recursive("rotor"), bool)
    assert isinstance(is_palindrome_recursive("String"), bool)


def test_is_palindrome_recursive_deep_recursion():
    """Recursion depth stays proportional to half the length; moderate input works."""
    half = "ab" * 200
    s = half + half[::-1]  # a genuine 800-char palindrome
    assert is_palindrome_recursive(s) is True
    assert is_palindrome_recursive(s + "c") is False


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s, expected", list(test_data.items()))
def test_is_palindrome_slice_matches_test_data(s, expected):
    """is_palindrome_slice must reproduce every expected value in test_data."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("", True),
        ("a", True),
        ("ab", False),
        ("aa", True),
        ("aba", True),
        ("abc", False),
        ("Aa", False),
        ("  ", True),
        ("a b a", True),
        ("a b c", False),
    ],
)
def test_is_palindrome_slice_edge_cases(s, expected):
    """is_palindrome_slice handles empty, single-char and whitespace inputs."""
    assert is_palindrome_slice(s) is expected


def test_is_palindrome_slice_returns_bool_type():
    """is_palindrome_slice returns a real bool from the equality comparison."""
    assert isinstance(is_palindrome_slice("level"), bool)
    assert isinstance(is_palindrome_slice("ABC"), bool)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    [
        "",
        "a",
        "ab",
        "aba",
        "abba",
        "abcba",
        "abcdba",
        "MALAYALAM",
        "amanaplanacanalpanama",
        "A man a plan a canal panama",
        "12321",
        "123421",
        "!@#@!",
        "x" * 51,
        "x" * 50 + "y",
    ],
)
def test_all_implementations_agree(s):
    """Every palindrome implementation must return the same answer for any input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result(capsys):
    """benchmark_function prints a formatted line containing the function name."""
    with patch("palindrome.timeit", return_value=0.12345) as mock_timeit:
        benchmark_function("is_palindrome_slice")

    mock_timeit.assert_called_once()
    kwargs = mock_timeit.call_args.kwargs
    assert "is_palindrome_slice" in kwargs["stmt"]
    assert "is_palindrome_slice" in kwargs["setup"]
    assert kwargs["number"] == 500000

    out = capsys.readouterr().out
    assert "is_palindrome_slice" in out
    assert "500,000" in out
    assert "0.12345" in out


def test_benchmark_function_returns_none():
    """benchmark_function is a side-effect-only helper and returns None."""
    with patch("palindrome.timeit", return_value=0.0):
        assert benchmark_function("is_palindrome") is None


@pytest.mark.parametrize(
    "name",
    ["is_palindrome", "is_palindrome_traversal", "is_palindrome_recursive", "is_palindrome_slice"],
)
def test_benchmark_function_builds_statement_for_each_impl(name, capsys):
    """benchmark_function embeds the requested implementation name in its stmt/setup."""
    with patch("palindrome.timeit", return_value=1.0) as mock_timeit:
        benchmark_function(name)

    kwargs = mock_timeit.call_args.kwargs
    assert kwargs["stmt"] == f"all({name}(key) == value for key, value in test_data.items())"
    assert kwargs["setup"] == f"from __main__ import test_data, {name}"
    assert name in capsys.readouterr().out


def test_benchmark_function_propagates_timeit_errors():
    """Errors raised by timeit are not swallowed by benchmark_function."""
    with patch("palindrome.timeit", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            benchmark_function("is_palindrome")
