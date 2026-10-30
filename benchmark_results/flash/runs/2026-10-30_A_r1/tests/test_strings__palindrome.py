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


def test_test_data_is_consistent():
    """The module-level test_data must map each string to its true palindrome status."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_expected_entries():
    """test_data should contain the documented sample strings."""
    assert test_data["MALAYALAM"] is True
    assert test_data["String"] is False
    assert test_data["A"] is True
    assert test_data["amanaplanacanalpanama"] is True


# ---------------------------------------------------------------------------
# Shared behaviour across all implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_match_test_data(impl):
    """Every implementation must agree with the expected values in test_data."""
    assert all(impl(key) == value for key, value in test_data.items())


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "value,expected",
    [
        ("", True),          # empty string is a palindrome
        ("a", True),         # single character
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd length palindrome
        ("abba", True),      # even length palindrome
        ("abc", False),      # odd length non-palindrome
        ("abca", False),     # differs only at the ends
        ("a b a", True),     # spaces are treated as ordinary characters
        ("Aa", False),       # case sensitive
        ("12321", True),     # digits
        ("!@#@!", True),     # punctuation
        # long even palindrome: (ab)^n (ba)^n reverses to itself
        ("ab" * 100 + "ba" * 100, True),
        # long string with a mismatch near the middle
        ("a" * 250 + "ba" + "a" * 250, False),
    ],
)
def test_all_implementations_edge_cases(impl, value, expected):
    """All implementations must handle empty, tiny, and long inputs identically."""
    assert impl(value) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_agree_on_random_like_inputs(impl):
    """All implementations must produce identical results on a mixed corpus."""
    corpus = [
        "",
        "x",
        "xy",
        "xyx",
        "xyyx",
        "xyz",
        "racecar",
        "Racecar",
        "noon",
        "moon",
        "a" * 101,
        "a" * 100 + "b",
    ]
    expected = [is_palindrome_slice(s) for s in corpus]
    assert [impl(s) for s in corpus] == expected


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer iterative)
# ---------------------------------------------------------------------------


def test_is_palindrome_happy_path():
    """A classic palindrome returns True."""
    assert is_palindrome("racecar") is True


def test_is_palindrome_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome("python") is False


def test_is_palindrome_empty_string():
    """The empty string is trivially a palindrome."""
    assert is_palindrome("") is True


def test_is_palindrome_single_character():
    """A one-character string is a palindrome."""
    assert is_palindrome("z") is True


def test_is_palindrome_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy object."""
    assert isinstance(is_palindrome("level"), bool)
    assert isinstance(is_palindrome("levels"), bool)


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


def test_is_palindrome_traversal_happy_path():
    """A palindrome is detected by comparing the first half with the reversed half."""
    assert is_palindrome_traversal("amanaplanacanalpanama") is True


def test_is_palindrome_traversal_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome_traversal("abcdba") is False


def test_is_palindrome_traversal_empty_string():
    """Empty string yields True because the loop body never executes."""
    assert is_palindrome_traversal("") is True


def test_is_palindrome_traversal_odd_length_middle_ignored():
    """For odd-length strings the middle character is never compared."""
    assert is_palindrome_traversal("abcba") is True
    assert is_palindrome_traversal("abcda") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


def test_is_palindrome_recursive_happy_path():
    """A palindrome is detected recursively."""
    assert is_palindrome_recursive("rotor") is True


def test_is_palindrome_recursive_non_palindrome():
    """A non-palindrome returns False."""
    assert is_palindrome_recursive("rotors") is False


@pytest.mark.parametrize("value", ["", "a", "ab", "aa"])
def test_is_palindrome_recursive_base_cases(value):
    """Base cases: length <= 1 returns True; length 2 compares both characters."""
    expected = value == value[::-1]
    assert is_palindrome_recursive(value) is expected


def test_is_palindrome_recursive_deep_recursion():
    """A moderately long palindrome must not hit the recursion limit."""
    # (ab)^200 (ba)^200 is a genuine palindrome of length 800, so the
    # recursive implementation recurses ~400 levels deep (well under the limit).
    value = "ab" * 200 + "ba" * 200
    assert is_palindrome_recursive(value) is True


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


def test_is_palindrome_slice_happy_path():
    """Slicing comparison detects a palindrome."""
    assert is_palindrome_slice("level") is True


def test_is_palindrome_slice_non_palindrome():
    """Slicing comparison rejects a non-palindrome."""
    assert is_palindrome_slice("lever") is False


def test_is_palindrome_slice_empty_string():
    """Empty string equals its reverse."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared correctly."""
    assert is_palindrome_slice("été") is True
    assert is_palindrome_slice("étéé") is False


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result():
    """benchmark_function prints a formatted line containing the function name."""
    fake_timeit = MagicMock(return_value=0.12345)
    buffer = io.StringIO()
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(buffer):
        benchmark_function("is_palindrome_slice")

    output = buffer.getvalue()
    assert "is_palindrome_slice" in output
    assert "0.12345" in output
    assert "500,000" in output
    fake_timeit.assert_called_once()


def test_benchmark_function_builds_expected_timeit_arguments():
    """benchmark_function must pass the correct stmt/setup/number to timeit."""
    fake_timeit = MagicMock(return_value=0.0)
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(io.StringIO()):
        benchmark_function("is_palindrome")

    _, kwargs = fake_timeit.call_args
    assert kwargs["stmt"] == (
        "all(is_palindrome(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"
    assert kwargs["number"] == 500000


def test_benchmark_function_returns_none():
    """benchmark_function is a side-effect-only helper and returns None."""
    with patch("palindrome.timeit", MagicMock(return_value=0.0)), redirect_stdout(
        io.StringIO()
    ):
        assert benchmark_function("is_palindrome_recursive") is None


def test_benchmark_function_name_padding():
    """The function name is left-aligned in a 35-character field."""
    fake_timeit = MagicMock(return_value=1.0)
    buffer = io.StringIO()
    with patch("palindrome.timeit", fake_timeit), redirect_stdout(buffer):
        benchmark_function("x")

    line = buffer.getvalue().strip()
    assert line.startswith("x" + " " * 34)
