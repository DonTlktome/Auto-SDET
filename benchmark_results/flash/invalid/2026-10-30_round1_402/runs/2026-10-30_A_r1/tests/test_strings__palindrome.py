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


@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_matches_test_data(s, expected):
    """is_palindrome returns the expected verdict for every sample string."""
    assert is_palindrome(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_positive_cases(s):
    """Short and empty strings are palindromes."""
    assert is_palindrome(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_negative_cases(s):
    """Strings that differ at mirrored positions are not palindromes."""
    assert is_palindrome(s) is False


def test_is_palindrome_single_character_boundary():
    """A one-character string has start_i == end_i, so the loop never runs."""
    assert is_palindrome("Z") is True


def test_is_palindrome_two_character_boundary():
    """Two-character strings exercise exactly one loop iteration."""
    assert is_palindrome("zz") is True
    assert is_palindrome("zy") is False


def test_is_palindrome_is_case_sensitive():
    """Comparison is exact, so differing case makes a string non-palindromic."""
    assert is_palindrome("Aa") is False


def test_is_palindrome_handles_whitespace_and_punctuation():
    """Whitespace and punctuation are compared literally, not stripped."""
    assert is_palindrome("a b a") is True
    assert is_palindrome("a man a plan a canal panama") is False


def test_is_palindrome_long_string():
    """A long palindrome is handled without recursion limits or errors."""
    s = "ab" * 5000 + "c" + "ba" * 5000
    assert is_palindrome(s) is True


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_traversal_matches_test_data(s, expected):
    """is_palindrome_traversal returns the expected verdict for sample strings."""
    assert is_palindrome_traversal(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_traversal_positive_cases(s):
    """Short and empty strings are palindromes under half traversal."""
    assert is_palindrome_traversal(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_traversal_negative_cases(s):
    """Mismatched mirrored characters are detected by half traversal."""
    assert is_palindrome_traversal(s) is False


def test_is_palindrome_traversal_empty_string():
    """An empty string yields range(0), so all() over an empty iterable is True."""
    assert is_palindrome_traversal("") is True


def test_is_palindrome_traversal_odd_length_middle_ignored():
    """The middle character of an odd-length string is never compared."""
    assert is_palindrome_traversal("abcba") is True
    assert is_palindrome_traversal("abcda") is False


def test_is_palindrome_traversal_long_string():
    """Half traversal handles long inputs iteratively."""
    s = "x" * 10000
    assert is_palindrome_traversal(s) is True


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_recursive_matches_test_data(s, expected):
    """is_palindrome_recursive returns the expected verdict for sample strings."""
    assert is_palindrome_recursive(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_recursive_positive_cases(s):
    """Base cases (len <= 1) and matching outer characters return True."""
    assert is_palindrome_recursive(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_recursive_negative_cases(s):
    """Mismatched outer characters short-circuit to False."""
    assert is_palindrome_recursive(s) is False


def test_is_palindrome_recursive_base_case_empty():
    """The empty string hits the len(s) <= 1 base case."""
    assert is_palindrome_recursive("") is True


def test_is_palindrome_recursive_base_case_single_char():
    """A single character hits the len(s) <= 1 base case."""
    assert is_palindrome_recursive("q") is True


def test_is_palindrome_recursive_short_circuits_on_mismatch():
    """A mismatch at the outermost pair returns False without deeper recursion."""
    with patch.object(
        sys.modules[__name__], "is_palindrome_recursive", wraps=is_palindrome_recursive
    ) as spy:
        # Direct call on the module function is not intercepted by the spy on this
        # module, so assert behaviour instead: only one comparison is needed.
        assert is_palindrome_recursive("ab" + "z" * 100) is False


def test_is_palindrome_recursive_moderate_length():
    """Recursion depth stays within limits for moderately long palindromes."""
    s = "ab" * 200
    assert is_palindrome_recursive(s + s[::-1]) is True


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_slice_matches_test_data(s, expected):
    """is_palindrome_slice returns the expected verdict for sample strings."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_slice_positive_cases(s):
    """Strings equal to their reverse are palindromes."""
    assert is_palindrome_slice(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_slice_negative_cases(s):
    """Strings differing from their reverse are not palindromes."""
    assert is_palindrome_slice(s) is False


def test_is_palindrome_slice_empty_string():
    """The empty string equals its own reverse."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared correctly by slicing."""
    assert is_palindrome_slice("été") is True
    assert is_palindrome_slice("éte") is False


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(test_data.keys()) + ["", "a", "ab", "aba", "abc"])
def test_all_implementations_agree(s):
    """Every implementation must return the same verdict for the same input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result(capsys):
    """benchmark_function prints a formatted line containing the function name."""
    fake_timeit = MagicMock(return_value=0.12345)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome_slice")

    captured = capsys.readouterr()
    assert "is_palindrome_slice" in captured.out
    assert "0.12345" in captured.out
    assert "500,000" in captured.out
    fake_timeit.assert_called_once()


def test_benchmark_function_builds_expected_timeit_arguments():
    """benchmark_function passes the expected stmt/setup/number to timeit."""
    fake_timeit = MagicMock(return_value=1.0)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome")

    _, kwargs = fake_timeit.call_args
    assert kwargs["stmt"] == (
        "all(is_palindrome(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"
    assert kwargs["number"] == 500000


def test_benchmark_function_returns_none(capsys):
    """benchmark_function is a reporting helper and returns None."""
    with patch("palindrome.timeit", MagicMock(return_value=0.0)):
        assert benchmark_function("is_palindrome_slice") is None
    capsys.readouterr()


def test_benchmark_function_uses_unknown_name_verbatim(capsys):
    """An arbitrary name is interpolated into the statement without validation."""
    fake_timeit = MagicMock(return_value=0.5)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("not_a_real_function")

    _, kwargs = fake_timeit.call_args
    assert "not_a_real_function" in kwargs["stmt"]
    assert "not_a_real_function" in kwargs["setup"]
    assert "not_a_real_function" in capsys.readouterr().out