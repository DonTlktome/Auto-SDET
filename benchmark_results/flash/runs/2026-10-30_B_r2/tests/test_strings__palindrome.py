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
    """test_data should include at least one palindrome and one non-palindrome."""
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
    """Comparison is exact, so differing case breaks the palindrome property."""
    assert is_palindrome("Aa") is False
    assert is_palindrome("aA") is False


def test_is_palindrome_does_not_ignore_whitespace():
    """Whitespace is treated as a regular character."""
    assert is_palindrome("a b a") is True
    assert is_palindrome("ab a") is False


def test_is_palindrome_long_string():
    """A long palindrome is handled without recursion limits."""
    s = "x" * 5000 + "y" + "x" * 5000
    assert is_palindrome(s) is True
    assert is_palindrome(s + "z") is False


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
    """Mismatched mirrored characters are detected."""
    assert is_palindrome_traversal(s) is False


def test_is_palindrome_traversal_empty_string_boundary():
    """len(s) // 2 == 0 for the empty string, so all() over an empty range is True."""
    assert is_palindrome_traversal("") is True


def test_is_palindrome_traversal_odd_length_middle_ignored():
    """The middle character of an odd-length string is never compared."""
    assert is_palindrome_traversal("abcba") is True
    assert is_palindrome_traversal("abcxa") is False


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


def test_is_palindrome_recursive_empty_and_single_boundary():
    """Empty and single-character strings hit the len(s) <= 1 base case."""
    assert is_palindrome_recursive("") is True
    assert is_palindrome_recursive("q") is True


def test_is_palindrome_recursive_uses_recursion():
    """The recursive implementation should call itself for multi-char input."""
    with patch(
        "palindrome.is_palindrome_recursive",
        wraps=is_palindrome_recursive,
    ) as spy:
        assert spy("abba") is True
        assert spy.call_count >= 2


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_slice_matches_test_data(s, expected):
    """is_palindrome_slice returns the expected verdict for sample strings."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize("s", ["", "a", "aa", "aba", "abba", "abcba"])
def test_is_palindrome_slice_positive_cases(s):
    """Slicing-based check accepts palindromes."""
    assert is_palindrome_slice(s) is True


@pytest.mark.parametrize("s", ["ab", "abc", "abca", "aab"])
def test_is_palindrome_slice_negative_cases(s):
    """Slicing-based check rejects non-palindromes."""
    assert is_palindrome_slice(s) is False


def test_is_palindrome_slice_empty_string_boundary():
    """The empty string equals its own reverse."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared by code point."""
    assert is_palindrome_slice("été") is True
    assert is_palindrome_slice("éte") is False


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", list(test_data.keys()) + ["", "a", "ab", "aba", "abcba"])
def test_all_implementations_agree(s):
    """Every implementation must produce identical results for the same input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------


def test_benchmark_function_prints_result(capsys):
    """benchmark_function prints the function name and run count."""
    fake_timeit = MagicMock(return_value=0.12345)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome_slice")

    out = capsys.readouterr().out
    assert "is_palindrome_slice" in out
    assert "500,000" in out
    assert "0.12345" in out


def test_benchmark_function_builds_expected_timeit_arguments():
    """benchmark_function passes the correct stmt/setup/number to timeit."""
    fake_timeit = MagicMock(return_value=0.0)
    with patch("palindrome.timeit", fake_timeit):
        benchmark_function("is_palindrome")

    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
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


def test_benchmark_function_uses_real_timeit(capsys, monkeypatch):
    """With the real timeit, output is still formatted as expected."""
    # benchmark_function's setup string is `from __main__ import ...`, which is
    # only resolvable when palindrome.py runs as a script. Under pytest the
    # `__main__` module is pytest's own entry point, so expose the names there.
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(main_module, "test_data", test_data, raising=False)
    monkeypatch.setattr(
        main_module, "is_palindrome_slice", is_palindrome_slice, raising=False
    )

    benchmark_function("is_palindrome_slice")

    out = capsys.readouterr().out
    assert out.startswith("is_palindrome_slice")
    assert "finished" in out
    assert out.rstrip().endswith("seconds")
