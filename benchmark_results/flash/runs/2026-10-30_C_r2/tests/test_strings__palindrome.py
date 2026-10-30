"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import builtins
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by file path (the module name "palindrome" is
# generic and may collide with other modules on sys.path).
# ---------------------------------------------------------------------------
MODULE_PATH = Path(__file__).resolve().parent.parent / "strings" / "palindrome.py"
if not MODULE_PATH.exists():  # pragma: no cover - fallback for flat layouts
    MODULE_PATH = Path(__file__).resolve().parent / "palindrome.py"

_spec = importlib.util.spec_from_file_location("palindrome_under_test", MODULE_PATH)
palindrome = importlib.util.module_from_spec(_spec)
sys.modules["palindrome_under_test"] = palindrome
_spec.loader.exec_module(palindrome)


ALL_IMPLS = [
    palindrome.is_palindrome,
    palindrome.is_palindrome_traversal,
    palindrome.is_palindrome_recursive,
    palindrome.is_palindrome_slice,
]

IMPL_IDS = [
    "is_palindrome",
    "is_palindrome_traversal",
    "is_palindrome_recursive",
    "is_palindrome_slice",
]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------
def test_test_data_is_self_consistent():
    """The module-level test_data must agree with Python's own reversal check."""
    assert palindrome.test_data
    for key, value in palindrome.test_data.items():
        assert (key == key[::-1]) is value


def test_test_data_contains_expected_entries():
    """Spot-check a few known entries of the module-level test_data mapping."""
    assert palindrome.test_data["MALAYALAM"] is True
    assert palindrome.test_data["String"] is False
    assert palindrome.test_data["A"] is True
    assert palindrome.test_data["AB"] is False


# ---------------------------------------------------------------------------
# Shared parametrized tests across all four implementations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("MALAYALAM", True),
        ("String", False),
        ("rotor", True),
        ("level", True),
        ("A", True),
        ("BB", True),
        ("ABC", False),
        ("amanaplanacanalpanama", True),
        ("abcdba", False),
        ("AB", False),
    ],
)
def test_all_implementations_match_test_data(impl, text, expected):
    """Every implementation must reproduce the expected result for test_data."""
    assert impl(text) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),          # empty string is trivially a palindrome
        ("a", True),         # single character
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd length palindrome
        ("abba", True),      # even length palindrome
        ("abca", False),     # mismatch in the middle
        ("a" * 1000, True),  # long palindrome
        ("a" * 999 + "b", False),  # long non-palindrome
        (" ", True),         # whitespace only
        ("  ", True),        # two spaces
        ("a b", False),      # spaces are significant
        ("Aa", False),       # case sensitive
        ("12321", True),     # digits
        ("12345", False),    # digits, non-palindrome
        ("!@#$%$#@!", True), # punctuation
    ],
)
def test_all_implementations_edge_cases(impl, text, expected):
    """Edge cases: empty, single char, long strings, digits, punctuation."""
    assert impl(text) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_return_bool(impl):
    """Each implementation must return an actual bool, not a truthy value."""
    assert isinstance(impl("level"), bool)
    assert isinstance(impl("levelx"), bool)


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_agree(impl):
    """All implementations must agree with the slice-based reference."""
    for key in palindrome.test_data:
        assert impl(key) is palindrome.is_palindrome_slice(key)


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------
def test_is_palindrome_happy_path():
    """A classic palindrome returns True."""
    assert palindrome.is_palindrome("racecar") is True


def test_is_palindrome_non_palindrome():
    """A string that differs at the outer positions returns False."""
    assert palindrome.is_palindrome("abca") is False


def test_is_palindrome_empty_string():
    """The empty string has no characters to compare and is a palindrome."""
    assert palindrome.is_palindrome("") is True


def test_is_palindrome_single_character():
    """A one-character string is always a palindrome."""
    assert palindrome.is_palindrome("x") is True


def test_is_palindrome_mismatch_at_innermost_pair():
    """Mismatch only at the middle pair must still be detected."""
    assert palindrome.is_palindrome("abxba") is True
    assert palindrome.is_palindrome("abxca") is False


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------
def test_is_palindrome_traversal_happy_path():
    """A classic palindrome returns True."""
    assert palindrome.is_palindrome_traversal("racecar") is True


def test_is_palindrome_traversal_non_palindrome():
    """A non-palindrome returns False."""
    assert palindrome.is_palindrome_traversal("abcdef") is False


def test_is_palindrome_traversal_empty_string():
    """Empty string: the range is empty so all() is vacuously True."""
    assert palindrome.is_palindrome_traversal("") is True


def test_is_palindrome_traversal_single_character():
    """Single character: half-length is 0, so all() is vacuously True."""
    assert palindrome.is_palindrome_traversal("z") is True


def test_is_palindrome_traversal_two_characters():
    """Two characters exercise the boundary of the half traversal."""
    assert palindrome.is_palindrome_traversal("aa") is True
    assert palindrome.is_palindrome_traversal("ab") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------
def test_is_palindrome_recursive_happy_path():
    """A classic palindrome returns True."""
    assert palindrome.is_palindrome_recursive("racecar") is True


def test_is_palindrome_recursive_non_palindrome():
    """A non-palindrome returns False."""
    assert palindrome.is_palindrome_recursive("abcdef") is False


def test_is_palindrome_recursive_empty_string():
    """Base case: len(s) <= 1 returns True for the empty string."""
    assert palindrome.is_palindrome_recursive("") is True


def test_is_palindrome_recursive_single_character():
    """Base case: len(s) <= 1 returns True for a single character."""
    assert palindrome.is_palindrome_recursive("q") is True


def test_is_palindrome_recursive_two_characters():
    """Two characters: recursion bottoms out at the empty string."""
    assert palindrome.is_palindrome_recursive("aa") is True
    assert palindrome.is_palindrome_recursive("ab") is False


def test_is_palindrome_recursive_calls_itself():
    """The recursive implementation must actually recurse on the inner slice."""
    with patch.object(
        palindrome, "is_palindrome_recursive", wraps=palindrome.is_palindrome_recursive
    ) as spy:
        assert palindrome.is_palindrome_recursive("abcba") is True
    # "abcba" -> "bcb" -> "c" -> base case
    assert spy.call_count >= 3


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------
def test_is_palindrome_slice_happy_path():
    """A classic palindrome returns True."""
    assert palindrome.is_palindrome_slice("racecar") is True


def test_is_palindrome_slice_non_palindrome():
    """A non-palindrome returns False."""
    assert palindrome.is_palindrome_slice("abcdef") is False


def test_is_palindrome_slice_empty_string():
    """The empty string equals its own reverse."""
    assert palindrome.is_palindrome_slice("") is True


def test_is_palindrome_slice_single_character():
    """A single character equals its own reverse."""
    assert palindrome.is_palindrome_slice("k") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared by code point, not by bytes."""
    assert palindrome.is_palindrome_slice("été") is True
    assert palindrome.is_palindrome_slice("éte") is False


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------
def test_benchmark_function_prints_result(capsys):
    """benchmark_function must print a single formatted line with the run count."""
    fake_timeit = MagicMock(return_value=0.12345)
    with patch.object(palindrome, "timeit", fake_timeit):
        result = palindrome.benchmark_function("is_palindrome_slice")

    assert result is None
    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["number"] == 500000
    assert "is_palindrome_slice" in kwargs["stmt"]
    assert "is_palindrome_slice" in kwargs["setup"]

    out = capsys.readouterr().out
    assert "is_palindrome_slice" in out
    assert "500,000" in out
    assert "0.12345" in out


def test_benchmark_function_uses_given_name(capsys):
    """The function name must be interpolated into both stmt and setup."""
    fake_timeit = MagicMock(return_value=1.0)
    with patch.object(palindrome, "timeit", fake_timeit):
        palindrome.benchmark_function("is_palindrome_recursive")

    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["stmt"] == (
        "all(is_palindrome_recursive(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome_recursive"
    assert "is_palindrome_recursive" in capsys.readouterr().out


def test_benchmark_function_propagates_timeit_errors():
    """Errors raised by timeit must not be swallowed."""
    with patch.object(palindrome, "timeit", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            palindrome.benchmark_function("is_palindrome")


def test_benchmark_function_output_format(capsys):
    """The printed line must be left-aligned in a 35-character name column."""
    with patch.object(palindrome, "timeit", MagicMock(return_value=0.5)):
        palindrome.benchmark_function("fn")

    out = capsys.readouterr().out
    assert out.startswith("fn" + " " * 33)
    assert out.endswith("seconds\n")