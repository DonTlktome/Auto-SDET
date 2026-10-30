"""Tests for the pangram detection functions in is_pangram.py."""

import builtins
from unittest.mock import patch

import pytest

from is_pangram import (
    benchmark,
    is_pangram,
    is_pangram_fastest,
    is_pangram_faster,
)

# All three implementations should agree on these inputs.
PANGRAM_STRINGS = [
    "The quick brown fox jumps over the lazy dog",
    "Waltz, bad nymph, for quick jigs vex.",
    "Jived fox nymph grabs quick waltz.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
]

NON_PANGRAM_STRINGS = [
    "My name is Unknown",
    "The quick brown fox jumps over the la_y dog",
    "",
    "a",
    "abcdefghijklmnopqrstuvwxy",  # missing 'z'
    "1234567890 !@#$%^&*()",
    "The quick brown fox jumps over the lazy do",  # missing 'g'
]

ALL_IMPLEMENTATIONS = [is_pangram, is_pangram_faster, is_pangram_fastest]


# ---------------------------------------------------------------------------
# is_pangram
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", PANGRAM_STRINGS)
def test_is_pangram_positive(text):
    """is_pangram returns True for strings containing every letter."""
    assert is_pangram(text) is True


@pytest.mark.parametrize("text", NON_PANGRAM_STRINGS)
def test_is_pangram_negative(text):
    """is_pangram returns False for strings missing at least one letter."""
    assert is_pangram(text) is False


def test_is_pangram_default_argument():
    """Calling is_pangram with no arguments uses the default pangram."""
    assert is_pangram() is True


def test_is_pangram_ignores_non_letters():
    """Digits, punctuation and symbols do not affect the result."""
    assert is_pangram("The quick brown fox jumps over the lazy dog 12345!!!") is True


def test_is_pangram_case_insensitive():
    """Upper and lower case letters are treated as the same letter."""
    assert is_pangram("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG") is True
    assert is_pangram("tHe QuIcK bRoWn FoX jUmPs OvEr ThE lAzY dOg") is True


def test_is_pangram_exactly_26_letters():
    """A string with exactly the 26 letters in mixed order is a pangram."""
    assert is_pangram("zyxwvutsrqponmlkjihgfedcba") is True


def test_is_pangram_missing_single_letter():
    """Removing one letter from a pangram makes it a non-pangram."""
    assert is_pangram("The quick brown fox jumps over the lazy do") is False


def test_is_pangram_empty_string():
    """An empty string is not a pangram."""
    assert is_pangram("") is False


def test_is_pangram_whitespace_only():
    """A whitespace-only string is not a pangram."""
    assert is_pangram("     \t\n  ") is False


def test_is_pangram_repeated_letters():
    """Repeated letters do not make a non-pangram into a pangram."""
    assert is_pangram("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") is False


# ---------------------------------------------------------------------------
# is_pangram_faster
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", PANGRAM_STRINGS)
def test_is_pangram_faster_positive(text):
    """is_pangram_faster returns True for strings containing every letter."""
    assert is_pangram_faster(text) is True


@pytest.mark.parametrize("text", NON_PANGRAM_STRINGS)
def test_is_pangram_faster_negative(text):
    """is_pangram_faster returns False for strings missing at least one letter."""
    assert is_pangram_faster(text) is False


def test_is_pangram_faster_default_argument():
    """Calling is_pangram_faster with no arguments uses the default pangram."""
    assert is_pangram_faster() is True


def test_is_pangram_faster_empty_string():
    """An empty string is not a pangram for the faster implementation."""
    assert is_pangram_faster("") is False


def test_is_pangram_faster_ignores_non_letters():
    """Digits and punctuation are ignored by the faster implementation."""
    assert is_pangram_faster("The quick brown fox jumps over the lazy dog 42!") is True


def test_is_pangram_faster_missing_single_letter():
    """A single missing letter makes the faster implementation return False."""
    assert is_pangram_faster("abcdefghijklmnopqrstuvwxy") is False


# ---------------------------------------------------------------------------
# is_pangram_fastest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", PANGRAM_STRINGS)
def test_is_pangram_fastest_positive(text):
    """is_pangram_fastest returns True for strings containing every letter."""
    assert is_pangram_fastest(text) is True


@pytest.mark.parametrize("text", NON_PANGRAM_STRINGS)
def test_is_pangram_fastest_negative(text):
    """is_pangram_fastest returns False for strings missing at least one letter."""
    assert is_pangram_fastest(text) is False


def test_is_pangram_fastest_default_argument():
    """Calling is_pangram_fastest with no arguments uses the default pangram."""
    assert is_pangram_fastest() is True


def test_is_pangram_fastest_empty_string():
    """An empty string is not a pangram for the fastest implementation."""
    assert is_pangram_fastest("") is False


def test_is_pangram_fastest_ignores_non_letters():
    """Digits and punctuation are ignored by the fastest implementation."""
    assert is_pangram_fastest("The quick brown fox jumps over the lazy dog 42!") is True


def test_is_pangram_fastest_missing_single_letter():
    """A single missing letter makes the fastest implementation return False."""
    assert is_pangram_fastest("abcdefghijklmnopqrstuvwxy") is False


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", PANGRAM_STRINGS + NON_PANGRAM_STRINGS)
def test_all_implementations_agree(text):
    """All three implementations must return the same result for any input."""
    results = {impl(text) for impl in ALL_IMPLEMENTATIONS}
    assert len(results) == 1


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def test_benchmark_prints_three_timings():
    """benchmark() prints one timing line per implementation."""
    with patch("timeit.timeit", return_value=1.23) as mock_timeit, patch.object(
        builtins, "print"
    ) as mock_print:
        benchmark()

    assert mock_timeit.call_count == 3
    assert mock_print.call_count == 3
    for call in mock_print.call_args_list:
        assert call.args[0] == 1.23


def test_benchmark_returns_none():
    """benchmark() returns None."""
    with patch("timeit.timeit", return_value=0.0), patch.object(builtins, "print"):
        assert benchmark() is None


def test_benchmark_uses_expected_setup():
    """benchmark() passes a setup string importing all three functions."""
    with patch("timeit.timeit", return_value=0.0) as mock_timeit, patch.object(
        builtins, "print"
    ):
        benchmark()

    for call in mock_timeit.call_args_list:
        setup = call.kwargs.get("setup")
        assert setup is not None
        for name in ("is_pangram", "is_pangram_faster", "is_pangram_fastest"):
            assert name in setup