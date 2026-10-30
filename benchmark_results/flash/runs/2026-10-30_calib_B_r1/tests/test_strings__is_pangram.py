"""Tests for the pangram detection functions in ``is_pangram.py``."""

from unittest.mock import patch

import pytest

from is_pangram import (
    benchmark,
    is_pangram,
    is_pangram_fastest,
    is_pangram_faster,
)


# ---------------------------------------------------------------------------
# Shared fixtures / data
# ---------------------------------------------------------------------------

PANGRAM_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Waltz, bad nymph, for quick jigs vex.",
    "Jived fox nymph grabs quick waltz.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
]

NON_PANGRAM_SENTENCES = [
    "My name is Unknown",
    "The quick brown fox jumps over the la_y dog",
    "",
    "abcdefghijklmnopqrstuvwxy",  # 25 letters, missing 'z'
    "1234567890 !@#$%^&*()",
]

ALL_IMPLEMENTATIONS = [is_pangram, is_pangram_faster, is_pangram_fastest]


# ---------------------------------------------------------------------------
# is_pangram
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", PANGRAM_SENTENCES)
def test_is_pangram_positive(sentence):
    """A sentence containing every letter of the alphabet returns True."""
    assert is_pangram(sentence) is True


@pytest.mark.parametrize("sentence", NON_PANGRAM_SENTENCES)
def test_is_pangram_negative(sentence):
    """Sentences missing at least one letter (or empty) return False."""
    assert is_pangram(sentence) is False


def test_is_pangram_default_argument():
    """Calling with no arguments uses the default pangram sentence."""
    assert is_pangram() is True


def test_is_pangram_ignores_non_alphabetic_characters():
    """Digits, punctuation and underscores do not count as letters."""
    assert is_pangram("The quick brown fox jumps over the la_y dog") is False
    assert is_pangram("The quick brown fox jumps over the lazy dog!!!") is True


def test_is_pangram_handles_whitespace_only():
    """A whitespace-only string contains no letters and is not a pangram."""
    assert is_pangram("   \t\n  ") is False


def test_is_pangram_exactly_26_letters():
    """A string with exactly the 26 lowercase letters is a pangram."""
    assert is_pangram("abcdefghijklmnopqrstuvwxyz") is True


def test_is_pangram_missing_single_letter():
    """Removing one letter from the alphabet makes it a non-pangram."""
    assert is_pangram("abcdefghijklmnopqrstuvwxy") is False


# ---------------------------------------------------------------------------
# is_pangram_faster
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", PANGRAM_SENTENCES)
def test_is_pangram_faster_positive(sentence):
    """The faster implementation detects pangrams correctly."""
    assert is_pangram_faster(sentence) is True


@pytest.mark.parametrize("sentence", NON_PANGRAM_SENTENCES)
def test_is_pangram_faster_negative(sentence):
    """The faster implementation rejects non-pangrams."""
    assert is_pangram_faster(sentence) is False


def test_is_pangram_faster_default_argument():
    """Calling with no arguments uses the default pangram sentence."""
    assert is_pangram_faster() is True


def test_is_pangram_faster_ignores_non_letters():
    """Digits, punctuation and whitespace are ignored by the flag table."""
    assert is_pangram_faster("The quick brown fox jumps over the lazy dog 123!") is True
    assert is_pangram_faster("   \t\n  ") is False


def test_is_pangram_faster_raises_on_non_ascii_letters():
    """Non-ASCII letters fall outside the 26-slot flag table.

    ``is_pangram_faster`` indexes a fixed 26-element list with
    ``ord(char) - 97`` (or ``- 65``), so lowercase/uppercase letters beyond
    the ASCII range produce an out-of-range index instead of being ignored.
    """
    with pytest.raises(IndexError):
        is_pangram_faster("abcdefghijklmnopqrstuvwxyz\u00e9")

    with pytest.raises(IndexError):
        is_pangram_faster("\u00e9")


def test_is_pangram_faster_handles_whitespace_only():
    """A whitespace-only string is not a pangram."""
    assert is_pangram_faster("   \t\n  ") is False


# ---------------------------------------------------------------------------
# is_pangram_fastest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", PANGRAM_SENTENCES)
def test_is_pangram_fastest_positive(sentence):
    """The fastest implementation detects pangrams correctly."""
    assert is_pangram_fastest(sentence) is True


@pytest.mark.parametrize("sentence", NON_PANGRAM_SENTENCES)
def test_is_pangram_fastest_negative(sentence):
    """The fastest implementation rejects non-pangrams."""
    assert is_pangram_fastest(sentence) is False


def test_is_pangram_fastest_default_argument():
    """Calling with no arguments uses the default pangram sentence."""
    assert is_pangram_fastest() is True


def test_is_pangram_fastest_handles_whitespace_only():
    """A whitespace-only string is not a pangram."""
    assert is_pangram_fastest("   \t\n  ") is False


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", PANGRAM_SENTENCES + NON_PANGRAM_SENTENCES)
def test_all_implementations_agree(sentence):
    """All three implementations must return the same result for any input."""
    results = {impl(sentence) for impl in ALL_IMPLEMENTATIONS}
    assert len(results) == 1, f"Implementations disagree for {sentence!r}"


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def test_benchmark_prints_three_timings():
    """benchmark() prints one timing line per implementation."""
    with patch("timeit.timeit", return_value=1.23) as mock_timeit:
        benchmark()

    assert mock_timeit.call_count == 3
    for call in mock_timeit.call_args_list:
        assert call.args[0] in {
            "is_pangram()",
            "is_pangram_faster()",
            "is_pangram_fastest()",
        }
        assert "setup" in call.kwargs


def test_benchmark_output_goes_to_stdout(capsys):
    """benchmark() writes its timings to standard output."""
    with patch("timeit.timeit", return_value=0.5):
        benchmark()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(line.strip() == "0.5" for line in lines)


def test_benchmark_returns_none():
    """benchmark() is a side-effect-only function and returns None."""
    with patch("timeit.timeit", return_value=0.0):
        assert benchmark() is None
