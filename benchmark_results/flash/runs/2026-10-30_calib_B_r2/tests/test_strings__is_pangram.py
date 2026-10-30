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


ALL_FUNCTIONS = [is_pangram, is_pangram_faster, is_pangram_fastest]

PANGRAMS = [
    "The quick brown fox jumps over the lazy dog",
    "Waltz, bad nymph, for quick jigs vex.",
    "Jived fox nymph grabs quick waltz.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
]

NON_PANGRAMS = [
    "My name is Unknown",
    "The quick brown fox jumps over the la_y dog",
    "",
    "a",
    "abcdefghijklmnopqrstuvwxy",  # 25 letters, missing 'z'
    "1234567890!@#$%^&*()",
]


# ---------------------------------------------------------------------------
# is_pangram
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
@pytest.mark.parametrize("text", PANGRAMS)
def test_pangram_positive(func, text):
    """Each implementation must return True for known pangram strings."""
    assert func(text) is True


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
@pytest.mark.parametrize("text", NON_PANGRAMS)
def test_pangram_negative(func, text):
    """Each implementation must return False for non-pangram strings."""
    assert func(text) is False


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_default_argument_is_pangram(func):
    """Calling with no arguments uses the default pangram sentence."""
    assert func() is True


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_case_insensitivity(func):
    """Upper-case and mixed-case pangrams are detected correctly."""
    assert func("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG") is True
    assert func("tHe QuIcK bRoWn FoX jUmPs OvEr ThE lAzY dOg") is True


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_punctuation_and_digits_ignored(func):
    """Non-alphabetic characters must not affect the result."""
    assert func("The quick brown fox jumps over the lazy dog!!! 12345") is True
    assert func("!!!") is False


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_repeated_letters(func):
    """Repeated letters do not break the uniqueness counting."""
    assert func("aaaaaaaaaabcdefghijklmnopqrstuvwxyz") is True


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_missing_single_letter(func):
    """Removing exactly one distinct letter makes the string a non-pangram."""
    for letter in "abcdefghijklmnopqrstuvwxyz":
        text = "abcdefghijklmnopqrstuvwxyz".replace(letter, "")
        assert func(text) is False


@pytest.mark.parametrize("func", ALL_FUNCTIONS)
def test_whitespace_only(func):
    """A whitespace-only string is not a pangram."""
    assert func("   \t\n  ") is False


def test_unicode_letters():
    """Non-ASCII alphabetic characters must not be counted as a-z letters.

    Only ``is_pangram`` guarantees this behaviour, because it explicitly
    restricts the accepted characters to the ASCII range ``"a" <= c <= "z"``.
    ``is_pangram_faster`` and ``is_pangram_fastest`` rely on ``str.islower`` /
    ``str.isupper`` / ``str.isalpha``, which also accept non-ASCII letters, so
    they are covered by dedicated tests below instead.
    """
    # 'é' is alphabetic but not in a-z; the rest of the alphabet is present.
    assert is_pangram("abcdefghijklmnopqrstuvwxyzé") is True
    # Only accented letters -> not a pangram.
    assert is_pangram("éèêëàâäôöûüç") is False


def test_unicode_letters_fastest_not_a_pangram():
    """Accented letters are counted by isalpha(), so this is not a pangram."""
    assert is_pangram_fastest("éèêëàâäôöûüç") is False


def test_unicode_letters_faster_rejects_out_of_range_letters():
    """is_pangram_faster only handles a-z, so other letters raise IndexError."""
    with pytest.raises(IndexError):
        is_pangram_faster("éèêëàâäôöûüç")


# ---------------------------------------------------------------------------
# is_pangram specific behaviour
# ---------------------------------------------------------------------------


def test_is_pangram_underscore_breaks_pangram():
    """An underscore replaces a letter, so the string is not a pangram."""
    assert is_pangram("The quick brown fox jumps over the la_y dog") is False


def test_is_pangram_returns_bool_type():
    """The return value must be a real bool, not a truthy object."""
    result = is_pangram("The quick brown fox jumps over the lazy dog")
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# is_pangram_faster specific behaviour
# ---------------------------------------------------------------------------


def test_is_pangram_faster_returns_bool_type():
    """The return value must be a real bool."""
    assert isinstance(is_pangram_faster("abc"), bool)


def test_is_pangram_faster_handles_uppercase_only():
    """An all-uppercase alphabet string is detected as a pangram."""
    assert is_pangram_faster("ABCDEFGHIJKLMNOPQRSTUVWXYZ") is True


# ---------------------------------------------------------------------------
# is_pangram_fastest specific behaviour
# ---------------------------------------------------------------------------


def test_is_pangram_fastest_returns_bool_type():
    """The return value must be a real bool."""
    assert isinstance(is_pangram_fastest("abc"), bool)


def test_is_pangram_fastest_ignores_non_alpha():
    """Digits and punctuation are filtered out by isalpha()."""
    assert is_pangram_fastest("abc123defghijklmnopqrstuvwxyz") is True


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_prints_three_timings():
    """benchmark() prints one timing line per implementation."""
    with patch("timeit.timeit", return_value=0.123) as mock_timeit:
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
    """benchmark() writes its results to standard output."""
    with patch("timeit.timeit", return_value=1.5):
        benchmark()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(line.strip() == "1.5" for line in lines)


def test_benchmark_returns_none():
    """benchmark() has no return value."""
    with patch("timeit.timeit", return_value=0.0):
        assert benchmark() is None
