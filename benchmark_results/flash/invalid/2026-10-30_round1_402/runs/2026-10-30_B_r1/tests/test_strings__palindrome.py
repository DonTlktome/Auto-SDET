"""Tests for strings/palindrome.py.

Covers the four palindrome implementations plus the benchmark helper.
All external dependencies (timeit) are mocked so the suite runs offline
and fast.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import palindrome as pal  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / data
# ---------------------------------------------------------------------------

ALL_IMPLS = [
    pal.is_palindrome,
    pal.is_palindrome_traversal,
    pal.is_palindrome_recursive,
    pal.is_palindrome_slice,
]

IMPL_IDS = [
    "is_palindrome",
    "is_palindrome_traversal",
    "is_palindrome_recursive",
    "is_palindrome_slice",
]

# (input, expected) pairs used across every implementation.
CASES = [
    ("", True),                      # empty string
    ("A", True),                     # single character
    ("BB", True),                    # two identical characters
    ("AB", False),                   # two different characters
    ("rotor", True),                 # odd-length palindrome
    ("level", True),                 # odd-length palindrome
    ("MALAYALAM", True),             # case-sensitive palindrome
    ("String", False),               # non-palindrome
    ("ABC", False),                  # odd-length non-palindrome
    ("abcdba", False),               # near-palindrome
    ("amanaplanacanalpanama", True), # long palindrome
    ("abca", False),                 # mismatch at the very end
    ("aaabaaa", True),               # repeated characters
    ("  ", True),                    # whitespace only
    ("ab a ba", True),               # spaces are significant but symmetric
    ("ab a", False),                 # spaces are significant: 'b' vs ' ' mismatch
    ("ab a ", False),                # trailing space breaks symmetry
]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------

def test_test_data_is_consistent():
    """The module-level test_data must agree with Python's own reversal check."""
    assert pal.test_data, "test_data should not be empty"
    for key, value in pal.test_data.items():
        assert (key == key[::-1]) is value


def test_test_data_contains_expected_entries():
    """Spot-check a few known entries of the module-level test_data mapping."""
    assert pal.test_data["MALAYALAM"] is True
    assert pal.test_data["String"] is False
    assert pal.test_data["A"] is True
    assert pal.test_data["AB"] is False


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("value,expected", CASES)
def test_all_implementations_agree(impl, value, expected):
    """Every implementation must return the expected result for each case."""
    assert impl(value) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_all_implementations_agree_on_module_test_data(impl):
    """Every implementation must reproduce the module's own test_data mapping."""
    for key, expected in pal.test_data.items():
        assert impl(key) is expected


@pytest.mark.parametrize("value,expected", CASES)
def test_is_palindrome(value, expected):
    """Two-pointer implementation returns the correct boolean."""
    assert pal.is_palindrome(value) is expected


def test_is_palindrome_empty_string():
    """Empty string is trivially a palindrome (loop body never executes)."""
    assert pal.is_palindrome("") is True


def test_is_palindrome_single_char():
    """A one-character string is a palindrome."""
    assert pal.is_palindrome("x") is True


def test_is_palindrome_returns_bool_type():
    """The return value must be a real bool, not a truthy object."""
    assert isinstance(pal.is_palindrome("rotor"), bool)
    assert isinstance(pal.is_palindrome("rotorx"), bool)


def test_is_palindrome_does_not_mutate_input():
    """The input string must be unchanged after the call."""
    s = "amanaplanacanalpanama"
    pal.is_palindrome(s)
    assert s == "amanaplanacanalpanama"


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", CASES)
def test_is_palindrome_traversal(value, expected):
    """Half-traversal implementation returns the correct boolean."""
    assert pal.is_palindrome_traversal(value) is expected


def test_is_palindrome_traversal_empty_string():
    """Empty string: range(0) yields no comparisons, so result is True."""
    assert pal.is_palindrome_traversal("") is True


def test_is_palindrome_traversal_odd_length_middle_ignored():
    """For odd lengths the middle character is never compared."""
    assert pal.is_palindrome_traversal("abcba") is True
    assert pal.is_palindrome_traversal("abcda") is False


def test_is_palindrome_traversal_two_chars():
    """Boundary: exactly two characters, both matching and mismatching."""
    assert pal.is_palindrome_traversal("aa") is True
    assert pal.is_palindrome_traversal("ab") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", CASES)
def test_is_palindrome_recursive(value, expected):
    """Recursive implementation returns the correct boolean."""
    assert pal.is_palindrome_recursive(value) is expected


@pytest.mark.parametrize("value", ["", "a", "Z"])
def test_is_palindrome_recursive_base_cases(value):
    """Strings of length <= 1 hit the base case and return True."""
    assert pal.is_palindrome_recursive(value) is True


def test_is_palindrome_recursive_deep_recursion():
    """A long palindrome exercises many recursive levels without error."""
    s = "ab" * 200 + "c" + "ba" * 200
    assert pal.is_palindrome_recursive(s) is True


def test_is_palindrome_recursive_mismatch_at_outermost_pair():
    """A mismatch on the first/last characters returns False immediately."""
    assert pal.is_palindrome_recursive("abcba") is True
    assert pal.is_palindrome_recursive("abcda") is False
    assert pal.is_palindrome_recursive("xabcy") is False


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", CASES)
def test_is_palindrome_slice(value, expected):
    """Slice-reversal implementation returns the correct boolean."""
    assert pal.is_palindrome_slice(value) is expected


def test_is_palindrome_slice_empty_string():
    """Empty string equals its own reversal."""
    assert pal.is_palindrome_slice("") is True


def test_is_palindrome_slice_is_case_sensitive():
    """Case differences make a string a non-palindrome."""
    assert pal.is_palindrome_slice("Aa") is False
    assert pal.is_palindrome_slice("aA") is False


def test_is_palindrome_slice_unicode():
    """Unicode characters are handled correctly."""
    assert pal.is_palindrome_slice("été") is True
    assert pal.is_palindrome_slice("éte") is False


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------

def test_benchmark_function_calls_timeit_with_expected_arguments(capsys):
    """benchmark_function must call timeit with the right stmt/setup/number."""
    fake_timeit = MagicMock(return_value=0.12345)

    with patch.object(pal, "timeit", fake_timeit):
        pal.benchmark_function("is_palindrome_slice")

    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["stmt"] == (
        "all(is_palindrome_slice(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome_slice"
    assert kwargs["number"] == 500000

    out = capsys.readouterr().out
    assert "is_palindrome_slice" in out
    assert "500,000" in out
    assert "0.12345" in out


@pytest.mark.parametrize(
    "name",
    ["is_palindrome", "is_palindrome_traversal", "is_palindrome_recursive", "is_palindrome_slice"],
)
def test_benchmark_function_accepts_each_implementation_name(name, capsys):
    """benchmark_function works for every implementation name and prints it."""
    fake_timeit = MagicMock(return_value=1.0)

    with patch.object(pal, "timeit", fake_timeit):
        result = pal.benchmark_function(name)

    assert result is None
    assert name in fake_timeit.call_args.kwargs["stmt"]
    assert name in fake_timeit.call_args.kwargs["setup"]
    assert name in capsys.readouterr().out


def test_benchmark_function_formats_output_width(capsys):
    """The printed name field is left-aligned and padded to 35 characters."""
    fake_timeit = MagicMock(return_value=0.5)

    with patch.object(pal, "timeit", fake_timeit):
        pal.benchmark_function("is_palindrome")

    out = capsys.readouterr().out
    assert out.startswith("is_palindrome" + " " * (35 - len("is_palindrome")))
    assert out.endswith("seconds\n")


def test_benchmark_function_propagates_timeit_errors():
    """Errors raised by timeit are not swallowed."""
    with patch.object(pal, "timeit", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            pal.benchmark_function("is_palindrome")
