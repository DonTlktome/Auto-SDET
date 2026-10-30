"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import builtins
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by file path (the module name "palindrome" is
# generic, so we load it explicitly to avoid collisions).
# ---------------------------------------------------------------------------
MODULE_PATH = Path(__file__).resolve().parent.parent / "strings" / "palindrome.py"

if not MODULE_PATH.exists():  # pragma: no cover - fallback for other layouts
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
    """Spot-check a few well-known entries of the module's test_data mapping."""
    assert palindrome.test_data["MALAYALAM"] is True
    assert palindrome.test_data["String"] is False
    assert palindrome.test_data["A"] is True
    assert palindrome.test_data["AB"] is False


# ---------------------------------------------------------------------------
# Shared parametrized coverage across all four implementations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_impl_matches_test_data(impl):
    """Every implementation must reproduce the expected results in test_data."""
    for key, value in palindrome.test_data.items():
        assert impl(key) is value, f"{impl.__name__}({key!r}) != {value}"


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", True),          # empty string is trivially a palindrome
        ("a", True),         # single character
        ("aa", True),        # two identical characters
        ("ab", False),       # two different characters
        ("aba", True),       # odd-length palindrome
        ("abba", True),      # even-length palindrome
        ("abc", False),      # odd-length non-palindrome
        ("abca", False),     # mismatch at the outer pair
        ("abcba", True),
        ("abccbx", False),   # mismatch near the middle
        (" ", True),         # whitespace-only
        ("  ", True),
        ("a b a", True),     # spaces are compared literally
        ("a b", False),
        ("Aa", False),       # case sensitive
        ("12321", True),     # digits
        ("12345", False),
        ("!@#$%$#@!", True), # punctuation
        ("x" * 1000, True),  # long palindrome
        ("x" * 999 + "y", False),  # long non-palindrome
    ],
)
def test_impl_edge_cases(impl, text, expected):
    """Boundary and edge inputs behave identically across implementations."""
    assert impl(text) is expected


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_impl_returns_bool_type(impl):
    """Each implementation returns an actual bool, not a truthy/falsy value."""
    assert isinstance(impl("racecar"), bool)
    assert isinstance(impl("racecars"), bool)


@pytest.mark.parametrize("impl", ALL_IMPLS, ids=IMPL_IDS)
def test_impl_does_not_mutate_input(impl):
    """The input string is immutable and must be returned unchanged in content."""
    text = "amanaplanacanalpanama"
    original = str(text)
    impl(text)
    assert text == original


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------
def test_is_palindrome_happy_path():
    """A classic palindrome is detected by the two-pointer implementation."""
    assert palindrome.is_palindrome("racecar") is True


def test_is_palindrome_non_palindrome():
    """A non-palindrome returns False."""
    assert palindrome.is_palindrome("racecars") is False


def test_is_palindrome_empty_string():
    """An empty string has no mismatching pair, so it is a palindrome."""
    assert palindrome.is_palindrome("") is True


def test_is_palindrome_single_char():
    """A one-character string is always a palindrome."""
    assert palindrome.is_palindrome("z") is True


def test_is_palindrome_mismatch_at_first_pair():
    """Mismatch on the very first comparison short-circuits to False."""
    assert palindrome.is_palindrome("ab...a") is False


def test_is_palindrome_mismatch_at_innermost_pair():
    """Mismatch only at the innermost pair still yields False."""
    assert palindrome.is_palindrome("abxba") is True
    assert palindrome.is_palindrome("abxca") is False


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------
def test_is_palindrome_traversal_happy_path():
    """Half-traversal implementation detects a palindrome."""
    assert palindrome.is_palindrome_traversal("level") is True


def test_is_palindrome_traversal_non_palindrome():
    """Half-traversal implementation rejects a non-palindrome."""
    assert palindrome.is_palindrome_traversal("levels") is False


def test_is_palindrome_traversal_empty_string():
    """Empty string: range(0) is empty, so all() is True."""
    assert palindrome.is_palindrome_traversal("") is True


def test_is_palindrome_traversal_odd_length_ignores_middle():
    """For odd lengths the middle character is never compared."""
    assert palindrome.is_palindrome_traversal("abcba") is True
    assert palindrome.is_palindrome_traversal("abcda") is False


def test_is_palindrome_traversal_two_chars():
    """Two-character boundary case compares exactly one pair."""
    assert palindrome.is_palindrome_traversal("aa") is True
    assert palindrome.is_palindrome_traversal("ab") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------
def test_is_palindrome_recursive_happy_path():
    """Recursive implementation detects a palindrome."""
    assert palindrome.is_palindrome_recursive("rotor") is True


def test_is_palindrome_recursive_non_palindrome():
    """Recursive implementation rejects a non-palindrome."""
    assert palindrome.is_palindrome_recursive("rotors") is False


@pytest.mark.parametrize("text", ["", "a", "ab", "aa"])
def test_is_palindrome_recursive_base_cases(text):
    """Base cases: len <= 1 returns True; len == 2 compares one pair."""
    expected = text in ("", "a", "aa")
    assert palindrome.is_palindrome_recursive(text) is expected


def test_is_palindrome_recursive_deep_recursion():
    """A moderately long palindrome exercises many recursive frames."""
    text = "ab" * 200 + "c" + "ba" * 200
    assert palindrome.is_palindrome_recursive(text) is True


def test_is_palindrome_recursive_calls_itself():
    """The recursive implementation delegates to itself for the inner slice."""
    with patch.object(
        palindrome, "is_palindrome_recursive", wraps=palindrome.is_palindrome_recursive
    ) as spy:
        assert palindrome.is_palindrome_recursive("abba") is True
    assert spy.call_count > 1


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------
def test_is_palindrome_slice_happy_path():
    """Slice-reversal implementation detects a palindrome."""
    assert palindrome.is_palindrome_slice("malayalam") is True


def test_is_palindrome_slice_non_palindrome():
    """Slice-reversal implementation rejects a non-palindrome."""
    assert palindrome.is_palindrome_slice("malayalams") is False


def test_is_palindrome_slice_empty_string():
    """Empty string equals its own reversal."""
    assert palindrome.is_palindrome_slice("") is True


def test_is_palindrome_slice_unicode():
    """Unicode characters are compared correctly by the slice implementation."""
    assert palindrome.is_palindrome_slice("été") is True
    assert palindrome.is_palindrome_slice("étéé") is False


# ---------------------------------------------------------------------------
# Cross-implementation agreement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
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
        "x" * 501,
        "x" * 500 + "y",
    ],
)
def test_all_implementations_agree(text):
    """All four algorithms must produce identical verdicts for the same input."""
    results = {impl.__name__: impl(text) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------
def test_benchmark_function_prints_result(capsys):
    """benchmark_function prints a formatted line containing the function name."""
    with patch.object(palindrome, "timeit", return_value=0.12345) as mock_timeit:
        palindrome.benchmark_function("is_palindrome_slice")

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
    """benchmark_function is a side-effecting helper and returns None."""
    with patch.object(palindrome, "timeit", return_value=0.0):
        assert palindrome.benchmark_function("is_palindrome") is None


@pytest.mark.parametrize(
    "name", ["is_palindrome", "is_palindrome_traversal", "is_palindrome_recursive"]
)
def test_benchmark_function_builds_expected_statements(name):
    """The generated stmt/setup strings reference the requested function name."""
    with patch.object(palindrome, "timeit", return_value=1.0) as mock_timeit:
        palindrome.benchmark_function(name)

    kwargs = mock_timeit.call_args.kwargs
    assert kwargs["stmt"] == (
        f"all({name}(key) == value for key, value in test_data.items())"
    )
    assert kwargs["setup"] == f"from __main__ import test_data, {name}"


def test_benchmark_function_propagates_timeit_errors():
    """Errors raised by timeit are not swallowed by benchmark_function."""
    with patch.object(palindrome, "timeit", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            palindrome.benchmark_function("is_palindrome")


def test_benchmark_function_uses_real_timeit_with_small_number():
    """Sanity check that the real timeit path works (patched number via wrapper)."""
    real_timeit = palindrome.timeit
    captured = {}

    def fake_timeit(stmt, setup, number):
        captured["number"] = number
        # The production helper builds a ``from __main__ import ...`` setup,
        # which cannot work under pytest (``__main__`` is pytest itself).
        # Feed the statement the module namespace instead so the *real*
        # timeit implementation is still exercised.
        return real_timeit(
            stmt=stmt,
            number=1,
            globals={
                "test_data": palindrome.test_data,
                "is_palindrome_slice": palindrome.is_palindrome_slice,
                "all": all,
            },
        )

    with patch.object(palindrome, "timeit", side_effect=fake_timeit):
        palindrome.benchmark_function("is_palindrome_slice")

    assert captured["number"] == 500000
