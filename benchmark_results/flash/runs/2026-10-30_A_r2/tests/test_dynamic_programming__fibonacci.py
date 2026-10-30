"""Tests for the dynamic-programming Fibonacci implementation."""

import builtins
from unittest.mock import patch

import pytest

from dynamic_programming.fibonacci import Fibonacci, main


# ---------------------------------------------------------------------------
# Fibonacci.__init__
# ---------------------------------------------------------------------------


def test_init_sets_default_sequence():
    """A fresh instance must start with the seed sequence [0, 1]."""
    fib = Fibonacci()
    assert fib.sequence == [0, 1]


def test_init_creates_independent_sequences():
    """Two instances must not share the same mutable list object."""
    first = Fibonacci()
    second = Fibonacci()
    first.get(10)
    assert second.sequence == [0, 1]
    assert first.sequence is not second.sequence


# ---------------------------------------------------------------------------
# Fibonacci.get — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        (5, [0, 1, 1, 2, 3]),
        (2, [0, 1]),
        (1, [0]),
        (0, []),
        (1, [0]),
        (3, [0, 1, 1]),
        (7, [0, 1, 1, 2, 3, 5, 8]),
    ],
)
def test_get_returns_expected_prefix(index, expected):
    """`get` must return the first `index` elements of the Fibonacci sequence."""
    assert Fibonacci().get(index) == expected


def test_get_matches_doctest_examples():
    """The documented doctest examples must hold exactly."""
    assert Fibonacci().get(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert Fibonacci().get(5) == [0, 1, 1, 2, 3]


def test_get_returns_list_type():
    """The return value must always be a list."""
    result = Fibonacci().get(6)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fibonacci.get — edge cases / boundaries
#
# The implementation caches ``index + 2`` values: the ``[0, 1]`` seed pair
# plus one appended value for every computed Fibonacci number.  ``get`` never
# truncates that cache, it only returns a slice (a copy) of it, so the
# assertions below check the cache size the implementation really produces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, []),
        (-1, [0]),
        (-5, []),
        (-100, []),
    ],
)
def test_get_non_positive_index_slices_without_extending(index, expected):
    """Indices <= 0 must never grow the cache; only a plain slice is returned."""
    fib = Fibonacci()
    assert fib.get(index) == expected
    assert fib.sequence == [0, 1]


def test_get_does_not_mutate_sequence_for_small_index():
    """Requesting fewer elements than cached must not extend the cache."""
    fib = Fibonacci()
    fib.get(6)
    cached = list(fib.sequence)
    fib.get(2)
    assert fib.sequence == cached


def test_get_is_idempotent_for_same_index():
    """Repeated calls with the same index must return equal results."""
    fib = Fibonacci()
    first = fib.get(12)
    second = fib.get(12)
    assert first == second
    assert len(fib.sequence) == 14  # 12 requested values + the [0, 1] seed pair


def test_get_caches_and_reuses_computed_values():
    """A larger request after a smaller one must extend, not recompute, the cache."""
    fib = Fibonacci()
    small = fib.get(5)
    large = fib.get(10)
    assert large[: len(small)] == small
    assert len(fib.sequence) == 12  # 10 requested values + the [0, 1] seed pair


def test_get_sequence_grows_monotonically():
    """The internal cache must only ever grow when larger indices are requested."""
    fib = Fibonacci()
    lengths = []
    for index in (3, 6, 4, 9, 2):
        fib.get(index)
        lengths.append(len(fib.sequence))
    assert lengths == [5, 8, 8, 11, 11]


def test_get_large_index_values_are_correct():
    """A moderately large index must produce the known Fibonacci value."""
    result = Fibonacci().get(50)
    assert len(result) == 50
    assert result[-1] == 7778742049
    assert result[-2] == 4807526976


def test_get_returns_copy_not_internal_list():
    """Mutating the returned list must not corrupt the internal cache."""
    fib = Fibonacci()
    result = fib.get(6)
    result.append(999)
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13]
    assert fib.get(6) == [0, 1, 1, 2, 3, 5]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_prints_sequence_for_valid_input():
    """`main` must print the Fibonacci prefix for a numeric input then exit."""
    with patch.object(builtins, "input", side_effect=["5", "exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert [0, 1, 1, 2, 3] in printed


def test_main_exits_on_quit_keyword():
    """`main` must terminate immediately when the user types 'quit'."""
    with patch.object(builtins, "input", side_effect=["quit"]) as mock_input, patch.object(
        builtins, "print"
    ):
        main()

    assert mock_input.call_count == 1


def test_main_handles_invalid_input_then_exits():
    """Non-numeric input must print an error message and keep looping."""
    with patch.object(
        builtins, "input", side_effect=["abc", "exit"]
    ) as mock_input, patch.object(builtins, "print") as mock_print:
        main()

    assert mock_input.call_count == 2
    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert "Enter a number or 'exit'" in printed


def test_main_handles_negative_input():
    """A negative index must print an empty list rather than raising."""
    with patch.object(builtins, "input", side_effect=["-3", "exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert [] in printed


def test_main_prints_banner():
    """`main` must print the introductory banner before prompting."""
    with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    first_call = mock_print.call_args_list[0]
    assert "Fibonacci Series Using Dynamic Programming" in first_call.args[0]
