"""Tests for the dynamic-programming Fibonacci implementation."""

import builtins
from unittest.mock import patch

import pytest

from dynamic_programming.fibonacci import Fibonacci, main


# --------------------------------------------------------------------------- #
# Fibonacci.__init__
# --------------------------------------------------------------------------- #
def test_init_sets_default_sequence():
    """A fresh instance must start with the seed sequence [0, 1]."""
    fib = Fibonacci()
    assert fib.sequence == [0, 1]


def test_init_creates_independent_sequences():
    """Two instances must not share the same underlying list object."""
    first = Fibonacci()
    second = Fibonacci()
    first.get(10)
    assert second.sequence == [0, 1]
    assert first.sequence is not second.sequence


# --------------------------------------------------------------------------- #
# Fibonacci.get — happy paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        (5, [0, 1, 1, 2, 3]),
        (2, [0, 1]),
        (1, [0]),
        (0, []),
        (3, [0, 1, 1]),
        (7, [0, 1, 1, 2, 3, 5, 8]),
    ],
)
def test_get_returns_expected_prefix(index, expected):
    """`get` must return the first `index` Fibonacci numbers."""
    assert Fibonacci().get(index) == expected


def test_get_matches_doctest_examples():
    """The documented doctest examples must hold exactly."""
    assert Fibonacci().get(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert Fibonacci().get(5) == [0, 1, 1, 2, 3]


def test_get_returns_list_type():
    """The return value must always be a list."""
    result = Fibonacci().get(4)
    assert isinstance(result, list)


# --------------------------------------------------------------------------- #
# Fibonacci.get — edge cases / boundaries
# --------------------------------------------------------------------------- #
def test_get_zero_returns_empty_list():
    """Index 0 is the lower boundary and must yield an empty list."""
    assert Fibonacci().get(0) == []


def test_get_negative_index_uses_list_slicing():
    """Negative indices follow Python slicing semantics on the cache."""
    assert Fibonacci().get(-1) == [0]
    assert Fibonacci().get(-100) == []


def test_get_does_not_mutate_returned_list():
    """Mutating the returned list must not corrupt the internal cache."""
    fib = Fibonacci()
    result = fib.get(5)
    result.append(999)
    assert fib.get(5) == [0, 1, 1, 2, 3]


def test_get_is_cached_and_reused():
    """Repeated calls must reuse the cached sequence and stay consistent."""
    fib = Fibonacci()
    first = fib.get(10)
    second = fib.get(10)
    assert first == second
    # The cache holds the two seeds plus `index` appended values.
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def test_get_smaller_index_after_larger_does_not_shrink_cache():
    """Requesting a smaller index must not truncate the internal sequence."""
    fib = Fibonacci()
    fib.get(10)
    assert fib.get(3) == [0, 1, 1]
    assert len(fib.sequence) == 12


def test_get_extends_sequence_incrementally():
    """Each call must extend the cache only as far as needed."""
    fib = Fibonacci()
    fib.get(5)
    assert len(fib.sequence) == 7
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
    fib.get(8)
    assert len(fib.sequence) == 10
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_get_large_index_values_are_correct():
    """A larger index must produce the mathematically correct values."""
    result = Fibonacci().get(20)
    assert len(result) == 20
    assert result[:10] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert result[-1] == 4181
    # Verify the recurrence relation holds throughout.
    for i in range(2, len(result)):
        assert result[i] == result[i - 1] + result[i - 2]


def test_get_with_non_integer_index_raises_type_error():
    """A non-integer index must raise TypeError from the arithmetic."""
    with pytest.raises(TypeError):
        Fibonacci().get("5")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def test_main_prints_sequence_and_exits_on_exit():
    """`main` must print the sequence for a valid index and stop on 'exit'."""
    with patch.object(builtins, "input", side_effect=["5", "exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [str(call.args[0]) if call.args else "" for call in mock_print.call_args_list]
    assert "[0, 1, 1, 2, 3]" in printed


def test_main_exits_on_quit():
    """`main` must also terminate when the user types 'quit'."""
    with patch.object(builtins, "input", side_effect=["quit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [str(call.args[0]) if call.args else "" for call in mock_print.call_args_list]
    assert not any("[" in text for text in printed)


def test_main_handles_invalid_input_then_exits():
    """Non-numeric input must print an error message and continue looping."""
    with patch.object(
        builtins, "input", side_effect=["abc", "exit"]
    ), patch.object(builtins, "print") as mock_print:
        main()

    printed = [str(call.args[0]) if call.args else "" for call in mock_print.call_args_list]
    assert "Enter a number or 'exit'" in printed


def test_main_handles_multiple_queries():
    """`main` must handle several valid queries before exiting."""
    with patch.object(
        builtins, "input", side_effect=["3", "6", "exit"]
    ), patch.object(builtins, "print") as mock_print:
        main()

    printed = [str(call.args[0]) if call.args else "" for call in mock_print.call_args_list]
    assert "[0, 1, 1]" in printed
    assert "[0, 1, 1, 2, 3, 5]" in printed


def test_main_prints_banner():
    """`main` must print the introductory banner before prompting."""
    with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    assert mock_print.call_count >= 1
    banner = str(mock_print.call_args_list[0].args[0])
    assert "Fibonacci Series Using Dynamic Programming" in banner
