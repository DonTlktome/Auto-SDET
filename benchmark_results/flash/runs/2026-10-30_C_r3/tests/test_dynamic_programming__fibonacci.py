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
        (3, [0, 1, 1]),
        (7, [0, 1, 1, 2, 3, 5, 8]),
    ],
)
def test_get_returns_expected_prefix(index, expected):
    """`get` must return the first `index` Fibonacci numbers."""
    assert Fibonacci().get(index) == expected


def test_get_matches_docstring_examples():
    """The doctest examples from the source must hold."""
    assert Fibonacci().get(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert Fibonacci().get(5) == [0, 1, 1, 2, 3]


def test_get_is_idempotent_for_same_index():
    """Calling `get` twice with the same index must return equal results."""
    fib = Fibonacci()
    assert fib.get(8) == fib.get(8)


def test_get_extends_sequence_incrementally():
    """Successive calls must grow the internal sequence without recomputation."""
    fib = Fibonacci()
    assert fib.get(5) == [0, 1, 1, 2, 3]
    assert len(fib.sequence) == 7
    assert fib.get(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert len(fib.sequence) == 12


def test_get_smaller_index_after_larger_does_not_shrink_sequence():
    """Requesting a smaller index must slice the cached sequence, not truncate it."""
    fib = Fibonacci()
    fib.get(10)
    assert len(fib.sequence) == 12
    assert fib.get(3) == [0, 1, 1]
    assert len(fib.sequence) == 12


def test_get_returns_new_list_object():
    """The returned list must be a copy so callers cannot corrupt the cache."""
    fib = Fibonacci()
    result = fib.get(5)
    result.append(999)
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
    assert fib.get(5) == [0, 1, 1, 2, 3]


# ---------------------------------------------------------------------------
# Fibonacci.get — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "expected_sequence", "expected_result"),
    [
        (0, [0, 1], []),
        (1, [0, 1, 1], [0]),
        (2, [0, 1, 1, 2], [0, 1]),
    ],
)
def test_get_small_index_only_appends_needed_values(
    index, expected_sequence, expected_result
):
    """Indices at or below the seed length only grow the cache as required."""
    fib = Fibonacci()
    assert fib.get(index) == expected_result
    assert fib.sequence == expected_sequence


@pytest.mark.parametrize(("index", "expected"), [(-1, [0]), (-5, [])])
def test_get_negative_index_slices_from_end(index, expected):
    """Negative indices slice from the end and must not raise."""
    fib = Fibonacci()
    assert fib.get(index) == expected
    assert fib.sequence == [0, 1]


def test_get_large_index_values_are_correct():
    """A larger index must produce a sequence whose last element is the sum of the previous two."""
    fib = Fibonacci()
    result = fib.get(50)
    assert len(result) == 50
    assert result[:2] == [0, 1]
    for i in range(2, len(result)):
        assert result[i] == result[i - 1] + result[i - 2]
    assert result[-1] == 7778742049


def test_get_does_not_mutate_input_index():
    """The integer argument must be treated as immutable input."""
    fib = Fibonacci()
    index = 6
    fib.get(index)
    assert index == 6


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_exits_immediately_on_exit_command():
    """`main` must terminate when the user types 'exit'."""
    with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    assert mock_print.called


def test_main_exits_on_quit_command():
    """`main` must also terminate when the user types 'quit'."""
    with patch.object(builtins, "input", side_effect=["quit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    assert mock_print.called


def test_main_prints_fibonacci_for_valid_index():
    """A numeric input must cause the computed sequence to be printed."""
    with patch.object(builtins, "input", side_effect=["5", "exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert [0, 1, 1, 2, 3] in printed


def test_main_handles_invalid_input_then_exits():
    """Non-numeric input must print an error message and keep prompting."""
    with patch.object(builtins, "input", side_effect=["abc", "exit"]), patch.object(
        builtins, "print"
    ) as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert "Enter a number or 'exit'" in printed


def test_main_handles_multiple_invalid_inputs():
    """Repeated invalid inputs must each produce an error message."""
    with patch.object(
        builtins, "input", side_effect=["foo", "bar", "exit"]
    ), patch.object(builtins, "print") as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert printed.count("Enter a number or 'exit'") == 2


def test_main_uses_single_fibonacci_instance_across_prompts():
    """The same Fibonacci instance must be reused so the cache persists."""
    with patch.object(
        builtins, "input", side_effect=["3", "6", "exit"]
    ), patch.object(builtins, "print") as mock_print:
        main()

    printed = [call.args[0] for call in mock_print.call_args_list if call.args]
    assert [0, 1, 1] in printed
    assert [0, 1, 1, 2, 3, 5] in printed


def test_main_propagates_keyboard_interrupt():
    """Ctrl-C during input must propagate out of `main`."""
    with patch.object(builtins, "input", side_effect=KeyboardInterrupt), patch.object(
        builtins, "print"
    ):
        with pytest.raises(KeyboardInterrupt):
            main()