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
    """Two instances must not share the same underlying list object."""
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
    result = Fibonacci().get(4)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Fibonacci.get — caching / statefulness
# ---------------------------------------------------------------------------


def test_get_caches_sequence_between_calls():
    """Repeated calls must extend the internal cache rather than recompute.

    The implementation caches two extra terms beyond the requested prefix,
    so after ``get(n)`` (n >= 1) the internal sequence holds ``n + 2`` items.
    """
    fib = Fibonacci()
    fib.get(5)
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
    fib.get(8)
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_get_smaller_index_after_larger_does_not_shrink_cache():
    """Requesting a smaller index must not truncate the cached sequence."""
    fib = Fibonacci()
    fib.get(10)
    cached = list(fib.sequence)
    assert fib.get(3) == [0, 1, 1]
    assert fib.sequence == cached


def test_get_is_idempotent_for_same_index():
    """Calling `get` twice with the same index must yield identical results."""
    fib = Fibonacci()
    first = fib.get(12)
    second = fib.get(12)
    assert first == second
    assert fib.sequence[: len(first)] == first
    assert len(fib.sequence) == len(first) + 2


def test_get_does_not_mutate_returned_list_cache():
    """Mutating the returned list must not corrupt the internal cache."""
    fib = Fibonacci()
    result = fib.get(6)
    result.append(999)
    assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13]
    assert fib.get(6) == [0, 1, 1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Fibonacci.get — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "expected_cache"),
    [
        (0, [0, 1]),
        (1, [0, 1, 1]),
        (2, [0, 1, 1, 2]),
    ],
)
def test_get_at_or_below_seed_length(index, expected_cache):
    """Small indices must only extend the cache by the terms they need."""
    fib = Fibonacci()
    expected = [0, 1][:index]
    assert fib.get(index) == expected
    assert fib.sequence == expected_cache


def test_get_negative_index_slices_off_last_element():
    """A negative index slices one element off the seed and keeps the cache."""
    fib = Fibonacci()
    assert fib.get(-1) == [0]
    assert fib.sequence == [0, 1]


def test_get_large_index_has_correct_length_and_values():
    """A large index must produce a sequence of the right length and values."""
    fib = Fibonacci()
    result = fib.get(50)
    assert len(result) == 50
    assert result[:10] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    # Known value: F(49) == 7778742049
    assert result[-1] == 7778742049
    assert result[-2] == 4807526976


def test_get_sequence_is_non_decreasing_after_seed():
    """From index 2 onward the sequence must be non-decreasing."""
    result = Fibonacci().get(30)
    assert all(a <= b for a, b in zip(result[1:], result[2:]))


def test_get_recurrence_relation_holds():
    """Each element from index 2 must equal the sum of the two before it."""
    result = Fibonacci().get(25)
    for i in range(2, len(result)):
        assert result[i] == result[i - 1] + result[i - 2]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_exits_immediately_on_exit_command():
    """`main` must terminate when the user types 'exit'."""
    with patch.object(builtins, "input", side_effect=["exit"]):
        main()  # must return without raising


def test_main_exits_on_quit_command():
    """`main` must also terminate when the user types 'quit'."""
    with patch.object(builtins, "input", side_effect=["quit"]):
        main()


def test_main_prints_fibonacci_for_valid_index(capsys):
    """A numeric input must print the corresponding Fibonacci prefix."""
    with patch.object(builtins, "input", side_effect=["5", "exit"]):
        main()
    out = capsys.readouterr().out
    assert "[0, 1, 1, 2, 3]" in out


def test_main_handles_invalid_input_then_exits(capsys):
    """Non-numeric input must print an error message and keep looping."""
    with patch.object(builtins, "input", side_effect=["abc", "exit"]):
        main()
    out = capsys.readouterr().out
    assert "Enter a number or 'exit'" in out


def test_main_handles_multiple_queries(capsys):
    """Several numeric queries in one session must all be answered."""
    with patch.object(builtins, "input", side_effect=["3", "6", "exit"]):
        main()
    out = capsys.readouterr().out
    assert "[0, 1, 1]" in out
    assert "[0, 1, 1, 2, 3, 5]" in out


def test_main_prints_banner(capsys):
    """The introductory banner must be printed before prompting."""
    with patch.object(builtins, "input", side_effect=["exit"]):
        main()
    out = capsys.readouterr().out
    assert "Fibonacci Series Using Dynamic Programming" in out


def test_main_propagates_keyboard_interrupt():
    """Ctrl-C (KeyboardInterrupt) must not be swallowed by `main`."""
    with patch.object(builtins, "input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            main()
