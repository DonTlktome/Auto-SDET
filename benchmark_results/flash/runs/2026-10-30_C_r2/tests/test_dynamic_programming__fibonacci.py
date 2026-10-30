"""Tests for the dynamic programming Fibonacci implementation."""

import builtins
from unittest.mock import patch

import pytest

from dynamic_programming.fibonacci import Fibonacci, main


class TestFibonacciInit:
    """Tests for the Fibonacci constructor."""

    def test_initial_sequence(self):
        """A fresh instance starts with the seed sequence [0, 1]."""
        assert Fibonacci().sequence == [0, 1]

    def test_instances_do_not_share_state(self):
        """Each instance owns its own sequence list (no shared mutable default)."""
        first = Fibonacci()
        second = Fibonacci()
        first.get(10)
        assert second.sequence == [0, 1]
        assert first.sequence is not second.sequence


class TestFibonacciGet:
    """Tests for Fibonacci.get.

    Note on caching semantics: the implementation computes
    ``difference = index - (len(self.sequence) - 2)`` and appends
    ``difference`` new values, so after ``get(index)`` the cached
    ``sequence`` holds ``index + 2`` entries (for ``index >= 2``) while the
    returned value is the slice ``sequence[:index]``.
    """

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0, []),
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 1]),
            (5, [0, 1, 1, 2, 3]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ],
    )
    def test_get_returns_expected_prefix(self, index, expected):
        """get(index) returns the first `index` Fibonacci numbers."""
        assert Fibonacci().get(index) == expected

    def test_get_zero_returns_empty_list(self):
        """Index 0 is the lower boundary and yields an empty list."""
        assert Fibonacci().get(0) == []

    def test_get_negative_index_uses_python_slicing(self):
        """Negative indices are handed straight to list slicing (no raise).

        ``get(-1)`` returns ``sequence[:-1]`` of the seed list, i.e. ``[0]``,
        and a large negative index simply yields the empty list.
        """
        assert Fibonacci().get(-1) == [0]
        assert Fibonacci().get(-5) == []

    def test_get_does_not_mutate_returned_list(self):
        """Mutating the returned list must not corrupt internal state."""
        fibonacci = Fibonacci()
        result = fibonacci.get(5)
        result.append(999)
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8]
        assert fibonacci.get(5) == [0, 1, 1, 2, 3]

    def test_get_is_idempotent(self):
        """Repeated calls with the same index return equal results."""
        fibonacci = Fibonacci()
        assert fibonacci.get(8) == fibonacci.get(8)

    def test_get_extends_sequence_incrementally(self):
        """Calling get with a larger index extends the cached sequence."""
        fibonacci = Fibonacci()
        fibonacci.get(5)
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8]
        fibonacci.get(8)
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_get_smaller_index_after_larger_does_not_shrink(self):
        """A smaller index after a larger one only slices the cached sequence."""
        fibonacci = Fibonacci()
        fibonacci.get(10)
        assert fibonacci.get(3) == [0, 1, 1]
        assert len(fibonacci.sequence) == 12

    def test_get_large_index_matches_recurrence(self):
        """A larger index satisfies the Fibonacci recurrence."""
        sequence = Fibonacci().get(50)
        assert len(sequence) == 50
        assert sequence[:2] == [0, 1]
        for i in range(2, 50):
            assert sequence[i] == sequence[i - 1] + sequence[i - 2]

    def test_get_accepts_bool_as_int(self):
        """Booleans are ints in Python, so True behaves like index 1."""
        assert Fibonacci().get(True) == [0]

    def test_get_with_non_int_raises_type_error(self):
        """A non-integer index raises TypeError from the arithmetic."""
        with pytest.raises(TypeError):
            Fibonacci().get("5")


class TestMain:
    """Tests for the interactive main() entry point."""

    def test_main_exits_on_exit_command(self):
        """Entering 'exit' terminates the loop immediately."""
        with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        assert mock_print.called

    def test_main_exits_on_quit_command(self):
        """Entering 'quit' also terminates the loop."""
        with patch.object(builtins, "input", side_effect=["quit"]), patch.object(
            builtins, "print"
        ):
            main()

    def test_main_prints_fibonacci_for_valid_index(self):
        """A numeric input prints the corresponding Fibonacci prefix."""
        with patch.object(builtins, "input", side_effect=["5", "exit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        mock_print.assert_any_call([0, 1, 1, 2, 3])

    def test_main_handles_invalid_input(self):
        """Non-numeric input prints an error message and continues."""
        with patch.object(
            builtins, "input", side_effect=["abc", "exit"]
        ), patch.object(builtins, "print") as mock_print:
            main()
        mock_print.assert_any_call("Enter a number or 'exit'")

    def test_main_handles_eof(self):
        """EOFError from input propagates out of main (Ctrl-D behaviour)."""
        with patch.object(builtins, "input", side_effect=EOFError), patch.object(
            builtins, "print"
        ):
            with pytest.raises(EOFError):
                main()

    def test_main_handles_keyboard_interrupt(self):
        """KeyboardInterrupt from input propagates out of main (Ctrl-C behaviour)."""
        with patch.object(
            builtins, "input", side_effect=KeyboardInterrupt
        ), patch.object(builtins, "print"):
            with pytest.raises(KeyboardInterrupt):
                main()

    def test_main_reuses_single_fibonacci_instance(self):
        """Multiple queries accumulate in one instance's cached sequence."""
        with patch.object(
            builtins, "input", side_effect=["3", "6", "exit"]
        ), patch.object(builtins, "print") as mock_print:
            main()
        mock_print.assert_any_call([0, 1, 1])
        mock_print.assert_any_call([0, 1, 1, 2, 3, 5])
