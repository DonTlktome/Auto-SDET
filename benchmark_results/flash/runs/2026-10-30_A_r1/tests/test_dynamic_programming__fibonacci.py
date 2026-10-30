"""Tests for the dynamic-programming Fibonacci implementation."""

import builtins
from unittest.mock import patch

import pytest

from dynamic_programming.fibonacci import Fibonacci, main


class TestFibonacciInit:
    """Tests for the Fibonacci constructor."""

    def test_initial_sequence(self):
        """A fresh instance starts with the seed sequence [0, 1]."""
        fib = Fibonacci()
        assert fib.sequence == [0, 1]

    def test_instances_do_not_share_state(self):
        """Each instance owns its own sequence list (no shared mutable state)."""
        first = Fibonacci()
        second = Fibonacci()
        first.get(10)
        assert second.sequence == [0, 1]
        assert first.sequence is not second.sequence


class TestFibonacciGet:
    """Tests for Fibonacci.get."""

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

    def test_get_does_not_mutate_returned_list(self):
        """Mutating the returned list must not corrupt internal state."""
        fib = Fibonacci()
        result = fib.get(5)
        result.append(999)
        # get() returns a fresh slice; the internal cache is untouched by
        # mutating the returned list and holds index + 2 values.
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
        assert fib.get(5) == [0, 1, 1, 2, 3]

    def test_get_is_idempotent(self):
        """Calling get twice with the same index yields the same result."""
        fib = Fibonacci()
        assert fib.get(8) == fib.get(8)

    def test_get_extends_sequence_incrementally(self):
        """A larger index after a smaller one extends rather than recomputes."""
        fib = Fibonacci()
        fib.get(5)
        # The cache is grown to index + 2 terms.
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
        fib.get(10)
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    def test_get_smaller_index_after_larger(self):
        """Requesting a smaller index after a larger one returns a prefix."""
        fib = Fibonacci()
        fib.get(10)
        assert fib.get(3) == [0, 1, 1]
        # Internal sequence is not truncated (index + 2 cached terms).
        assert len(fib.sequence) == 12

    def test_get_negative_index(self):
        """A negative index slices the cache from the end."""
        fib = Fibonacci()
        # No growth happens for negative indexes; slicing [:-1] drops the last
        # seed element, and more negative indexes yield an empty list.
        assert fib.get(-1) == [0]
        assert fib.get(-2) == []
        assert Fibonacci().get(-3) == []

    def test_get_large_index_matches_recurrence(self):
        """A large index satisfies the Fibonacci recurrence."""
        fib = Fibonacci()
        result = fib.get(50)
        assert len(result) == 50
        assert result[0] == 0
        assert result[1] == 1
        for i in range(2, 50):
            assert result[i] == result[i - 1] + result[i - 2]

    def test_get_returns_list_type(self):
        """get always returns a list."""
        assert isinstance(Fibonacci().get(7), list)


class TestMain:
    """Tests for the interactive main() loop."""

    def test_main_exits_on_exit_keyword(self):
        """Typing 'exit' terminates the loop without printing a sequence."""
        with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        # Only the banner is printed.
        assert mock_print.call_count == 1

    def test_main_exits_on_quit_keyword(self):
        """Typing 'quit' also terminates the loop."""
        with patch.object(builtins, "input", side_effect=["quit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        assert mock_print.call_count == 1

    def test_main_prints_fibonacci_for_valid_index(self):
        """A numeric input prints the corresponding Fibonacci prefix."""
        with patch.object(builtins, "input", side_effect=["5", "exit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        printed = [call.args[0] for call in mock_print.call_args_list if call.args]
        assert [0, 1, 1, 2, 3] in printed

    def test_main_handles_invalid_input(self):
        """Non-numeric input prints an error message and continues."""
        with patch.object(
            builtins, "input", side_effect=["abc", "exit"]
        ), patch.object(builtins, "print") as mock_print:
            main()
        printed = [call.args[0] for call in mock_print.call_args_list if call.args]
        assert "Enter a number or 'exit'" in printed

    def test_main_handles_multiple_queries(self):
        """Several queries in one session are all processed."""
        with patch.object(
            builtins, "input", side_effect=["3", "6", "exit"]
        ), patch.object(builtins, "print") as mock_print:
            main()
        printed = [call.args[0] for call in mock_print.call_args_list if call.args]
        assert [0, 1, 1] in printed
        assert [0, 1, 1, 2, 3, 5] in printed

    def test_main_propagates_keyboard_interrupt(self):
        """Ctrl-C (KeyboardInterrupt) is not swallowed by main()."""
        with patch.object(builtins, "input", side_effect=KeyboardInterrupt), patch.object(
            builtins, "print"
        ):
            with pytest.raises(KeyboardInterrupt):
                main()
