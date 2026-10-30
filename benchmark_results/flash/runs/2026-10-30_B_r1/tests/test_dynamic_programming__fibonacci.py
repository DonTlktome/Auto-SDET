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
        """Each instance owns its own sequence list (no shared mutable default)."""
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

    def test_get_zero_returns_empty_list(self):
        """Index 0 is the lower boundary and yields an empty list."""
        fib = Fibonacci()
        assert fib.get(0) == []
        assert fib.sequence == [0, 1]

    def test_get_negative_index_slices_from_end(self):
        """A negative index slices the seed sequence without raising."""
        fib = Fibonacci()
        assert fib.get(-1) == [0]
        assert fib.get(-2) == []
        assert fib.sequence == [0, 1]

    def test_get_does_not_mutate_returned_list(self):
        """Mutating the returned list must not corrupt internal state."""
        fib = Fibonacci()
        result = fib.get(5)
        result.append(999)
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
        assert fib.get(5) == [0, 1, 1, 2, 3]

    def test_get_is_idempotent(self):
        """Repeated calls with the same index return equal results."""
        fib = Fibonacci()
        assert fib.get(8) == fib.get(8)

    def test_get_extends_sequence_incrementally(self):
        """Calling get with a larger index extends the cached sequence."""
        fib = Fibonacci()
        fib.get(5)
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
        fib.get(8)
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_get_smaller_index_after_larger_does_not_shrink(self):
        """Requesting a smaller index later returns a prefix of the cache."""
        fib = Fibonacci()
        fib.get(10)
        cached = list(fib.sequence)
        assert fib.get(3) == [0, 1, 1]
        assert fib.sequence == cached

    def test_get_large_index_matches_known_value(self):
        """A larger index produces the correct Fibonacci value at the end."""
        fib = Fibonacci()
        result = fib.get(30)
        assert len(result) == 30
        assert result[-1] == 514229
        assert result[-2] == 317811

    def test_get_returns_list_type(self):
        """The return value is always a list."""
        assert isinstance(Fibonacci().get(7), list)


class TestMain:
    """Tests for the interactive main() entry point."""

    def test_main_exits_on_exit_command(self):
        """Entering 'exit' terminates the loop immediately."""
        with patch.object(builtins, "input", return_value="exit") as mock_input:
            main()
        assert mock_input.call_count == 1

    def test_main_exits_on_quit_command(self):
        """Entering 'quit' also terminates the loop."""
        with patch.object(builtins, "input", return_value="quit") as mock_input:
            main()
        assert mock_input.call_count == 1

    def test_main_prints_fibonacci_for_valid_index(self, capsys):
        """A numeric input prints the corresponding Fibonacci prefix."""
        with patch.object(builtins, "input", side_effect=["5", "exit"]):
            main()
        out = capsys.readouterr().out
        assert "[0, 1, 1, 2, 3]" in out

    def test_main_handles_invalid_input(self, capsys):
        """Non-numeric input prints an error message and continues."""
        with patch.object(builtins, "input", side_effect=["abc", "exit"]):
            main()
        out = capsys.readouterr().out
        assert "Enter a number or 'exit'" in out

    def test_main_prints_banner(self, capsys):
        """The introductory banner is printed before prompting."""
        with patch.object(builtins, "input", return_value="exit"):
            main()
        out = capsys.readouterr().out
        assert "Fibonacci Series Using Dynamic Programming" in out

    def test_main_handles_multiple_inputs(self, capsys):
        """Several indices can be processed in one session."""
        with patch.object(builtins, "input", side_effect=["3", "6", "exit"]):
            main()
        out = capsys.readouterr().out
        assert "[0, 1, 1]" in out
        assert "[0, 1, 1, 2, 3, 5]" in out

    def test_main_propagates_keyboard_interrupt(self):
        """Ctrl-C (KeyboardInterrupt) is not swallowed by main()."""
        with patch.object(builtins, "input", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                main()
