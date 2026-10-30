"""Tests for the dynamic-programming Fibonacci implementation."""

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

    def test_get_zero_returns_empty_list(self):
        """Index 0 is the lower boundary and yields an empty list."""
        assert Fibonacci().get(0) == []

    def test_get_negative_index_uses_slice_semantics(self):
        """Negative indices follow Python slice semantics without raising."""
        assert Fibonacci().get(-1) == [0]
        assert Fibonacci().get(-2) == []
        assert Fibonacci().get(-100) == []

    def test_get_does_not_mutate_returned_list(self):
        """Mutating the returned list must not corrupt internal state."""
        fibonacci = Fibonacci()
        result = fibonacci.get(5)
        result.append(999)
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8]
        assert fibonacci.get(5) == [0, 1, 1, 2, 3]

    def test_get_is_cached_across_calls(self):
        """Repeated calls reuse the cached sequence instead of recomputing."""
        fibonacci = Fibonacci()
        expected = fibonacci.get(10)
        cached = list(fibonacci.sequence)
        assert fibonacci.get(10) == expected
        assert fibonacci.sequence == cached

    def test_get_smaller_index_after_larger(self):
        """A smaller index after a larger one returns a prefix of the cache."""
        fibonacci = Fibonacci()
        fibonacci.get(10)
        assert fibonacci.get(3) == [0, 1, 1]

    def test_get_extends_sequence_incrementally(self):
        """Calling with increasing indices grows the cache correctly."""
        fibonacci = Fibonacci()
        assert fibonacci.get(5) == [0, 1, 1, 2, 3]
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8]
        assert fibonacci.get(8) == [0, 1, 1, 2, 3, 5, 8, 13]
        assert fibonacci.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_get_large_index_matches_known_value(self):
        """A larger index produces the correct Fibonacci value at the end."""
        result = Fibonacci().get(50)
        assert len(result) == 50
        assert result[-1] == 7778742049
        assert result[-2] == 4807526976

    def test_get_returns_list_type(self):
        """The return value is always a list."""
        assert isinstance(Fibonacci().get(7), list)


class TestMain:
    """Tests for the interactive main() entry point."""

    def test_main_exits_on_exit_command(self, capsys):
        """Entering 'exit' terminates the loop immediately."""
        with patch.object(builtins, "input", return_value="exit"):
            main()
        out = capsys.readouterr().out
        assert "Fibonacci Series Using Dynamic Programming" in out

    def test_main_exits_on_quit_command(self, capsys):
        """Entering 'quit' also terminates the loop."""
        with patch.object(builtins, "input", return_value="quit"):
            main()
        assert "Fibonacci Series Using Dynamic Programming" in capsys.readouterr().out

    def test_main_prints_fibonacci_for_valid_index(self, capsys):
        """A numeric input prints the corresponding Fibonacci prefix."""
        inputs = iter(["5", "exit"])
        with patch.object(builtins, "input", lambda *_: next(inputs)):
            main()
        out = capsys.readouterr().out
        assert "[0, 1, 1, 2, 3]" in out

    def test_main_handles_invalid_input(self, capsys):
        """Non-numeric input prints an error message and continues."""
        inputs = iter(["abc", "exit"])
        with patch.object(builtins, "input", lambda *_: next(inputs)):
            main()
        out = capsys.readouterr().out
        assert "Enter a number or 'exit'" in out

    def test_main_handles_multiple_queries(self, capsys):
        """Several queries in one session are all answered."""
        inputs = iter(["3", "6", "exit"])
        with patch.object(builtins, "input", lambda *_: next(inputs)):
            main()
        out = capsys.readouterr().out
        assert "[0, 1, 1]" in out
        assert "[0, 1, 1, 2, 3, 5]" in out

    def test_main_handles_keyboard_interrupt(self):
        """Ctrl-C during input propagates out of main (not swallowed)."""
        with patch.object(builtins, "input", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                main()
