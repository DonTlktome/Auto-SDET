"""
Pytest test suite for the Fibonacci dynamic programming implementation.

Covers the ``Fibonacci`` class (``get`` method) and the ``main`` function,
including happy paths, edge cases, boundary conditions, and isolation of
external I/O dependencies.
"""

import pytest

from fibonacci import Fibonacci, main


class TestFibonacciGet:
    """Tests for the ``Fibonacci.get`` method."""

    def test_get_initial_sequence(self):
        """A fresh instance returns the correct sequence for a small index."""
        fib = Fibonacci()
        assert fib.get(2) == [0, 1]

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, []),
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 1]),
            (5, [0, 1, 1, 2, 3]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ],
    )
    def test_get_known_values(self, index, expected):
        """``get`` returns the correct Fibonacci sequence for known indices."""
        assert Fibonacci().get(index) == expected

    def test_get_larger_index(self):
        """``get`` computes missing values for an index beyond the cache."""
        fib = Fibonacci()
        result = fib.get(15)
        assert result[-1] == 377
        assert len(result) == 15

    def test_get_reuses_cached_values(self):
        """Calling ``get`` with a smaller index after a larger one reuses cache."""
        fib = Fibonacci()
        fib.get(10)
        # The internal sequence should already contain values up to index 10.
        assert len(fib.sequence) >= 10
        assert fib.get(5) == [0, 1, 1, 2, 3]

    def test_get_zero_index(self):
        """``get(0)`` returns an empty list (edge case)."""
        assert Fibonacci().get(0) == []

    def test_get_negative_index(self):
        """``get`` with a negative index returns a truncated slice (edge case)."""
        fib = Fibonacci()
        # Negative slicing returns everything except the last ``abs(index)`` items.
        assert fib.get(-1) == [0]

    def test_get_does_not_mutate_returned_list(self):
        """The returned list is a slice, so mutating it does not affect the cache."""
        fib = Fibonacci()
        result = fib.get(5)
        result.append(999)
        assert fib.sequence[:5] == [0, 1, 1, 2, 3]


class TestFibonacciInit:
    """Tests for the ``Fibonacci.__init__`` method."""

    def test_init_sequence(self):
        """A new instance starts with the base sequence [0, 1]."""
        fib = Fibonacci()
        assert fib.sequence == [0, 1]


class TestMain:
    """Tests for the ``main`` function."""

    def test_main_exits_on_quit(self, capsys):
        """``main`` exits immediately when the user enters 'quit'."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _="": "quit")
            main()
        captured = capsys.readouterr()
        assert "Fibonacci Series Using Dynamic Programming" in captured.out

    def test_main_exits_on_exit(self, capsys):
        """``main`` exits immediately when the user enters 'exit'."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _="": "exit")
            main()
        captured = capsys.readouterr()
        assert "Fibonacci Series Using Dynamic Programming" in captured.out

    def test_main_handles_invalid_input_then_quit(self, capsys):
        """``main`` prints an error for non-numeric input, then exits on 'quit'."""
        inputs = iter(["not-a-number", "quit"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _="": next(inputs))
            main()

        captured = capsys.readouterr()
        assert "Enter a number or 'exit'" in captured.out

    def test_main_prints_fibonacci_for_valid_input(self, capsys):
        """``main`` prints the Fibonacci sequence for a valid numeric input."""
        inputs = iter(["5", "quit"])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("builtins.input", lambda _="": next(inputs))
            main()

        captured = capsys.readouterr()
        assert "[0, 1, 1, 2, 3]" in captured.out