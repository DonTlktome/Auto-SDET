import pytest

from fibonacci import Fibonacci, main


class TestFibonacciGet:
    """Tests for the ``Fibonacci.get`` method."""

    def test_get_initial_sequence(self):
        """A fresh instance returns the base sequence [0, 1] for index 2."""
        fib = Fibonacci()
        assert fib.get(2) == [0, 1]

    def test_get_zero_returns_empty_list(self):
        """Requesting index 0 returns an empty list."""
        fib = Fibonacci()
        assert fib.get(0) == []

    def test_get_one_returns_single_element(self):
        """Requesting index 1 returns only the first element."""
        fib = Fibonacci()
        assert fib.get(1) == [0]

    @pytest.mark.parametrize(
        "index, expected",
        [
            (5, [0, 1, 1, 2, 3]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
            (3, [0, 1, 1]),
            (4, [0, 1, 1, 2]),
        ],
    )
    def test_get_known_sequences(self, index, expected):
        """Known Fibonacci sequences are returned correctly for various indices."""
        fib = Fibonacci()
        assert fib.get(index) == expected

    def test_get_larger_index_computes_missing_values(self):
        """Requesting a larger index computes all missing values up to it."""
        fib = Fibonacci()
        result = fib.get(15)
        assert len(result) == 15
        assert result[-1] == 377  # F(14)
        assert result[-2] == 233  # F(13)

    def test_get_is_cumulative_across_calls(self):
        """Subsequent calls reuse previously computed values (memoization)."""
        fib = Fibonacci()
        fib.get(10)
        # The internal sequence should now be longer than the base [0, 1].
        assert len(fib.sequence) >= 10
        # A smaller subsequent call should not recompute or truncate storage.
        assert fib.get(5) == [0, 1, 1, 2, 3]
        assert len(fib.sequence) >= 10

    def test_get_does_not_mutate_returned_list(self):
        """The returned list is a slice, so mutating it does not affect state."""
        fib = Fibonacci()
        result = fib.get(5)
        result.append(999)
        assert fib.get(5) == [0, 1, 1, 2, 3]

    def test_get_negative_index_returns_first_element(self):
        """A negative index slices off only the last element, returning [0]."""
        fib = Fibonacci()
        assert fib.get(-1) == [0]


class TestFibonacciMain:
    """Tests for the ``main`` function, isolating I/O via mocks."""

    def test_main_exits_on_quit(self, monkeypatch, capsys):
        """Entering 'quit' immediately terminates the loop."""
        inputs = iter(["quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        main()
        captured = capsys.readouterr()
        assert "Fibonacci Series Using Dynamic Programming" in captured.out

    def test_main_exits_on_exit(self, monkeypatch):
        """Entering 'exit' immediately terminates the loop."""
        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        main()

    def test_main_prints_fibonacci_for_valid_input(self, monkeypatch, capsys):
        """A valid numeric input prints the corresponding Fibonacci sequence."""
        inputs = iter(["10", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        main()
        captured = capsys.readouterr()
        assert "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]" in captured.out

    def test_main_handles_invalid_input(self, monkeypatch, capsys):
        """Non-numeric input prints an error message and continues the loop."""
        inputs = iter(["not_a_number", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        main()
        captured = capsys.readouterr()
        assert "Enter a number or 'exit'" in captured.out

    def test_main_handles_invalid_then_valid_input(self, monkeypatch, capsys):
        """After an invalid input, a subsequent valid input is still processed."""
        inputs = iter(["abc", "5", "quit"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        main()
        captured = capsys.readouterr()
        assert "Enter a number or 'exit'" in captured.out
        assert "[0, 1, 1, 2, 3]" in captured.out
