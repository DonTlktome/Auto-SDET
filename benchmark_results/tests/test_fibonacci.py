"""
Pytest test suite for the Fibonacci dynamic programming module.

Tests cover:
- Fibonacci.get() — positive, edge, boundary, and caching behaviour
- Fibonacci.__init__() — initial state
- main() — interactive loop with mocked I/O
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from dynamic_programming.fibonacci import Fibonacci, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_fib() -> Fibonacci:
    """Return a freshly constructed Fibonacci instance."""
    return Fibonacci()


# ---------------------------------------------------------------------------
# Fibonacci.get() tests
# ---------------------------------------------------------------------------

class TestFibonacciGet:
    """Tests for Fibonacci.get(index)."""

    # -- Positive / happy-path tests ----------------------------------------

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0, []),
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 1]),
            (4, [0, 1, 1, 2]),
            (5, [0, 1, 1, 2, 3]),
            (6, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]),
        ],
    )
    def test_get_known_indices(
        self, fresh_fib: Fibonacci, index: int, expected: list[int]
    ) -> None:
        """Fibonacci.get should return the correct sequence for known indices."""
        assert fresh_fib.get(index) == expected

    def test_get_large_index(self, fresh_fib: Fibonacci) -> None:
        """Fibonacci.get should handle a larger index (e.g. 20) correctly."""
        result = fresh_fib.get(20)
        assert len(result) == 20
        assert result[0] == 0
        assert result[1] == 1
        # Spot-check known values
        assert result[7] == 13   # F(7)
        assert result[19] == 4181  # F(19)

    # -- Caching / repeated-call tests -------------------------------------

    def test_get_caches_results(self, fresh_fib: Fibonacci) -> None:
        """Calling get with a larger index then a smaller one should use cache."""
        fresh_fib.get(10)
        # Internal sequence should now have at least 10 entries
        assert len(fresh_fib.sequence) >= 10
        # A second call with a smaller index must return the same slice
        # without re-computation.
        result = fresh_fib.get(5)
        assert result == [0, 1, 1, 2, 3]

    def test_get_progressive_calls(self, fresh_fib: Fibonacci) -> None:
        """Progressive calls with increasing indices extend the cache correctly."""
        assert fresh_fib.get(3) == [0, 1, 1]
        assert fresh_fib.get(5) == [0, 1, 1, 2, 3]
        assert fresh_fib.get(7) == [0, 1, 1, 2, 3, 5, 8]

    # -- Edge / boundary tests ----------------------------------------------

    def test_get_zero_returns_empty(self, fresh_fib: Fibonacci) -> None:
        """Fibonacci.get(0) should return an empty list."""
        assert fresh_fib.get(0) == []

    def test_get_one_returns_single_element(self, fresh_fib: Fibonacci) -> None:
        """Fibonacci.get(1) should return [0]."""
        assert fresh_fib.get(1) == [0]

    def test_get_negative_index(self, fresh_fib: Fibonacci) -> None:
        """Fibonacci.get with a negative index should return an empty slice
        (Python slicing behaviour with negative stop on a list)."""
        # For index = -1, difference = -1 - 0 = -1, loop is skipped,
        # and self.sequence[:-1] returns [0] (all but last).
        # This is a design edge-case; we document the actual behaviour.
        result = fresh_fib.get(-1)
        assert result == [0]

    def test_get_very_large_index(self, fresh_fib: Fibonacci) -> None:
        """Fibonacci.get should handle index=1000 without recursion issues."""
        result = fresh_fib.get(1000)
        assert len(result) == 1000
        # Verify Fibonacci property: F(n) = F(n-1) + F(n-2)
        assert result[-1] == result[-2] + result[-3]

    # -- Instance isolation -------------------------------------------------

    def test_instances_are_independent(self) -> None:
        """Each Fibonacci instance maintains its own cache."""
        fib1 = Fibonacci()
        fib2 = Fibonacci()
        fib1.get(10)
        # fib2 should still have only the initial two elements
        assert fib2.sequence == [0, 1]
        assert fib2.get(3) == [0, 1, 1]


# ---------------------------------------------------------------------------
# Fibonacci.__init__() tests
# ---------------------------------------------------------------------------

class TestFibonacciInit:
    """Tests for Fibonacci constructor."""

    def test_initial_sequence(self, fresh_fib: Fibonacci) -> None:
        """A new Fibonacci instance should have sequence [0, 1]."""
        assert fresh_fib.sequence == [0, 1]

    def test_sequence_is_list(self, fresh_fib: Fibonacci) -> None:
        """The sequence attribute should be a list."""
        assert isinstance(fresh_fib.sequence, list)


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the interactive main() function."""

    def test_main_exit_immediately(self) -> None:
        """main() should exit immediately when the user types 'exit'."""
        with patch.object(builtins, "input", return_value="exit"):
            with patch.object(builtins, "print") as mock_print:
                main()
        # The banner is printed at least once
        assert mock_print.call_count >= 1

    def test_main_quit_immediately(self) -> None:
        """main() should exit immediately when the user types 'quit'."""
        with patch.object(builtins, "input", return_value="quit"):
            with patch.object(builtins, "print") as mock_print:
                main()
        assert mock_print.call_count >= 1

    def test_main_single_valid_index(self) -> None:
        """main() should print the Fibonacci sequence for a valid index."""
        inputs = iter(["5", "exit"])
        with patch.object(builtins, "input", side_effect=lambda _="": next(inputs)):
            with patch.object(builtins, "print") as mock_print:
                main()
        # Collect all positional args from print calls
        printed = [
            str(arg) for call in mock_print.call_args_list for arg in call.args
        ]
        # One of the prints should contain the sequence [0, 1, 1, 2, 3]
        assert any("[0, 1, 1, 2, 3]" in p for p in printed)

    def test_main_invalid_input_then_exit(self) -> None:
        """main() should handle non-integer input gracefully and continue."""
        inputs = iter(["abc", "exit"])
        with patch.object(builtins, "input", side_effect=lambda _="": next(inputs)):
            with patch.object(builtins, "print") as mock_print:
                main()
        printed = [
            str(arg) for call in mock_print.call_args_list for arg in call.args
        ]
        assert any("Enter a number or 'exit'" in p for p in printed)

    def test_main_multiple_valid_indices(self) -> None:
        """main() should handle multiple valid indices before exit."""
        inputs = iter(["3", "7", "quit"])
        with patch.object(builtins, "input", side_effect=lambda _="": next(inputs)):
            with patch.object(builtins, "print") as mock_print:
                main()
        printed = [
            str(arg) for call in mock_print.call_args_list for arg in call.args
        ]
        assert any("[0, 1, 1]" in p for p in printed)
        assert any("[0, 1, 1, 2, 3, 5, 8]" in p for p in printed)

    def test_main_zero_index(self) -> None:
        """main() should print an empty list for index 0."""
        inputs = iter(["0", "exit"])
        with patch.object(builtins, "input", side_effect=lambda _="": next(inputs)):
            with patch.object(builtins, "print") as mock_print:
                main()
        printed = [
            str(arg) for call in mock_print.call_args_list for arg in call.args
        ]
        assert any("[]" in p for p in printed)
