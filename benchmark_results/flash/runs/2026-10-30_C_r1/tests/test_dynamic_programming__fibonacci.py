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
        # The internal cache must not be mutated by a zero-length request.
        assert fib.sequence == [0, 1]

    def test_get_negative_index_returns_empty_list(self):
        """Negative indices slice to an empty list and leave the cache intact."""
        fib = Fibonacci()
        assert fib.get(-5) == []
        assert fib.sequence == [0, 1]

    def test_get_does_not_mutate_returned_list(self):
        """Mutating the returned list must not corrupt the internal cache."""
        fib = Fibonacci()
        result = fib.get(5)
        result.append(999)
        assert fib.get(5) == [0, 1, 1, 2, 3]

    def test_get_is_idempotent(self):
        """Repeated calls with the same index return equal results."""
        fib = Fibonacci()
        assert fib.get(8) == fib.get(8)

    def test_get_extends_cache_incrementally(self):
        """Calling get with a larger index extends the cached sequence."""
        fib = Fibonacci()
        fib.get(5)
        # The cache is grown to index + 2 entries, one past the returned slice.
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8]
        fib.get(8)
        assert fib.sequence == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_get_smaller_index_after_large_does_not_shrink_cache(self):
        """A smaller subsequent request slices the cache without shrinking it."""
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
        """Typing 'exit' terminates the loop immediately."""
        with patch.object(builtins, "input", side_effect=["exit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        # Only the banner is printed; no sequence output.
        assert mock_print.call_count == 1

    def test_main_exits_on_quit_command(self):
        """Typing 'quit' also terminates the loop."""
        with patch.object(builtins, "input", side_effect=["quit"]), patch.object(
            builtins, "print"
        ) as mock_print:
            main()
        assert mock_print.call_count == 1

    def test_main_prints_sequence_for_valid_index(self):
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

    def test_main_reuses_single_fibonacci_instance(self):
        """Multiple queries accumulate in one instance's cache."""
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
