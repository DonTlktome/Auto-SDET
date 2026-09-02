"""
Pytest test suite for the calculator module.

Covers:
- All pure functions: add, subtract, multiply, divide, sqrt
- The stateful Calculator class (compute, get_history, clear_history)
- Happy-path, edge-case, and boundary tests
- Error-propagation tests (ValueError on division by zero, negative sqrt, unknown op)
- Mocking of math.sqrt to demonstrate external-dependency isolation
"""

import math
from unittest.mock import patch

import pytest

from calculator import (
    Calculator,
    add,
    divide,
    multiply,
    sqrt,
    subtract,
)


# ============================================================================
# add
# ============================================================================
class TestAdd:
    """Tests for the add() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (2, 3, 5),           # positive integers
            (-1, 1, 0),          # mixed sign
            (0, 0, 0),           # zeros
            (-5, -7, -12),       # both negative
            (1.5, 2.5, 4.0),     # floats
            (1e10, 1e10, 2e10),  # large numbers
        ],
    )
    def test_add_happy_path(self, a, b, expected):
        """add() should correctly sum integers and floats, including negatives and zero."""
        assert add(a, b) == expected

    def test_add_commutative(self):
        """add() should be commutative: a + b == b + a."""
        assert add(7, 3) == add(3, 7)


# ============================================================================
# subtract
# ============================================================================
class TestSubtract:
    """Tests for the subtract() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (10, 3, 7),          # positive
            (0, 5, -5),          # zero minus positive
            (-3, -3, 0),         # negatives cancel
            (2.5, 1.0, 1.5),     # floats
            (0, 0, 0),           # zeros
        ],
    )
    def test_subtract_happy_path(self, a, b, expected):
        """subtract() should return a - b for various inputs."""
        assert subtract(a, b) == expected

    def test_subtract_not_commutative(self):
        """subtract() should NOT be commutative in general."""
        assert subtract(10, 3) != subtract(3, 10)


# ============================================================================
# multiply
# ============================================================================
class TestMultiply:
    """Tests for the multiply() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (3, 4, 12),          # positive integers
            (-2, 5, -10),        # mixed sign
            (0, 100, 0),         # zero property
            (-3, -3, 9),         # both negative
            (1.5, 2, 3.0),       # float × int
            (0, 0, 0),           # zero × zero
        ],
    )
    def test_multiply_happy_path(self, a, b, expected):
        """multiply() should return the product of two numbers."""
        assert multiply(a, b) == expected

    def test_multiply_commutative(self):
        """multiply() should be commutative: a * b == b * a."""
        assert multiply(6, 7) == multiply(7, 6)


# ============================================================================
# divide
# ============================================================================
class TestDivide:
    """Tests for the divide() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (10, 2, 5),          # exact integer division
            (7, 2, 3.5),         # non-exact → float
            (-9, 3, -3),         # negative numerator
            (0, 5, 0),           # zero numerator
            (1, 3, 1 / 3),       # repeating decimal
        ],
    )
    def test_divide_happy_path(self, a, b, expected):
        """divide() should return a / b for valid non-zero divisors."""
        assert divide(a, b) == expected

    @pytest.mark.parametrize("a", [10, 0, -5, 3.14])
    def test_divide_by_zero_raises_value_error(self, a):
        """divide() should raise ValueError when divisor is zero, regardless of numerator."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(a, 0)

    def test_divide_by_zero_error_message(self):
        """The ValueError raised by divide() should contain the expected message."""
        with pytest.raises(ValueError) as exc_info:
            divide(42, 0)
        assert str(exc_info.value) == "Cannot divide by zero"


# ============================================================================
# sqrt
# ============================================================================
class TestSqrt:
    """Tests for the sqrt() function."""

    @pytest.mark.parametrize(
        "x, expected",
        [
            (4, 2.0),
            (9, 3.0),
            (0, 0.0),            # boundary: sqrt(0) == 0
            (2.25, 1.5),
            (1, 1.0),            # boundary: sqrt(1) == 1
        ],
    )
    def test_sqrt_happy_path(self, x, expected):
        """sqrt() should return the square root for non-negative inputs."""
        assert sqrt(x) == expected

    @pytest.mark.parametrize("x", [-1, -0.001, -100])
    def test_sqrt_negative_raises_value_error(self, x):
        """sqrt() should raise ValueError for any negative input."""
        with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
            sqrt(x)

    def test_sqrt_calls_math_sqrt(self):
        """sqrt() should delegate to math.sqrt for the actual computation."""
        with patch("calculator.math.sqrt", return_value=42.0) as mock_sqrt:
            result = sqrt(100)
            mock_sqrt.assert_called_once_with(100)
            assert result == 42.0

    def test_sqrt_does_not_call_math_sqrt_when_negative(self):
        """sqrt() should NOT call math.sqrt when the input is negative."""
        with patch("calculator.math.sqrt") as mock_sqrt:
            with pytest.raises(ValueError):
                sqrt(-9)
            mock_sqrt.assert_not_called()


# ============================================================================
# Calculator class
# ============================================================================
class TestCalculator:
    """Tests for the stateful Calculator class."""

    # -- __init__ / get_history --------------------------------------------
    def test_init_empty_history(self):
        """A newly created Calculator should have an empty history."""
        calc = Calculator()
        assert calc.get_history() == []

    def test_init_history_is_list(self):
        """get_history() should return a list (empty initially)."""
        calc = Calculator()
        assert isinstance(calc.get_history(), list)

    # -- compute -----------------------------------------------------------
    @pytest.mark.parametrize(
        "operation, a, b, expected",
        [
            ("add", 3, 4, 7),
            ("subtract", 10, 3, 7),
            ("multiply", 2, 5, 10),
            ("divide", 8, 2, 4),
            ("sqrt", 16, 0, 4),  # b is unused for sqrt
        ],
    )
    def test_compute_valid_operations(self, operation, a, b, expected):
        """compute() should return correct results for all supported operations."""
        calc = Calculator()
        result = calc.compute(operation, a, b)
        assert result == expected

    def test_compute_unknown_operation_raises_value_error(self):
        """compute() should raise ValueError for an unsupported operation string."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Unknown operation: power"):
            calc.compute("power", 2, 3)

    def test_compute_unknown_operation_empty_string(self):
        """compute() should raise ValueError for an empty operation string."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Unknown operation:"):
            calc.compute("", 1, 2)

    def test_compute_records_history(self):
        """compute() should append an entry to history after each operation."""
        calc = Calculator()
        calc.compute("add", 1, 2)
        calc.compute("multiply", 3, 4)

        history = calc.get_history()
        assert len(history) == 2
        assert history[0] == "add(1, 2) = 3"
        assert history[1] == "multiply(3, 4) = 12"

    def test_compute_propagates_divide_by_zero(self):
        """compute('divide', ...) should propagate ValueError on division by zero."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.compute("divide", 5, 0)

    def test_compute_propagates_sqrt_negative(self):
        """compute('sqrt', ...) should propagate ValueError for negative input."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
            calc.compute("sqrt", -9, 0)

    def test_compute_does_not_record_history_on_error(self):
        """compute() should NOT append to history when the operation raises."""
        calc = Calculator()
        with pytest.raises(ValueError):
            calc.compute("divide", 1, 0)
        assert calc.get_history() == []

    # -- get_history -------------------------------------------------------
    def test_get_history_returns_copy(self):
        """get_history() should return a copy, not the internal list reference."""
        calc = Calculator()
        calc.compute("add", 1, 1)
        hist = calc.get_history()
        hist.append("tampered")
        assert calc.get_history() == ["add(1, 1) = 2"]

    def test_get_history_preserves_order(self):
        """get_history() should return entries in insertion order."""
        calc = Calculator()
        ops = [("add", 1, 1), ("subtract", 5, 2), ("multiply", 3, 3)]
        for op, a, b in ops:
            calc.compute(op, a, b)
        history = calc.get_history()
        assert len(history) == 3
        assert "add" in history[0]
        assert "subtract" in history[1]
        assert "multiply" in history[2]

    # -- clear_history -----------------------------------------------------
    def test_clear_history_empties_history(self):
        """clear_history() should remove all entries."""
        calc = Calculator()
        calc.compute("add", 1, 1)
        calc.clear_history()
        assert calc.get_history() == []

    def test_clear_history_on_empty_is_idempotent(self):
        """clear_history() on an already-empty history should be safe (idempotent)."""
        calc = Calculator()
        calc.clear_history()  # first clear
        calc.clear_history()  # second clear — should not raise
        assert calc.get_history() == []

    # -- integration-style: full lifecycle ---------------------------------
    def test_full_lifecycle(self):
        """End-to-end: compute multiple operations, check history, clear, verify empty."""
        calc = Calculator()
        calc.compute("add", 10, 5)
        calc.compute("subtract", 10, 5)
        calc.compute("multiply", 10, 5)
        calc.compute("divide", 10, 5)
        calc.compute("sqrt", 25, 0)

        assert len(calc.get_history()) == 5

        calc.clear_history()
        assert calc.get_history() == []

    def test_multiple_calculator_instances_are_independent(self):
        """Two Calculator instances should have independent histories."""
        calc1 = Calculator()
        calc2 = Calculator()

        calc1.compute("add", 1, 2)
        calc2.compute("multiply", 3, 4)

        assert calc1.get_history() == ["add(1, 2) = 3"]
        assert calc2.get_history() == ["multiply(3, 4) = 12"]