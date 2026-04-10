"""
A simple calculator module — used as demo target for auto-sdet.

This file deliberately includes:
- Pure functions (easy to test)
- Edge cases (division by zero)
- A class with state (Calculator)
- An external dependency call (for mock demonstration)
"""
import math


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError on division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def sqrt(x: float) -> float:
    """Return the square root of x. Raises ValueError for negative input."""
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    return math.sqrt(x)


class Calculator:
    """A stateful calculator that tracks history."""

    def __init__(self):
        self.history: list[str] = []

    def compute(self, operation: str, a: float, b: float = 0) -> float:
        """
        Perform a computation and record it in history.

        Args:
            operation: One of 'add', 'subtract', 'multiply', 'divide', 'sqrt'
            a: First operand
            b: Second operand (unused for sqrt)

        Returns:
            The computed result
        """
        ops = {
            "add": lambda: add(a, b),
            "subtract": lambda: subtract(a, b),
            "multiply": lambda: multiply(a, b),
            "divide": lambda: divide(a, b),
            "sqrt": lambda: sqrt(a),
        }

        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation}")

        result = ops[operation]()
        self.history.append(f"{operation}({a}, {b}) = {result}")
        return result

    def get_history(self) -> list[str]:
        """Return computation history."""
        return list(self.history)

    def clear_history(self) -> None:
        """Clear computation history."""
        self.history.clear()
