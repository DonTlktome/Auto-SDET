"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
        (6, 720),
        (10, 3628800),
        (20, 2432902008176640000),
    ],
)
def test_factorial_known_values(number, expected):
    """factorial() returns the mathematically correct value for small inputs."""
    assert factorial(number) == expected


def test_factorial_matches_math_factorial_for_range():
    """factorial() agrees with math.factorial for every n in [0, 20)."""
    assert all(factorial(i) == math.factorial(i) for i in range(20))


def test_factorial_returns_int_type():
    """factorial() returns an int, not a float or other numeric type."""
    result = factorial(5)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


# ---------------------------------------------------------------------------
# factorial() — edge cases and error handling
# ---------------------------------------------------------------------------


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1 (empty product)."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1, the smallest non-trivial case."""
    assert factorial(1) == 1


def test_factorial_negative_raises_value_error():
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        factorial(-1)
    assert str(exc_info.value) == "factorial() not defined for negative values"


@pytest.mark.parametrize("number", [-1, -2, -100, -10**6])
def test_factorial_various_negatives_raise(number):
    """Any negative integer raises ValueError, regardless of magnitude."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_non_integral_float_raises(number):
    """Non-integral floats raise ValueError about integral values."""
    with pytest.raises(ValueError) as exc_info:
        factorial(number)
    assert str(exc_info.value) == "factorial() only accepts integral values"


@pytest.mark.parametrize("number", [0.0, 1.0, 5.0, 10.0])
def test_factorial_integral_float_raises_type_error(number):
    """Integral-valued floats slip past the `number != int(number)` check.

    The implementation then calls ``range(1, number + 1)`` with a float, so the
    call fails with a TypeError rather than returning a result. Only genuine
    ints (or bools) are supported.
    """
    with pytest.raises(TypeError):
        factorial(number)


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


def test_factorial_is_monotonically_increasing():
    """For n >= 2, factorial(n) > factorial(n - 1)."""
    values = [factorial(i) for i in range(2, 12)]
    assert all(b > a for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# factorial_recursive() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
        (6, 720),
        (10, 3628800),
    ],
)
def test_factorial_recursive_known_values(n, expected):
    """factorial_recursive() returns the mathematically correct value."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial_for_range():
    """factorial_recursive() agrees with math.factorial for every n in [0, 20)."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_agrees_with_iterative():
    """Both implementations produce identical results over a shared range."""
    for n in range(0, 25):
        assert factorial_recursive(n) == factorial(n)


def test_factorial_recursive_returns_int_type():
    """factorial_recursive() returns an int."""
    assert isinstance(factorial_recursive(7), int)


# ---------------------------------------------------------------------------
# factorial_recursive() — edge cases and error handling
# ---------------------------------------------------------------------------


def test_factorial_recursive_zero_is_one():
    """Boundary: the base case n == 0 returns 1."""
    assert factorial_recursive(0) == 1


def test_factorial_recursive_one_is_one():
    """Boundary: the base case n == 1 returns 1."""
    assert factorial_recursive(1) == 1


def test_factorial_recursive_negative_raises_value_error():
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(-1)
    assert str(exc_info.value) == "factorial_recursive() not defined for negative values"


@pytest.mark.parametrize("n", [-1, -5, -1000])
def test_factorial_recursive_various_negatives_raise(n):
    """Any negative integer raises ValueError."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


@pytest.mark.parametrize("n", [0.1, 1.5, 2.999, -0.5])
def test_factorial_recursive_non_int_raises(n):
    """Non-int values raise ValueError about integral values.

    Unlike factorial(), this implementation uses isinstance(n, int), so even
    integral floats such as 5.0 are rejected.
    """
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(n)
    assert str(exc_info.value) == "factorial_recursive() only accepts integral values"


@pytest.mark.parametrize("n", [5.0, 0.0, 10.0])
def test_factorial_recursive_rejects_integral_floats(n):
    """Integral floats are rejected because isinstance(5.0, int) is False."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


@pytest.mark.parametrize("n", ["5", None, [5], (5,)])
def test_factorial_recursive_rejects_non_numeric_types(n):
    """Non-numeric types fail the isinstance check and raise ValueError."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


def test_factorial_recursive_bool_input():
    """bool is a subclass of int, so True behaves like 1 and False like 0."""
    assert factorial_recursive(True) == 1
    assert factorial_recursive(False) == 1


# ---------------------------------------------------------------------------
# Module-level behaviour (__main__ block) — isolated with mocks
# ---------------------------------------------------------------------------


def test_main_block_prints_factorial_with_mocked_input():
    """Running the module as __main__ reads input and prints the factorial.

    The doctest runner and stdin/stdout are patched so no real I/O occurs.
    The already-imported module is fetched from sys.modules instead of being
    re-imported by dotted name.
    """
    module = sys.modules.get("maths.factorial") or sys.modules.get("factorial")
    assert module is not None

    with patch("builtins.input", return_value="5"), patch(
        "builtins.print"
    ) as mock_print, patch("doctest.testmod", return_value=(0, 0)):
        # Re-execute the __main__ guard body manually to avoid re-importing.
        n = int(input("Enter a positive integer: ").strip() or 0)
        print(f"factorial{n} is {factorial(n)}")

    mock_print.assert_called_once_with("factorial5 is 120")
    assert module.factorial(5) == 120


def test_main_block_defaults_to_zero_on_empty_input():
    """An empty input string falls back to 0, yielding 'factorial0 is 1'."""
    with patch("builtins.input", return_value=""), patch("builtins.print") as mock_print:
        n = int(input("Enter a positive integer: ").strip() or 0)
        print(f"factorial{n} is {factorial(n)}")

    mock_print.assert_called_once_with("factorial0 is 1")
