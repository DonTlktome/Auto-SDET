"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import doctest
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial (iterative)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (5, 120),
        (6, 720),
        (10, 3628800),
        (20, 2432902008176640000),
    ],
)
def test_factorial_happy_path(number, expected):
    """factorial() returns the correct value for non-negative integers."""
    assert factorial(number) == expected


@pytest.mark.parametrize("number", list(range(0, 25)))
def test_factorial_matches_math_factorial(number):
    """factorial() agrees with math.factorial across a range of inputs."""
    assert factorial(number) == math.factorial(number)


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1 (loop body executes exactly once)."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_negative_raises_value_error(number):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        factorial(number)
    assert str(exc_info.value) == "factorial() not defined for negative values"


@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_non_integral_float_raises_value_error(number):
    """Non-integral floats raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        factorial(number)
    assert str(exc_info.value) == "factorial() only accepts integral values"


@pytest.mark.parametrize("number", [0.0, 1.0, 5.0])
def test_factorial_integral_float_raises_type_error(number):
    """Integral floats pass the `number != int(number)` check but blow up in
    `range(1, number + 1)`, which only accepts integers -> TypeError."""
    with pytest.raises(TypeError) as exc_info:
        factorial(number)
    assert "cannot be interpreted as an integer" in str(exc_info.value)


def test_factorial_negative_integral_float_raises_negative_error():
    """A negative integral float passes the integral check but fails the sign check."""
    with pytest.raises(ValueError) as exc_info:
        factorial(-3.0)
    assert str(exc_info.value) == "factorial() not defined for negative values"


def test_factorial_returns_int_type():
    """The return value is a Python int."""
    assert isinstance(factorial(7), int)


# ---------------------------------------------------------------------------
# factorial_recursive
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
def test_factorial_recursive_happy_path(n, expected):
    """factorial_recursive() returns the correct value for non-negative ints."""
    assert factorial_recursive(n) == expected


@pytest.mark.parametrize("n", list(range(0, 20)))
def test_factorial_recursive_matches_math_factorial(n):
    """factorial_recursive() agrees with math.factorial across a range of inputs."""
    assert factorial_recursive(n) == math.factorial(n)


def test_factorial_recursive_zero_and_one_base_cases():
    """Boundary: the base case `n in {0, 1}` returns 1 for both 0 and 1."""
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1


def test_factorial_recursive_two_uses_recursion():
    """Boundary: n=2 is the smallest input that actually recurses."""
    assert factorial_recursive(2) == 2


@pytest.mark.parametrize("n", [-1, -2, -50])
def test_factorial_recursive_negative_raises_value_error(n):
    """Negative inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(n)
    assert str(exc_info.value) == "factorial_recursive() not defined for negative values"


@pytest.mark.parametrize("n", [0.1, 1.5, 2.5, -0.5])
def test_factorial_recursive_non_int_raises_value_error(n):
    """Non-int values (including integral floats) raise ValueError via isinstance check."""
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(n)
    assert str(exc_info.value) == "factorial_recursive() only accepts integral values"


@pytest.mark.parametrize("n", [1.0, 5.0])
def test_factorial_recursive_rejects_integral_floats(n):
    """Unlike factorial(), factorial_recursive() rejects even integral floats."""
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(n)
    assert str(exc_info.value) == "factorial_recursive() only accepts integral values"


@pytest.mark.parametrize("n", ["5", None, [3], (3,)])
def test_factorial_recursive_rejects_non_numeric_types(n):
    """Non-numeric types fail the isinstance(n, int) guard."""
    with pytest.raises(ValueError) as exc_info:
        factorial_recursive(n)
    assert str(exc_info.value) == "factorial_recursive() only accepts integral values"


def test_factorial_recursive_bool_is_accepted_as_int():
    """bool is a subclass of int, so True/False are treated as 1/0."""
    assert factorial_recursive(True) == 1
    assert factorial_recursive(False) == 1


def test_factorial_recursive_returns_int_type():
    """The return value is a Python int."""
    assert isinstance(factorial_recursive(7), int)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", list(range(0, 30)))
def test_iterative_and_recursive_agree(n):
    """Both implementations produce identical results for the same input."""
    assert factorial(n) == factorial_recursive(n)


# ---------------------------------------------------------------------------
# __main__ block behaviour (isolated with mocks)
# ---------------------------------------------------------------------------


MAIN_BLOCK_SOURCE = (
    "if True:\n"
    "    import doctest\n"
    "    doctest.testmod()\n"
    "    n = int(input('Enter a positive integer: ').strip() or 0)\n"
    "    print(f'factorial{n} is {factorial(n)}')\n"
)


def _exec_main_block():
    """Execute a copy of the module's __main__ guard body in an isolated namespace."""
    exec(
        compile(MAIN_BLOCK_SOURCE, "<main-block>", "exec"),
        {"factorial": factorial, "doctest": doctest},
    )


def test_main_block_prints_factorial_of_input():
    """Running the module as __main__ reads an int and prints its factorial."""
    with patch("builtins.input", return_value="5"), patch(
        "builtins.print"
    ) as mock_print, patch("doctest.testmod") as mock_doctest:
        _exec_main_block()

    mock_doctest.assert_called_once()
    mock_print.assert_called_once_with("factorial5 is 120")


def test_main_block_defaults_to_zero_on_empty_input():
    """An empty input string falls back to 0, so 0! == 1 is printed."""
    with patch("builtins.input", return_value=""), patch(
        "builtins.print"
    ) as mock_print, patch("doctest.testmod"):
        _exec_main_block()

    mock_print.assert_called_once_with("factorial0 is 1")
