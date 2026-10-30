"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path

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
    """factorial() returns the correct value for known non-negative integers."""
    assert factorial(number) == expected


def test_factorial_matches_math_factorial():
    """factorial() agrees with math.factorial for a range of inputs."""
    assert all(factorial(i) == math.factorial(i) for i in range(25))


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1 (smallest positive input)."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_rejects_non_integral_values(number):
    """factorial() raises ValueError for non-integral numeric input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """factorial() raises ValueError for negative integers."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_integral_float_passes_guard_but_fails_range():
    """An integral float passes the guard but is rejected by range() -> TypeError.

    ``factorial`` only checks ``number != int(number)``, so 5.0 is accepted by the
    validation step; the for-loop over ``range(1, number + 1)`` then raises
    TypeError because range() requires an integer.
    """
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_rejects_non_numeric_type():
    """factorial() raises ValueError when given a non-numeric type.

    ``int("5")`` succeeds, so the length check ``"5" != 5`` triggers the
    ValueError branch rather than a TypeError.
    """
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial("5")


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
        (5, 120),
        (6, 720),
        (10, 3628800),
    ],
)
def test_factorial_recursive_happy_path(n, expected):
    """factorial_recursive() returns the correct value for known inputs."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive() agrees with math.factorial for a range of inputs."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_base_cases():
    """Boundary: base cases 0 and 1 both return 1."""
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1


def test_factorial_recursive_agrees_with_iterative():
    """Both implementations produce identical results for the same inputs."""
    for i in range(15):
        assert factorial_recursive(i) == factorial(i)


@pytest.mark.parametrize("n", [0.1, 1.5, 2.999, -0.5])
def test_factorial_recursive_rejects_non_integral_values(n):
    """factorial_recursive() raises ValueError for non-int input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(n):
    """factorial_recursive() raises ValueError for negative integers."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_integral_float():
    """factorial_recursive() uses isinstance, so 5.0 is rejected (unlike factorial)."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


def test_factorial_recursive_rejects_string():
    """factorial_recursive() raises ValueError for a string argument."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_bool_like_non_int():
    """factorial_recursive() rejects None as a non-integral value."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(None)


# ---------------------------------------------------------------------------
# Module-level behaviour (__main__ guard) — isolated with mocks
# ---------------------------------------------------------------------------


def test_main_block_prints_factorial(monkeypatch, capsys):
    """Running the module as __main__ reads input and prints the factorial."""
    import runpy

    import maths.factorial as factorial_module

    monkeypatch.setattr("builtins.input", lambda *a, **k: "5")
    runpy.run_path(factorial_module.__file__, run_name="__main__")
    captured = capsys.readouterr()
    assert "factorial5 is 120" in captured.out


def test_main_block_defaults_to_zero_on_empty_input(monkeypatch, capsys):
    """Empty input in the __main__ block falls back to 0, printing 0! == 1."""
    import runpy

    import maths.factorial as factorial_module

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    runpy.run_path(factorial_module.__file__, run_name="__main__")
    captured = capsys.readouterr()
    assert "factorial0 is 1" in captured.out
