"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve().parent
for _candidate in (str(_HERE), str(_HERE.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from maths.factorial import factorial, factorial_recursive


def _module_source_path() -> Path:
    """Absolute path of maths/factorial.py, derived from the imported function."""
    return Path(factorial.__globals__["__file__"]).resolve()


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
        (4, 24),
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
    assert all(factorial(i) == math.factorial(i) for i in range(30))


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0 + 1e-12])
def test_factorial_rejects_non_integral_values(number):
    """factorial() raises ValueError for non-integral numeric input."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


def test_factorial_integral_float_raises_type_error():
    """An integral float passes the integral check, then fails inside range()."""
    with pytest.raises(TypeError):
        factorial(5.0)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """factorial() raises ValueError for negative input."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_rejects_non_numeric():
    """factorial() raises ValueError from its own integral check for a string."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial("5")


def test_factorial_returns_int_type():
    """factorial() returns an int, not a float."""
    result = factorial(7)
    assert isinstance(result, int)
    assert result == 5040


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
    """factorial_recursive() returns the correct value for known inputs."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive() agrees with math.factorial for a range of inputs."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_zero_and_one_base_cases():
    """Boundary: recursive base cases 0 and 1 both return 1."""
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


def test_factorial_recursive_rejects_integral_float():
    """factorial_recursive() rejects even integral floats because it type-checks."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(n):
    """factorial_recursive() raises ValueError for negative input."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_non_numeric():
    """factorial_recursive() raises ValueError for a non-numeric type."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_returns_int_type():
    """factorial_recursive() returns an int."""
    result = factorial_recursive(7)
    assert isinstance(result, int)
    assert result == 5040


def test_factorial_recursive_recursion_depth():
    """factorial_recursive() recurses n times; verify with a call counter."""
    calls = []
    original = factorial_recursive

    def counting(n):
        calls.append(n)
        return original(n)

    with patch.object(sys.modules[__name__], "factorial_recursive", counting):
        assert counting(5) == 120
    assert calls == [5]


# ---------------------------------------------------------------------------
# __main__ block behaviour
# ---------------------------------------------------------------------------


def test_main_block_prints_factorial(monkeypatch, capsys):
    """Running the module as __main__ reads input and prints the factorial."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "5")
    monkeypatch.setattr("doctest.testmod", lambda *a, **k: None)
    runpy.run_path(str(_module_source_path()), run_name="__main__")
    captured = capsys.readouterr()
    assert "factorial5 is 120" in captured.out


def test_main_block_defaults_to_zero_on_empty_input(monkeypatch, capsys):
    """Empty input in the __main__ block falls back to 0, printing 1."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr("doctest.testmod", lambda *a, **k: None)
    runpy.run_path(str(_module_source_path()), run_name="__main__")
    captured = capsys.readouterr()
    assert "factorial0 is 1" in captured.out
