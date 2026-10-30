"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path

import pytest

# Insert the directory that actually contains the `maths` package (the test file
# may live either inside <root>/maths/ or directly in <root>/).
_TEST_DIR = Path(__file__).resolve().parent
for _candidate in (_TEST_DIR, _TEST_DIR.parent):
    if (_candidate / "maths" / "factorial.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from maths.factorial import factorial, factorial_recursive  # noqa: E402


# --------------------------------------------------------------------------- #
# factorial() — happy path
# --------------------------------------------------------------------------- #
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


@pytest.mark.parametrize("number", list(range(0, 30)))
def test_factorial_matches_math_factorial(number):
    """factorial() agrees with math.factorial across a range of inputs."""
    assert factorial(number) == math.factorial(number)


def test_factorial_zero_is_one():
    """The empty product convention: 0! == 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary case: 1! == 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial() handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


def test_factorial_returns_int_type():
    """The return value is always a plain int."""
    result = factorial(7)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


# --------------------------------------------------------------------------- #
# factorial() — edge / error cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("number", [0.1, 1.5, 2.999, -0.5, 3.0000001])
def test_factorial_rejects_non_integral_floats(number):
    """Non-integral floats raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """Negative integers raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_rejects_integral_float_via_type_error():
    """An integral float survives the `number != int(number)` guard, but the

    subsequent `range(1, number + 1)` requires a real int, so a TypeError is
    raised instead of returning a value.
    """
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_rejects_numeric_string():
    """int("5") == 5 != "5", so the integral-value guard raises ValueError."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial("5")


def test_factorial_rejects_non_convertible_object():
    """A value int() cannot convert propagates a TypeError from the comparison."""
    with pytest.raises(TypeError):
        factorial(None)


# --------------------------------------------------------------------------- #
# factorial_recursive() — happy path
# --------------------------------------------------------------------------- #
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


@pytest.mark.parametrize("n", list(range(0, 25)))
def test_factorial_recursive_matches_math_factorial(n):
    """factorial_recursive() agrees with math.factorial across a range of inputs."""
    assert factorial_recursive(n) == math.factorial(n)


@pytest.mark.parametrize("n", list(range(0, 15)))
def test_iterative_and_recursive_agree(n):
    """Both implementations produce identical results for the same input."""
    assert factorial(n) == factorial_recursive(n)


def test_factorial_recursive_base_cases():
    """The base cases 0 and 1 both return 1 without recursing."""
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1


def test_factorial_recursive_returns_int_type():
    """The recursive variant also returns a plain int."""
    result = factorial_recursive(8)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


# --------------------------------------------------------------------------- #
# factorial_recursive() — edge / error cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0.1, 1.5, -0.5, 2.999])
def test_factorial_recursive_rejects_non_int(n):
    """Non-int values raise ValueError (isinstance check, not equality check)."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_integral_float():
    """Unlike factorial(), an integral float is rejected because of isinstance()."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_factorial_recursive_rejects_negative_values(n):
    """Negative integers raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_bool():
    """bool is a subclass of int, so True/False are accepted as 1/0."""
    assert factorial_recursive(True) == 1
    assert factorial_recursive(False) == 1


def test_factorial_recursive_rejects_string():
    """A string fails the isinstance check and raises ValueError."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


# --------------------------------------------------------------------------- #
# __main__ block behaviour
# --------------------------------------------------------------------------- #
def _run_as_main():
    """Execute maths/factorial.py as __main__ via its real file path."""
    import runpy

    import maths.factorial as module

    runpy.run_path(str(Path(module.__file__)), run_name="__main__")


def test_main_block_prints_factorial(monkeypatch, capsys):
    """Running the module as __main__ reads stdin and prints the factorial."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("5\n"))
    _run_as_main()
    captured = capsys.readouterr()
    assert "factorial5 is 120" in captured.out


def test_main_block_defaults_to_zero_on_empty_input(monkeypatch, capsys):
    """An empty stdin line falls back to 0, so 0! == 1 is printed."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    _run_as_main()
    captured = capsys.readouterr()
    assert "factorial0 is 1" in captured.out


def test_doctests_pass():
    """The module's embedded doctests all pass."""
    import doctest

    import maths.factorial as module

    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
