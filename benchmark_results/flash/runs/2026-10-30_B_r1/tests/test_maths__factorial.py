"""Tests for maths/factorial.py — iterative and recursive factorial implementations."""

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _find_project_root() -> Path:
    """Return the directory that contains the ``maths`` package."""
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "maths" / "factorial.py").is_file():
            return candidate
    # Fall back to the directory holding the test file; imports may still work
    # when pytest has already put the project root on sys.path.
    return here.parent


_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    """factorial(n) returns the mathematically correct value for small n."""
    assert factorial(number) == expected


def test_factorial_matches_math_factorial():
    """factorial agrees with math.factorial for a range of inputs."""
    assert all(factorial(i) == math.factorial(i) for i in range(25))


def test_factorial_zero_is_one():
    """Boundary: 0! is defined as 1."""
    assert factorial(0) == 1


def test_factorial_one_is_one():
    """Boundary: 1! is 1."""
    assert factorial(1) == 1


def test_factorial_large_value():
    """factorial handles a moderately large input without overflow."""
    assert factorial(50) == math.factorial(50)


@pytest.mark.parametrize("number", [0.1, 1.5, -0.5, 2.0000001, 3.7])
def test_factorial_rejects_non_integral_values(number):
    """Non-integral floats raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial(number)


@pytest.mark.parametrize("number", [-1, -2, -100])
def test_factorial_rejects_negative_values(number):
    """Negative integers raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial(number)


def test_factorial_accepts_integral_float():
    """An integral float such as 5.0 passes the integral check but the range()
    call cannot consume a float, so a TypeError is raised."""
    with pytest.raises(TypeError):
        factorial(5.0)


def test_factorial_rejects_non_numeric_string():
    """A non-numeric string raises an error rather than silently succeeding."""
    with pytest.raises((ValueError, TypeError)):
        factorial("abc")


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
    """factorial_recursive(n) returns the mathematically correct value."""
    assert factorial_recursive(n) == expected


def test_factorial_recursive_matches_math_factorial():
    """factorial_recursive agrees with math.factorial for a range of inputs."""
    assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))


def test_factorial_recursive_zero_and_one_base_cases():
    """Boundary: the recursion base cases 0 and 1 both return 1."""
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1


def test_factorial_recursive_agrees_with_iterative():
    """Both implementations produce identical results for the same inputs."""
    for i in range(15):
        assert factorial_recursive(i) == factorial(i)


@pytest.mark.parametrize("n", [0.1, 1.5, -0.5, 2.5])
def test_factorial_recursive_rejects_non_integral_values(n):
    """Non-integer inputs raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(n)


@pytest.mark.parametrize("n", [-1, -2, -50])
def test_factorial_recursive_rejects_negative_values(n):
    """Negative integers raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="not defined for negative values"):
        factorial_recursive(n)


def test_factorial_recursive_rejects_integral_float():
    """Unlike the iterative version, the recursive one requires a real int."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(5.0)


def test_factorial_recursive_rejects_string():
    """A string input is rejected by the isinstance check."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive("5")


def test_factorial_recursive_rejects_bool_like_non_int():
    """A non-int object (e.g. None) is rejected with ValueError."""
    with pytest.raises(ValueError, match="only accepts integral values"):
        factorial_recursive(None)


# ---------------------------------------------------------------------------
# __main__ block behaviour
# ---------------------------------------------------------------------------


def _module_path() -> Path:
    """Locate the source file of maths/factorial.py."""
    import maths.factorial as factorial_module

    path = getattr(factorial_module, "__file__", None)
    if path:
        return Path(path).resolve()

    candidate = _PROJECT_ROOT / "maths" / "factorial.py"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError("could not locate maths/factorial.py")


def _run_main(input_value):
    """Execute the module's __main__ block via runpy using its real file path."""
    import runpy

    module_path = _module_path()
    with patch("builtins.input", return_value=input_value):
        with patch("builtins.print") as mock_print:
            runpy.run_path(str(module_path), run_name="__main__")
    return " ".join(
        str(call.args[0]) for call in mock_print.call_args_list if call.args
    )


def test_main_block_reads_input_and_prints_result():
    """Running the module as a script reads stdin and prints the factorial."""
    printed = _run_main("5")
    assert "120" in printed


def test_main_block_defaults_to_zero_on_empty_input():
    """An empty input line falls back to 0, whose factorial is 1."""
    printed = _run_main("")
    assert "1" in printed
