"""Tests for maths/perfect_number.py."""

import builtins
import inspect
import runpy
from unittest.mock import patch

import pytest

from maths.perfect_number import perfect


# Absolute path to the source file backing the imported module.  It is used to
# execute the module's ``__main__`` block without relying on ``sys.path``.
MODULE_FILE = inspect.getfile(perfect)


def _run_as_script():
    """Execute the module's source as if it were run via ``python -m``.

    ``runpy.run_module`` cannot be used here because it resolves the module
    through ``sys.path``, which does not necessarily contain the ``maths``
    package when pytest imports the test module directly.  ``runpy.run_path``
    only needs the file that actually backs the imported module, so it works
    regardless of how the test module itself was imported.

    The module's doctest suite is stubbed out so this helper exercises only the
    interactive ``__main__`` block.
    """
    with patch("doctest.testmod"):
        return runpy.run_path(MODULE_FILE, run_name="__main__")


# ---------------------------------------------------------------------------
# perfect() -- happy path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "number",
    [6, 28, 496, 8128],
)
def test_perfect_returns_true_for_known_perfect_numbers(number):
    """Known perfect numbers must be reported as perfect."""
    assert perfect(number) is True


@pytest.mark.parametrize(
    "number",
    [1, 2, 3, 4, 5, 7, 12, 27, 29, 100, 33550337],
)
def test_perfect_returns_false_for_non_perfect_numbers(number):
    """Non-perfect positive integers must be reported as not perfect."""
    assert perfect(number) is False


def test_perfect_large_perfect_number():
    """A large perfect number (33550336) must be detected as perfect."""
    assert perfect(33550336) is True


# ---------------------------------------------------------------------------
# perfect() -- edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("number", [0, -1, -6, -28, -100])
def test_perfect_returns_false_for_zero_and_negatives(number):
    """Zero and negative numbers are never perfect and must return False."""
    assert perfect(number) is False


def test_perfect_one_is_not_perfect():
    """1 has no proper divisors, so it is not a perfect number."""
    assert perfect(1) is False


def test_perfect_two_is_not_perfect():
    """2's only proper divisor is 1, so it is not perfect."""
    assert perfect(2) is False


def test_perfect_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy other object."""
    assert isinstance(perfect(6), bool)
    assert isinstance(perfect(7), bool)


# ---------------------------------------------------------------------------
# perfect() -- invalid input types
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_value",
    ["123", "Hello", "", 12.34, 0.0, None, [6], (6,), {"n": 6}, True],
)
def test_perfect_raises_value_error_for_non_int(bad_value):
    """Non-int inputs must raise ValueError with the documented message.

    Note: ``bool`` is a subclass of ``int`` in Python, so ``True`` is accepted
    by the isinstance check; it is included here only to document behaviour
    and is expected to raise only if the implementation rejects bools.
    """
    if isinstance(bad_value, bool):
        # bool is an int subclass -> implementation treats it as an integer.
        assert perfect(bad_value) is False
        return
    with pytest.raises(ValueError, match="number must be an integer"):
        perfect(bad_value)


def test_perfect_raises_value_error_message_exact():
    """The raised ValueError must carry the exact documented message."""
    with pytest.raises(ValueError) as exc_info:
        perfect("123")
    assert str(exc_info.value) == "number must be an integer"


def test_perfect_accepts_int_subclass():
    """An int subclass instance should be handled like a plain int."""

    class MyInt(int):
        pass

    assert perfect(MyInt(6)) is True
    assert perfect(MyInt(7)) is False


# ---------------------------------------------------------------------------
# perfect() -- divisor logic sanity checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "number, expected",
    [
        (6, True),
        (28, True),
        (496, True),
        (8128, True),
        (8, False),
        (10, False),
        (20, False),
        (30, False),
    ],
)
def test_perfect_divisor_sum_logic(number, expected):
    """Verify the proper-divisor sum logic across a range of inputs."""
    assert perfect(number) is expected


def test_perfect_does_not_use_zero_division():
    """Calling perfect(0) must not raise ZeroDivisionError."""
    try:
        result = perfect(0)
    except ZeroDivisionError:  # pragma: no cover - explicit failure path
        pytest.fail("perfect(0) raised ZeroDivisionError")
    assert result is False


# ---------------------------------------------------------------------------
# __main__ block behaviour
# ---------------------------------------------------------------------------
def test_main_block_prints_perfect_message():
    """Running the module as a script with a perfect number prints the message."""
    with patch.object(builtins, "input", return_value="6"):
        with patch.object(builtins, "print") as mock_print:
            _run_as_script()

    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert "6 is a Perfect Number." in printed


def test_main_block_prints_not_perfect_message():
    """Running the module as a script with a non-perfect number prints 'not'."""
    with patch.object(builtins, "input", return_value="7"):
        with patch.object(builtins, "print") as mock_print:
            _run_as_script()

    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert "7 is not a Perfect Number." in printed


def test_main_block_raises_value_error_for_non_integer_input():
    """Non-integer script input must raise ValueError with the documented message."""
    with patch.object(builtins, "input", return_value="abc"):
        with patch.object(builtins, "print"):
            with pytest.raises(ValueError, match="number must be an integer"):
                _run_as_script()


def test_main_block_strips_whitespace_from_input():
    """Whitespace around the script input must be stripped before parsing."""
    with patch.object(builtins, "input", return_value="  28  "):
        with patch.object(builtins, "print") as mock_print:
            _run_as_script()

    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert "28 is a Perfect Number." in printed
