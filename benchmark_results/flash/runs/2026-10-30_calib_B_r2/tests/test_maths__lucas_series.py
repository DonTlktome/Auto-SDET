"""Tests for the Lucas series implementations in maths/lucas_series.py."""

import pytest

from maths.lucas_series import dynamic_lucas_number, recursive_lucas_number


# ---------------------------------------------------------------------------
# Known Lucas numbers: L(0)=2, L(1)=1, L(2)=3, L(3)=4, L(4)=7, L(5)=11, ...
# ---------------------------------------------------------------------------
KNOWN_LUCAS = [
    (0, 2),
    (1, 1),
    (2, 3),
    (3, 4),
    (4, 7),
    (5, 11),
    (6, 18),
    (7, 29),
    (8, 47),
    (9, 76),
    (10, 123),
    (20, 15127),
    (25, 167761),
]


# ---------------------------------------------------------------------------
# recursive_lucas_number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", KNOWN_LUCAS)
def test_recursive_lucas_number_known_values(n, expected):
    """recursive_lucas_number must return the documented Lucas numbers."""
    assert recursive_lucas_number(n) == expected


def test_recursive_lucas_number_base_cases():
    """The two base cases (n=0 -> 2, n=1 -> 1) must be handled explicitly."""
    assert recursive_lucas_number(0) == 2
    assert recursive_lucas_number(1) == 1


def test_recursive_lucas_number_large_index():
    """A larger index must still produce the correct value (L(30) == 1860498)."""
    assert recursive_lucas_number(30) == 1860498


@pytest.mark.parametrize("bad_value", [1.5, -1.5, 0.0, "3", None, [1], (2,), True and 2.0])
def test_recursive_lucas_number_rejects_non_int(bad_value):
    """Non-integer arguments must raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        recursive_lucas_number(bad_value)
    assert "recursive_lucas_number accepts only integer arguments." in str(
        exc_info.value
    )


def test_recursive_lucas_number_accepts_bool():
    """bool is a subclass of int, so it must be accepted (True == 1, False == 0)."""
    assert recursive_lucas_number(True) == 1
    assert recursive_lucas_number(False) == 2


def test_recursive_lucas_number_negative_index():
    """Negative indices are not validated; they recurse until RecursionError."""
    with pytest.raises(RecursionError):
        recursive_lucas_number(-1)


# ---------------------------------------------------------------------------
# dynamic_lucas_number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", KNOWN_LUCAS)
def test_dynamic_lucas_number_known_values(n, expected):
    """dynamic_lucas_number must return the documented Lucas numbers."""
    assert dynamic_lucas_number(n) == expected


def test_dynamic_lucas_number_base_cases():
    """The iterative version must also honour L(0)=2 and L(1)=1."""
    assert dynamic_lucas_number(0) == 2
    assert dynamic_lucas_number(1) == 1


def test_dynamic_lucas_number_large_index():
    """A larger index must still produce the correct value (L(30) == 1860498)."""
    assert dynamic_lucas_number(30) == 1860498


@pytest.mark.parametrize("bad_value", [1.5, -1.5, 0.0, "3", None, [1], (2,)])
def test_dynamic_lucas_number_rejects_non_int(bad_value):
    """Non-integer arguments must raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        dynamic_lucas_number(bad_value)
    assert "dynamic_lucas_number accepts only integer arguments." in str(exc_info.value)


def test_dynamic_lucas_number_accepts_bool():
    """bool is a subclass of int, so it must be accepted (True == 1, False == 0)."""
    assert dynamic_lucas_number(True) == 1
    assert dynamic_lucas_number(False) == 2


def test_dynamic_lucas_number_negative_index_returns_initial_value():
    """A negative index makes the loop body never execute, returning the seed 2."""
    assert dynamic_lucas_number(-1) == 2
    assert dynamic_lucas_number(-100) == 2


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", list(range(0, 16)))
def test_recursive_and_dynamic_agree(n):
    """Both implementations must agree for every index in a small range."""
    assert recursive_lucas_number(n) == dynamic_lucas_number(n)


def test_implementations_agree_on_larger_index():
    """Both implementations must agree on a moderately large index."""
    assert recursive_lucas_number(22) == dynamic_lucas_number(22) == 39603


def test_implementations_raise_same_error_type():
    """Both implementations must raise TypeError for the same invalid input."""
    with pytest.raises(TypeError):
        recursive_lucas_number(2.5)
    with pytest.raises(TypeError):
        dynamic_lucas_number(2.5)