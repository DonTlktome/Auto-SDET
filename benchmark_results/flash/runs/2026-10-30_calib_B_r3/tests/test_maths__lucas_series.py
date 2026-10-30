"""Tests for the Lucas series implementations in `lucas_series.py`."""

import pytest

from lucas_series import dynamic_lucas_number, recursive_lucas_number


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


def test_recursive_lucas_number_base_case_zero():
    """The base case n == 0 must return exactly 2."""
    assert recursive_lucas_number(0) == 2


def test_recursive_lucas_number_base_case_one():
    """The base case n == 1 must return exactly 1."""
    assert recursive_lucas_number(1) == 1


def test_recursive_lucas_number_recurrence_relation():
    """L(n) == L(n-1) + L(n-2) must hold for the recursive implementation."""
    for n in range(2, 15):
        assert recursive_lucas_number(n) == (
            recursive_lucas_number(n - 1) + recursive_lucas_number(n - 2)
        )


def test_recursive_lucas_number_returns_int():
    """The return type must be a plain int."""
    result = recursive_lucas_number(10)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


@pytest.mark.parametrize("bad_value", [1.5, -1.5, 0.0, "3", None, [1], (2,), 3 + 0j])
def test_recursive_lucas_number_rejects_non_int(bad_value):
    """Non-integer arguments must raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        recursive_lucas_number(bad_value)
    assert "recursive_lucas_number accepts only integer arguments." in str(
        exc_info.value
    )


def test_recursive_lucas_number_accepts_bool_as_int():
    """bool is a subclass of int, so True/False are accepted (True -> L(1), False -> L(0))."""
    assert recursive_lucas_number(True) == 1
    assert recursive_lucas_number(False) == 2


def test_recursive_lucas_number_negative_int_raises_recursion_error():
    """A negative integer is not guarded, so recursion never terminates."""
    with pytest.raises(RecursionError):
        recursive_lucas_number(-1)


# ---------------------------------------------------------------------------
# dynamic_lucas_number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, expected", KNOWN_LUCAS)
def test_dynamic_lucas_number_known_values(n, expected):
    """dynamic_lucas_number must return the documented Lucas numbers."""
    assert dynamic_lucas_number(n) == expected


def test_dynamic_lucas_number_base_case_zero():
    """The loop runs zero times for n == 0, returning the initial a == 2."""
    assert dynamic_lucas_number(0) == 2


def test_dynamic_lucas_number_base_case_one():
    """For n == 1 the loop runs once, returning the initial b == 1."""
    assert dynamic_lucas_number(1) == 1


def test_dynamic_lucas_number_returns_int():
    """The return type must be a plain int."""
    result = dynamic_lucas_number(10)
    assert isinstance(result, int)
    assert not isinstance(result, bool)


@pytest.mark.parametrize("bad_value", [1.5, -1.5, 0.0, "3", None, [1], (2,), 3 + 0j])
def test_dynamic_lucas_number_rejects_non_int(bad_value):
    """Non-integer arguments must raise TypeError with the documented message."""
    with pytest.raises(TypeError) as exc_info:
        dynamic_lucas_number(bad_value)
    assert "dynamic_lucas_number accepts only integer arguments." in str(exc_info.value)


def test_dynamic_lucas_number_accepts_bool_as_int():
    """bool is a subclass of int, so True/False are accepted (True -> L(1), False -> L(0))."""
    assert dynamic_lucas_number(True) == 1
    assert dynamic_lucas_number(False) == 2


def test_dynamic_lucas_number_negative_int_returns_two():
    """A negative integer makes range() empty, so the initial value 2 is returned."""
    assert dynamic_lucas_number(-1) == 2
    assert dynamic_lucas_number(-100) == 2


def test_dynamic_lucas_number_large_input():
    """The iterative version must handle large n without recursion limits."""
    # L(100) is a well-known value of the Lucas sequence.
    assert dynamic_lucas_number(100) == 792070839848372253127


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", list(range(0, 21)))
def test_both_implementations_agree(n):
    """The recursive and dynamic implementations must agree for all small n."""
    assert recursive_lucas_number(n) == dynamic_lucas_number(n)


def test_implementations_agree_on_larger_value():
    """Both implementations must agree on a moderately large index."""
    assert recursive_lucas_number(30) == dynamic_lucas_number(30) == 1860498