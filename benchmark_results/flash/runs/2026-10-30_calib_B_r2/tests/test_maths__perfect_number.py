"""Tests for maths/perfect_number.py — the `perfect` function."""

import pytest

from maths.perfect_number import perfect


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "number",
    [6, 28, 496, 8128],
)
def test_perfect_returns_true_for_known_perfect_numbers(number):
    """Known perfect numbers must be identified as perfect."""
    assert perfect(number) is True


def test_perfect_returns_true_for_large_perfect_number():
    """A large perfect number (33550336) must still be detected correctly."""
    assert perfect(33550336) is True


@pytest.mark.parametrize(
    "number",
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 27, 29, 100],
)
def test_perfect_returns_false_for_non_perfect_numbers(number):
    """Non-perfect positive integers must return False."""
    assert perfect(number) is False


def test_perfect_returns_false_just_above_large_perfect_number():
    """The integer directly after a large perfect number is not perfect."""
    assert perfect(33550337) is False


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number", [0, -1, -6, -28, -1000])
def test_perfect_returns_false_for_zero_and_negative_numbers(number):
    """Zero and negative numbers are not perfect numbers."""
    assert perfect(number) is False


def test_perfect_returns_false_for_one():
    """1 is the smallest positive integer and is not a perfect number."""
    assert perfect(1) is False


def test_perfect_returns_false_for_two():
    """2 is the smallest prime; its only proper divisor is 1, so it is not perfect."""
    assert perfect(2) is False


def test_perfect_returns_true_for_smallest_perfect_number():
    """6 is the smallest perfect number — lower boundary of the perfect set."""
    assert perfect(6) is True


def test_perfect_returns_bool_type():
    """The return value must be a real bool, not a truthy int."""
    assert isinstance(perfect(6), bool)
    assert isinstance(perfect(7), bool)


# ---------------------------------------------------------------------------
# Invalid input handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_value",
    ["123", "Hello", "", "6", 12.34, 0.0, -1.5, None, [6], (6,), {"n": 6}, True],
)
def test_perfect_raises_value_error_for_non_integer_input(bad_value):
    """Non-integer inputs must raise ValueError with the documented message.

    Note: ``bool`` is a subclass of ``int`` in Python, so ``True`` is accepted
    by the isinstance check; it is therefore excluded from the raising cases
    below and handled separately.
    """
    if isinstance(bad_value, bool):
        pytest.skip("bool is a subclass of int and is accepted by the function")
    with pytest.raises(ValueError, match="number must be an integer"):
        perfect(bad_value)


def test_perfect_accepts_bool_as_int_subclass():
    """Because bool subclasses int, True behaves like 1 and False like 0."""
    assert perfect(True) is False  # equivalent to perfect(1)
    assert perfect(False) is False  # equivalent to perfect(0)


def test_perfect_error_message_is_exact():
    """The raised ValueError must carry the exact documented message."""
    with pytest.raises(ValueError) as exc_info:
        perfect("not a number")
    assert str(exc_info.value) == "number must be an integer"


# ---------------------------------------------------------------------------
# Consistency / property-style checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number", [6, 28, 496, 8128])
def test_perfect_matches_brute_force_divisor_sum(number):
    """Result must agree with an independent brute-force proper-divisor sum."""
    expected = sum(d for d in range(1, number) if number % d == 0) == number
    assert perfect(number) is expected


def test_perfect_is_deterministic():
    """Repeated calls with the same input must yield the same result."""
    results = {perfect(28) for _ in range(5)}
    assert results == {True}