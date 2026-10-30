"""
Tests for maths/gcd_of_n_numbers.py

Covers:
    - get_factors
    - get_greatest_common_divisor
"""

from collections import Counter

import pytest

from maths.gcd_of_n_numbers import get_factors, get_greatest_common_divisor


# ---------------------------------------------------------------------------
# get_factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number, expected",
    [
        (45, Counter({3: 2, 5: 1})),
        (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
        (23, Counter({23: 1})),
        (2, Counter({2: 1})),
        (4, Counter({2: 2})),
        (100, Counter({2: 2, 5: 2})),
        (97, Counter({97: 1})),  # prime
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime-factor Counter for valid inputs."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


@pytest.mark.parametrize("bad_number", [0, -1, -100, 1.5, "10", None])
def test_get_factors_invalid_input_raises_type_error(bad_number):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError, match="number must be integer and greater than zero"):
        get_factors(bad_number)


def test_get_factors_accepts_prepopulated_counter():
    """A pre-populated Counter is mutated/extended rather than replaced."""
    seed = Counter({2: 1})
    result = get_factors(4, factors=seed)
    # The supplied Counter is used as-is, so its existing counts accumulate:
    # 1 (seed) + 1 (4 // 2 == 2 step) + 1 (break condition at number == factor).
    assert result == Counter({2: 3})
    assert result is seed


def test_get_factors_custom_starting_factor():
    """Starting from a factor that divides the number yields that factor."""
    # 9 with factor=3 -> 3 * 3
    assert get_factors(9, factor=3) == Counter({3: 2})


def test_get_factors_product_of_factors_equals_number():
    """The product of factor**power must reconstruct the original number."""
    number = 360
    factors = get_factors(number)
    product = 1
    for base, power in factors.items():
        product *= base**power
    assert product == number


# ---------------------------------------------------------------------------
# get_greatest_common_divisor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "numbers, expected",
    [
        ((18, 45), 9),
        ((23, 37), 1),
        ((2520, 8350), 10),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
        ((12, 18, 24), 6),
        ((7, 7, 7), 7),
        ((100, 10), 10),
        ((2, 4, 8, 16), 2),
        ((1, 1), 1),
        ((36, 60, 84), 12),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """get_greatest_common_divisor returns the correct gcd for valid inputs."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number():
    """A single number's gcd is the number itself."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_single_one():
    """A single 1 returns 1."""
    assert get_greatest_common_divisor(1) == 1


def test_get_greatest_common_divisor_coprime_numbers():
    """Coprime numbers have a gcd of 1."""
    assert get_greatest_common_divisor(17, 19, 23) == 1


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3, 4, 5, 6, 7, 8, 9, 10),
        (10, None),
    ],
)
def test_get_greatest_common_divisor_invalid_input_raises_exception(numbers):
    """Invalid inputs raise a generic Exception with the documented message."""
    with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
        get_greatest_common_divisor(*numbers)


def test_get_greatest_common_divisor_no_arguments_raises():
    """Calling with no numbers raises (unpacking an empty map fails)."""
    with pytest.raises(Exception):
        get_greatest_common_divisor()


def test_get_greatest_common_divisor_is_commutative():
    """The gcd is independent of the order of the arguments."""
    assert get_greatest_common_divisor(18, 45, 27) == get_greatest_common_divisor(
        27, 18, 45
    )


def test_get_greatest_common_divisor_divides_all_inputs():
    """The returned gcd must divide every input number exactly."""
    numbers = (48, 72, 120)
    gcd = get_greatest_common_divisor(*numbers)
    assert all(n % gcd == 0 for n in numbers)
    assert gcd == 24
