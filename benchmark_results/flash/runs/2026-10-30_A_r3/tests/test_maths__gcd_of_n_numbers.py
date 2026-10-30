"""Tests for maths/gcd_of_n_numbers.py"""

import importlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve().parent
for _candidate in (str(_HERE), str(_HERE.parent)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


# The source module may live at the repo root (``gcd_of_n_numbers.py``) or
# inside a ``maths`` package depending on the checkout layout, and there is no
# guarantee the package is importable.  Load it defensively and patch
# attributes on the already-imported module object rather than using string
# targets such as ``"maths.gcd_of_n_numbers.get_factors"`` (which force
# unittest.mock to re-import the top-level package and blow up here).
def _load_gcd_module():
    for name in ("maths.gcd_of_n_numbers", "gcd_of_n_numbers"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue

    for base in (_HERE, _HERE / "maths", _HERE.parent / "maths"):
        path = base / "gcd_of_n_numbers.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "gcd_of_n_numbers", path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("gcd_of_n_numbers", module)
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError("cannot locate gcd_of_n_numbers module")


gcd_module = _load_gcd_module()

get_factors = gcd_module.get_factors
get_greatest_common_divisor = gcd_module.get_greatest_common_divisor


# --------------------------------------------------------------------------- #
# get_factors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "number, expected",
    [
        (2, Counter({2: 1})),
        (3, Counter({3: 1})),
        (4, Counter({2: 2})),
        (23, Counter({23: 1})),
        (45, Counter({3: 2, 5: 1})),
        (100, Counter({2: 2, 5: 2})),
        (2520, Counter({2: 3, 3: 2, 5: 1, 7: 1})),
        (1024, Counter({2: 10})),
    ],
)
def test_get_factors_happy_path(number, expected):
    """get_factors returns the correct prime factorisation for positive ints."""
    assert get_factors(number) == expected


def test_get_factors_one_returns_counter_with_one():
    """The special case number == 1 returns Counter({1: 1})."""
    assert get_factors(1) == Counter({1: 1})


def test_get_factors_prime_number_returns_itself():
    """A prime number's only factor is itself with exponent 1."""
    assert get_factors(97) == Counter({97: 1})


def test_get_factors_accepts_prefilled_counter():
    """A pre-populated Counter is mutated/extended rather than replaced."""
    seed = Counter({2: 1})
    result = get_factors(9, factors=seed)
    assert result[3] == 2
    assert result[2] == 1


def test_get_factors_accepts_custom_starting_factor():
    """Starting the search at a higher factor skips smaller primes."""
    result = get_factors(15, factor=3)
    assert result == Counter({3: 1, 5: 1})


@pytest.mark.parametrize("bad_value", [0, -1, -100, 1.5, "10", None])
def test_get_factors_rejects_invalid_input(bad_value):
    """Non-positive or non-integer inputs raise TypeError."""
    with pytest.raises(TypeError, match="number must be integer and greater than zero"):
        get_factors(bad_value)


def test_get_factors_product_of_factors_equals_number():
    """Sanity check: multiplying factors**powers reconstructs the number."""
    number = 8316
    factors = get_factors(number)
    product = 1
    for base, power in factors.items():
        product *= base**power
    assert product == number


# --------------------------------------------------------------------------- #
# get_greatest_common_divisor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "numbers, expected",
    [
        ((18, 45), 9),
        ((23, 37), 1),
        ((2520, 8350), 10),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1),
        ((12, 18, 24), 6),
        ((7, 7, 7), 7),
        ((100, 100), 100),
        ((2, 4, 8, 16), 2),
        ((36, 60, 84), 12),
    ],
)
def test_get_greatest_common_divisor_happy_path(numbers, expected):
    """GCD of n positive integers is computed correctly."""
    assert get_greatest_common_divisor(*numbers) == expected


def test_get_greatest_common_divisor_single_number():
    """A single number's GCD is the number itself."""
    assert get_greatest_common_divisor(42) == 42


def test_get_greatest_common_divisor_one_is_identity():
    """Including 1 in the list forces the GCD to be 1."""
    assert get_greatest_common_divisor(1, 100, 250) == 1


def test_get_greatest_common_divisor_coprime_numbers():
    """Coprime numbers have a GCD of 1."""
    assert get_greatest_common_divisor(17, 19, 23) == 1


@pytest.mark.parametrize(
    "numbers",
    [
        (-10, 20),
        (0, 5),
        (1.5, 2),
        ("1", 2, 3),
        (10, None),
    ],
)
def test_get_greatest_common_divisor_rejects_invalid_input(numbers):
    """Invalid inputs raise a generic Exception with the documented message."""
    with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
        get_greatest_common_divisor(*numbers)


def test_get_greatest_common_divisor_propagates_type_error_as_exception():
    """A TypeError raised by get_factors is wrapped into a plain Exception."""
    with patch.object(
        gcd_module,
        "get_factors",
        side_effect=TypeError("boom"),
    ):
        with pytest.raises(Exception, match="numbers must be integer and greater than zero"):
            get_greatest_common_divisor(4, 8)


def test_get_greatest_common_divisor_uses_get_factors():
    """get_greatest_common_divisor delegates factorisation to get_factors."""
    with patch.object(
        gcd_module,
        "get_factors",
        side_effect=lambda n: Counter({2: 1}) if n % 2 == 0 else Counter({3: 1}),
    ) as mocked:
        result = get_greatest_common_divisor(4, 8, 16)
    assert result == 2
    assert mocked.call_count == 3


def test_get_greatest_common_divisor_result_divides_all_inputs():
    """Property check: the returned GCD divides every input exactly."""
    numbers = (144, 180, 240, 360)
    gcd = get_greatest_common_divisor(*numbers)
    assert all(n % gcd == 0 for n in numbers)
    assert gcd == 12
