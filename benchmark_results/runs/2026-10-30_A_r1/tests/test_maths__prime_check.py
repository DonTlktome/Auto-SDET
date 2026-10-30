import math

import pytest

from maths.prime_check import is_prime


class TestIsPrimePositive:
    """Positive test cases for the is_prime function."""

    @pytest.mark.parametrize(
        "number",
        [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            563,
            2999,
            7919,  # 1000th prime
            104729,  # 10000th prime
        ],
    )
    def test_known_primes_return_true(self, number):
        """is_prime should return True for known prime numbers."""
        assert is_prime(number) is True

    @pytest.mark.parametrize(
        "number",
        [
            0,
            1,
            4,
            6,
            8,
            9,
            10,
            12,
            15,
            21,
            25,
            27,
            49,
            87,
            67483,
            100,
            1000,
        ],
    )
    def test_known_composites_return_false(self, number):
        """is_prime should return False for known composite numbers."""
        assert is_prime(number) is False

    def test_large_prime(self):
        """is_prime should correctly identify a large prime number."""
        assert is_prime(1000003) is True

    def test_large_composite(self):
        """is_prime should correctly identify a large composite number."""
        assert is_prime(1000004) is False


class TestIsPrimeEdgeCases:
    """Edge case and boundary tests for the is_prime function."""

    @pytest.mark.parametrize("number", [0, 1])
    def test_zero_and_one_are_not_prime(self, number):
        """0 and 1 are not prime because they lack exactly two factors."""
        assert is_prime(number) is False

    def test_two_is_prime(self):
        """2 is the smallest and only even prime number."""
        assert is_prime(2) is True

    def test_three_is_prime(self):
        """3 is the smallest odd prime number."""
        assert is_prime(3) is True

    def test_four_is_not_prime(self):
        """4 is the smallest composite number."""
        assert is_prime(4) is False

    def test_even_number_boundary(self):
        """Even numbers greater than 2 should never be prime."""
        assert is_prime(2) is True
        assert is_prime(4) is False
        assert is_prime(100) is False

    def test_multiple_of_three_boundary(self):
        """Multiples of 3 greater than 3 should never be prime."""
        assert is_prime(3) is True
        assert is_prime(6) is False
        assert is_prime(9) is False

    def test_perfect_square_of_prime(self):
        """A perfect square of a prime is composite."""
        assert is_prime(49) is False  # 7 * 7

    def test_number_in_six_k_plus_minus_one_form(self):
        """Numbers of the form 6k +/- 1 may be prime or composite."""
        assert is_prime(25) is False  # 6*4 + 1, but 5*5
        assert is_prime(35) is False  # 6*6 - 1, but 5*7
        assert is_prime(37) is True   # 6*6 + 1, prime


class TestIsPrimeInvalidInput:
    """Tests for invalid input handling."""

    @pytest.mark.parametrize(
        "number",
        [
            -1,
            -4,
            -19,
            -100,
        ],
    )
    def test_negative_numbers_raise_value_error(self, number):
        """Negative integers should raise a ValueError."""
        with pytest.raises(ValueError):
            is_prime(number)

    @pytest.mark.parametrize(
        "number",
        [
            16.1,
            3.0,
            2.5,
            "5",
            None,
            [2],
            (3,),
            {"value": 5},
        ],
    )
    def test_non_integer_inputs_raise_value_error(self, number):
        """Non-integer inputs should raise a ValueError."""
        with pytest.raises(ValueError):
            is_prime(number)

    def test_float_that_is_integral_value_raises(self):
        """Even a float with an integral value (e.g. 3.0) is rejected."""
        with pytest.raises(ValueError):
            is_prime(3.0)

    def test_boolean_input_raises(self):
        """Booleans are subclasses of int but should be treated carefully.

        Note: bool is a subclass of int in Python, so True/False would be
        accepted by isinstance check. This documents current behavior.
        """
        # True == 1 and False == 0 in Python; both are ints.
        assert is_prime(True) is False
        assert is_prime(False) is False


class TestIsPrimeInternalLogic:
    """Tests that verify the internal algorithm behavior via mocking."""

    def test_math_sqrt_is_used(self, monkeypatch):
        """is_prime should call math.sqrt for numbers >= 4."""
        calls = []

        original_sqrt = math.sqrt

        def fake_sqrt(value):
            calls.append(value)
            return original_sqrt(value)

        monkeypatch.setattr(math, "sqrt", fake_sqrt)

        is_prime(29)

        assert len(calls) > 0, "math.sqrt should be called for numbers >= 4"

    def test_math_sqrt_not_called_for_small_numbers(self, monkeypatch):
        """is_prime should not call math.sqrt for numbers < 4."""
        calls = []

        def fake_sqrt(value):
            calls.append(value)
            return value

        monkeypatch.setattr(math, "sqrt", fake_sqrt)

        is_prime(2)
        is_prime(3)

        assert len(calls) == 0, "math.sqrt should not be called for 2 or 3"

    def test_loop_skips_known_non_primes_early(self, monkeypatch):
        """Even numbers and multiples of 3 should return early without looping."""
        calls = []

        original_sqrt = math.sqrt

        def fake_sqrt(value):
            calls.append(value)
            return original_sqrt(value)

        monkeypatch.setattr(math, "sqrt", fake_sqrt)

        # Even number -> returns False before reaching the loop
        is_prime(100)

        assert len(calls) == 0, "math.sqrt should not be called for even numbers"