import math
from unittest.mock import Mock

import pytest

from maths.prime_check import is_prime


class TestIsPrimePositive:
    '''Positive (happy path) test cases for is_prime().'''

    @pytest.mark.parametrize(
        'number',
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
    def test_returns_true_for_prime_numbers(self, number):
        '''is_prime() should return True for known prime numbers.'''
        assert is_prime(number) is True

    @pytest.mark.parametrize(
        'number',
        [
            0,
            1,
            4,
            6,
            8,
            9,
            15,
            21,
            25,
            27,
            87,
            67483,
            100,  # even composite
            121,  # 11 * 11
        ],
    )
    def test_returns_false_for_non_prime_numbers(self, number):
        '''is_prime() should return False for non-prime (composite/0/1) numbers.'''
        assert is_prime(number) is False


class TestIsPrimeEdgeCases:
    '''Edge case and boundary test cases for is_prime().'''

    @pytest.mark.parametrize('number', [0, 1, 2, 3, 4])
    def test_small_boundary_values(self, number):
        '''is_prime() should handle the smallest boundary values correctly.'''
        expected = number in (2, 3)
        assert is_prime(number) is expected

    @pytest.mark.parametrize('number', [4, 5, 6, 7])
    def test_transition_around_small_primes(self, number):
        '''is_prime() should correctly classify numbers around the 2/3/5/7 boundary.'''
        expected = number in (5, 7)
        assert is_prime(number) is expected

    def test_large_prime(self):
        '''is_prime() should correctly identify a large prime number.'''
        assert is_prime(1000003) is True

    def test_large_composite(self):
        '''is_prime() should correctly identify a large composite number.'''
        assert is_prime(1000004) is False

    def test_perfect_square_of_prime(self):
        '''is_prime() should return False for the square of a prime.'''
        assert is_prime(13 * 13) is False

    def test_product_of_two_large_primes(self):
        '''is_prime() should return False for a semiprime (product of two primes).'''
        assert is_prime(7919 * 7907) is False

    def test_boolean_values_return_false(self):
        '''is_prime() should return False for bool values because bool is a subclass of int.'''
        assert is_prime(True) is False
        assert is_prime(False) is False


class TestIsPrimeInvalidInput:
    '''Tests for invalid input handling in is_prime().'''

    @pytest.mark.parametrize(
        'number',
        [
            -1,
            -4,
            -19,
            -1000000,
        ],
    )
    def test_negative_numbers_raise_value_error(self, number):
        '''is_prime() should raise ValueError for negative integers.'''
        with pytest.raises(ValueError, match='positive integers'):
            is_prime(number)

    @pytest.mark.parametrize(
        'number',
        [
            16.1,
            2.0,
            0.5,
            '5',
            None,
            [2],
            (3,),
        ],
    )
    def test_non_integer_inputs_raise_value_error(self, number):
        '''is_prime() should raise ValueError for non-integer inputs.'''
        with pytest.raises(ValueError, match='positive integers'):
            is_prime(number)

    def test_float_that_is_integral_value_raises(self):
        '''is_prime() should reject floats even if they represent whole numbers.'''
        with pytest.raises(ValueError):
            is_prime(2.0)


class TestIsPrimeInternalBehavior:
    '''Tests that verify the internal algorithm behavior via mocking.'''

    def test_uses_sqrt_bound_in_loop(self, monkeypatch):
        '''is_prime() should only iterate up to sqrt(number) using the 6k±1 pattern.'''
        original_sqrt = math.sqrt
        sqrt_mock = Mock(wraps=original_sqrt)
        monkeypatch.setattr(math, 'sqrt', sqrt_mock)

        result = is_prime(97)

        assert result is True
        sqrt_mock.assert_called_once_with(97)

    def test_returns_early_for_small_primes_without_loop(self, monkeypatch):
        '''is_prime() should short-circuit for 2 and 3 without entering the loop.'''
        sqrt_mock = Mock()
        monkeypatch.setattr(math, 'sqrt', sqrt_mock)

        assert is_prime(2) is True
        assert is_prime(3) is True

        sqrt_mock.assert_not_called()

    def test_returns_early_for_even_and_multiple_of_three(self, monkeypatch):
        '''is_prime() should short-circuit for even numbers and multiples of 3.'''
        sqrt_mock = Mock()
        monkeypatch.setattr(math, 'sqrt', sqrt_mock)

        assert is_prime(4) is False
        assert is_prime(9) is False
        assert is_prime(100) is False

        sqrt_mock.assert_not_called()
