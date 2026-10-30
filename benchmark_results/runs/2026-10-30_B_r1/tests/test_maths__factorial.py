'''
Pytest test suite for the factorial module.

Covers the iterative ``factorial`` and recursive ``factorial_recursive``
functions, including happy paths, edge cases, and error handling.
'''

import math

import pytest

from maths.factorial import factorial, factorial_recursive


# ---------------------------------------------------------------------------
# factorial (iterative)
# ---------------------------------------------------------------------------

class TestFactorial:
    '''Tests for the iterative ``factorial`` function.'''

    @pytest.mark.parametrize(
        ('number', 'expected'),
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
    def test_factorial_positive_values(self, number, expected):
        '''Verify factorial returns correct results for non-negative integers.'''
        assert factorial(number) == expected

    def test_factorial_matches_math_factorial(self):
        '''Verify factorial agrees with math.factorial across a range.'''
        assert all(factorial(i) == math.factorial(i) for i in range(20))

    def test_factorial_zero(self):
        '''Verify factorial(0) returns 1 (empty product).'''
        assert factorial(0) == 1

    def test_factorial_one(self):
        '''Verify factorial(1) returns 1 (base case).'''
        assert factorial(1) == 1

    def test_factorial_large_value(self):
        '''Verify factorial handles a moderately large input correctly.'''
        assert factorial(20) == math.factorial(20)

    @pytest.mark.parametrize('number', [-1, -5, -100])
    def test_factorial_negative_raises(self, number):
        '''Verify factorial raises ValueError for negative inputs.'''
        with pytest.raises(ValueError, match='not defined for negative values'):
            factorial(number)

    @pytest.mark.parametrize('number', [0.1, 1.5, 2.999])
    def test_factorial_non_integral_raises(self, number):
        '''Verify factorial raises ValueError for non-integral inputs.'''
        with pytest.raises(ValueError, match='only accepts integral values'):
            factorial(number)

    def test_factorial_float_that_is_integral(self):
        '''Verify factorial raises TypeError for an integral float.'''
        with pytest.raises(TypeError):
            factorial(5.0)


# ---------------------------------------------------------------------------
# factorial_recursive
# ---------------------------------------------------------------------------

class TestFactorialRecursive:
    '''Tests for the recursive ``factorial_recursive`` function.'''

    @pytest.mark.parametrize(
        ('number', 'expected'),
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
    def test_factorial_recursive_positive_values(self, number, expected):
        '''Verify recursive factorial returns correct results for integers.'''
        assert factorial_recursive(number) == expected

    def test_factorial_recursive_matches_math_factorial(self):
        '''Verify recursive factorial agrees with math.factorial across a range.'''
        assert all(factorial_recursive(i) == math.factorial(i) for i in range(20))

    def test_factorial_recursive_zero(self):
        '''Verify recursive factorial(0) returns 1 (base case).'''
        assert factorial_recursive(0) == 1

    def test_factorial_recursive_one(self):
        '''Verify recursive factorial(1) returns 1 (base case).'''
        assert factorial_recursive(1) == 1

    def test_factorial_recursive_large_value(self):
        '''Verify recursive factorial handles a moderately large input correctly.'''
        assert factorial_recursive(20) == math.factorial(20)

    @pytest.mark.parametrize('number', [-1, -5, -100])
    def test_factorial_recursive_negative_raises(self, number):
        '''Verify recursive factorial raises ValueError for negative inputs.'''
        with pytest.raises(ValueError, match='not defined for negative values'):
            factorial_recursive(number)

    @pytest.mark.parametrize('number', [0.1, 1.5, 3.0, 2.999])
    def test_factorial_recursive_non_integral_raises(self, number):
        '''Verify recursive factorial raises ValueError for non-integer inputs.'''
        with pytest.raises(ValueError, match='only accepts integral values'):
            factorial_recursive(number)

    def test_factorial_recursive_rejects_float_integral(self):
        '''Verify recursive factorial rejects floats even if mathematically integral.'''
        with pytest.raises(ValueError, match='only accepts integral values'):
            factorial_recursive(5.0)

    def test_factorial_recursive_rejects_bool(self):
        '''Verify recursive factorial treats bool as int and returns 1.'''
        assert factorial_recursive(True) == 1
