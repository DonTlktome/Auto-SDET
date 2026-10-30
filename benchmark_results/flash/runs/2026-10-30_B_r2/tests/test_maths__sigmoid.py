'''Tests for the sigmoid implementation in maths/sigmoid.py.'''

import math

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    '''Positive test cases for the sigmoid function.'''

    def test_matches_documented_doctest_example(self):
        '''The documented example must produce the documented output.'''
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_single_zero_returns_half(self):
        '''sigmoid(0) must be exactly 0.5.'''
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result, np.array([0.5]), rtol=1e-12)

    @pytest.mark.parametrize(
        'value, expected',
        [
            (0.0, 0.5),
            (1.0, 1.0 / (1.0 + math.exp(-1.0))),
            (-1.0, 1.0 / (1.0 + math.exp(1.0))),
            (2.0, 1.0 / (1.0 + math.exp(-2.0))),
            (-2.0, 1.0 / (1.0 + math.exp(2.0))),
            (10.0, 1.0 / (1.0 + math.exp(-10.0))),
            (-10.0, 1.0 / (1.0 + math.exp(10.0))),
        ],
    )
    def test_scalar_values_match_math_formula(self, value, expected):
        '''Each scalar input must match the analytic 1/(1+exp(-x)) formula.'''
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result, np.array([expected]), rtol=1e-12)

    def test_returns_numpy_array(self):
        '''The return type must be a numpy ndarray.'''
        result = sigmoid(np.array([0.5, 1.5]))
        assert isinstance(result, np.ndarray)

    def test_preserves_input_shape(self):
        '''Output shape must equal input shape.'''
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert sigmoid(vector).shape == vector.shape

    def test_does_not_mutate_input(self):
        '''The input array must not be modified in place.'''
        vector = np.array([-1.0, 0.0, 1.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    def test_two_dimensional_input(self):
        '''A 2-D array of shape (1, n) must be supported element-wise.'''
        vector = np.array([[0.0, 1.0, -1.0]])
        result = sigmoid(vector)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(
            result,
            np.array([[0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))]]),
            rtol=1e-12,
        )

    def test_output_range_is_between_zero_and_one(self):
        '''All outputs must lie within [0, 1] for finite inputs.

        Extreme finite inputs saturate to exactly 0.0 or 1.0 because of
        floating-point rounding.
        '''
        vector = np.linspace(-50.0, 50.0, 101)
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        # Strict interior for moderate inputs.
        moderate = sigmoid(np.linspace(-20.0, 20.0, 101))
        assert np.all(moderate > 0.0)
        assert np.all(moderate < 1.0)

    def test_monotonically_increasing(self):
        '''The sigmoid must be strictly increasing over finite inputs.'''
        vector = np.linspace(-20.0, 20.0, 200)
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0.0)

    def test_symmetry_property(self):
        '''sigmoid(-x) must equal 1 - sigmoid(x).'''
        vector = np.array([0.3, 1.7, 4.2, 9.9])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-9, atol=1e-12
        )


class TestSigmoidEdgeCases:
    '''Boundary and edge case tests for the sigmoid function.'''

    def test_empty_array_returns_empty_array(self):
        '''An empty input must yield an empty output without raising.'''
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_large_positive_input_saturates_to_one(self):
        '''Very large positive inputs must saturate to 1.0 without overflow.'''
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0)

    def test_large_negative_input_saturates_to_zero(self):
        '''Very large negative inputs must saturate to 0.0 without overflow.'''
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_positive_infinity_returns_one(self):
        '''Positive infinity must map to 1.0.'''
        result = sigmoid(np.array([np.inf]))
        assert result[0] == pytest.approx(1.0)

    def test_negative_infinity_returns_zero(self):
        '''Negative infinity must map to 0.0.'''
        result = sigmoid(np.array([-np.inf]))
        assert result[0] == pytest.approx(0.0)

    def test_nan_input_propagates_nan(self):
        '''NaN input must propagate as NaN output.'''
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_integer_dtype_input(self):
        '''Integer arrays must be accepted and produce float results.'''
        result = sigmoid(np.array([0, 1, -1]))
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(
            result,
            np.array([0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))]),
            rtol=1e-12,
        )

    def test_array_converted_from_list_is_supported(self):
        '''A list converted to a numpy array must work.'''
        result = sigmoid(np.asarray([0.0, 1.0]))
        np.testing.assert_allclose(
            result, np.array([0.5, 1.0 / (1.0 + math.exp(-1.0))]), rtol=1e-12
        )

    def test_very_small_positive_input(self):
        '''Tiny positive inputs must yield values just above 0.5.'''
        result = sigmoid(np.array([1e-12]))
        assert result[0] > 0.5
        assert result[0] == pytest.approx(0.5 + 2.5e-13, rel=1e-6)

    def test_very_small_negative_input(self):
        '''Tiny negative inputs must yield values just below 0.5.'''
        result = sigmoid(np.array([-1e-12]))
        assert result[0] < 0.5
        assert result[0] == pytest.approx(0.5 - 2.5e-13, rel=1e-6)

    def test_large_batch_of_values(self):
        '''A large batch must be processed element-wise correctly.'''
        vector = np.linspace(-100.0, 100.0, 1000)
        result = sigmoid(vector)
        assert result.shape == vector.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
