"""Tests for the sigmoid function implemented in maths/sigmoid.py."""

import math

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Positive test cases for the sigmoid function."""

    def test_sigmoid_matches_documented_doctest_example(self):
        """The documented doctest example must produce the documented output."""
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_sigmoid_of_zero_is_half(self):
        """sigmoid(0) must equal exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize(
        "value, expected",
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
    def test_sigmoid_scalar_values_match_math_exp(self, value, expected):
        """Each element must equal 1 / (1 + exp(-x)) computed with math.exp."""
        result = sigmoid(np.array([value]))
        assert result[0] == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_sigmoid_preserves_input_shape(self):
        """The output array must have the same shape as the input array."""
        vector = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        result = sigmoid(vector)
        assert result.shape == vector.shape

    def test_sigmoid_returns_ndarray(self):
        """The function must return a numpy ndarray."""
        result = sigmoid(np.array([0.5, 1.5]))
        assert isinstance(result, np.ndarray)

    def test_sigmoid_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-1.0, 0.0, 1.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    def test_sigmoid_is_symmetric_around_origin(self):
        """sigmoid(x) + sigmoid(-x) must equal 1 for any x."""
        vector = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(
            sigmoid(vector) + sigmoid(-vector), np.ones_like(vector), rtol=1e-12
        )

    def test_sigmoid_output_is_strictly_increasing(self):
        """The sigmoid function must be monotonically increasing."""
        vector = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)


class TestSigmoidEdgeCases:
    """Boundary and edge case tests for the sigmoid function."""

    def test_sigmoid_empty_array_returns_empty_array(self):
        """An empty input array must yield an empty output array."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to 1.0 without overflow."""
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-12)
        assert result[0] <= 1.0

    def test_sigmoid_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to 0.0 without overflow."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)
        assert result[0] >= 0.0

    def test_sigmoid_output_always_within_unit_interval(self):
        """All outputs must lie in the closed interval [0, 1]."""
        vector = np.array([-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6])
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_accepts_float_array_with_multiple_elements(self):
        """A float array with several elements must be handled element-wise."""
        result = sigmoid(np.array([0.0, 1.0]))
        np.testing.assert_allclose(
            result, np.array([0.5, 1.0 / (1.0 + math.exp(-1.0))])
        )

    def test_sigmoid_accepts_integer_array(self):
        """An integer array must be handled correctly."""
        result = sigmoid(np.array([0, 1, -1]))
        expected = np.array([0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sigmoid_handles_nan_input(self):
        """NaN inputs must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_handles_infinite_input(self):
        """Positive and negative infinity must map to 1.0 and 0.0 respectively."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]))

    def test_sigmoid_single_element_2d_array(self):
        """A 2D array with a single element must keep its 2D shape."""
        result = sigmoid(np.array([[0.0]]))
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.5, abs=1e-12)
