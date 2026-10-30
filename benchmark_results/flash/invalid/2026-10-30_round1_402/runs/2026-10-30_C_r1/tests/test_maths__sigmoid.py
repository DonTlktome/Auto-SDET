"""Tests for the sigmoid implementation in maths/sigmoid.py."""

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
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.5)

    def test_sigmoid_returns_ndarray_of_same_shape(self):
        """The output must be a numpy array with the same shape as the input."""
        vector = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = sigmoid(vector)
        assert isinstance(result, np.ndarray)
        assert result.shape == vector.shape

    def test_sigmoid_preserves_input_array(self):
        """The input array must not be mutated by the function."""
        vector = np.array([-2.0, 0.0, 2.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    @pytest.mark.parametrize(
        "value",
        [-10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0],
    )
    def test_sigmoid_matches_math_formula(self, value):
        """Each element must equal 1 / (1 + exp(-x)) computed independently."""
        result = sigmoid(np.array([value]))
        expected = 1.0 / (1.0 + math.exp(-value))
        assert result[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize(
        "vector",
        [
            np.array([-1.0, 1.0, 2.0]),
            np.array([0.0]),
            np.array([1e-8, -1e-8]),
            np.array([5.0, -5.0, 0.0]),
        ],
    )
    def test_sigmoid_output_is_within_unit_interval(self, vector):
        """All sigmoid outputs must lie strictly within (0, 1)."""
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)


class TestSigmoidEdgeCases:
    """Boundary and edge cases for the sigmoid function."""

    def test_sigmoid_of_empty_array_returns_empty_array(self):
        """An empty input must yield an empty output of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_of_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to approximately 1.0."""
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0)

    def test_sigmoid_of_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to approximately 0.0."""
        with np.errstate(over="ignore"):
            result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_sigmoid_is_symmetric_around_zero(self):
        """sigmoid(x) + sigmoid(-x) must equal 1 for any x."""
        values = np.array([0.5, 1.5, 4.0, 12.0])
        positive = sigmoid(values)
        negative = sigmoid(-values)
        np.testing.assert_allclose(positive + negative, np.ones_like(values))

    def test_sigmoid_is_monotonically_increasing(self):
        """The sigmoid function must be strictly increasing over sorted input."""
        vector = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)

    def test_sigmoid_accepts_integer_array(self):
        """Integer input arrays must be handled via numpy type promotion."""
        result = sigmoid(np.array([0, 1, -1]))
        expected = sigmoid(np.array([0.0, 1.0, -1.0]))
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_requires_ndarray_but_accepts_converted_list(self):
        """A plain Python list must be converted via np.asarray before calling."""
        with pytest.raises(TypeError):
            sigmoid([0.0, 1.0])
        result = sigmoid(np.asarray([0.0, 1.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(
            result, np.array([0.5, 1.0 / (1.0 + math.exp(-1.0))])
        )

    def test_sigmoid_handles_nan_input(self):
        """NaN inputs must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_handles_infinite_input(self):
        """Infinite inputs must saturate to 1.0 and 0.0 respectively."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]))
