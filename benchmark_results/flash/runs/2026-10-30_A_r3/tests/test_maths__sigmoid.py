"""Tests for the sigmoid implementation in maths/sigmoid.py."""

import math

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Positive test cases for the sigmoid function."""

    def test_sigmoid_matches_documented_example(self):
        """The docstring example must produce the documented output."""
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_sigmoid_of_zero_is_half(self):
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result, np.array([0.5]), rtol=1e-12)

    def test_sigmoid_preserves_input_shape(self):
        """The output array must have the same shape as the input array."""
        vector = np.array([[0.0, 1.0, 2.0], [-1.0, -2.0, -3.0]])
        result = sigmoid(vector)
        assert result.shape == vector.shape

    def test_sigmoid_returns_float_array(self):
        """The output dtype must be floating point."""
        result = sigmoid(np.array([1, 2, 3]))
        assert np.issubdtype(result.dtype, np.floating)

    def test_sigmoid_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-1.0, 0.0, 1.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)


class TestSigmoidProperties:
    """Mathematical properties that the sigmoid function must satisfy."""

    @pytest.mark.parametrize("value", [-10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0])
    def test_sigmoid_matches_math_formula(self, value):
        """Each element must equal 1 / (1 + exp(-x)) computed with math.exp."""
        expected = 1.0 / (1.0 + math.exp(-value))
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    @pytest.mark.parametrize("value", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_sigmoid_symmetry(self, value):
        """sigmoid(x) + sigmoid(-x) must equal 1."""
        positive = sigmoid(np.array([value]))[0]
        negative = sigmoid(np.array([-value]))[0]
        np.testing.assert_allclose(positive + negative, 1.0, rtol=1e-12)

    @pytest.mark.parametrize("value", [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    def test_sigmoid_output_within_unit_interval(self, value):
        """Outputs must stay inside the unit interval [0, 1].

        For very large magnitudes the gap to the asymptote is not
        representable in float64 (exp(-100) ~ 3.7e-44 vanishes when added
        to 1.0), so the bounds are inclusive at those extremes and only
        strictly open for moderate inputs.
        """
        result = sigmoid(np.array([value]))[0]
        assert 0.0 <= result <= 1.0
        if abs(value) <= 10.0:
            assert 0.0 < result < 1.0

    def test_sigmoid_is_monotonically_increasing(self):
        """Larger inputs must produce larger outputs."""
        vector = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)


class TestSigmoidEdgeCases:
    """Boundary and edge case tests for the sigmoid function."""

    def test_sigmoid_empty_array(self):
        """An empty input must return an empty array of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_single_element_list_like(self):
        """A one-element array must return a one-element array."""
        result = sigmoid(np.array([2.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result, np.array([0.8807970779778823]), rtol=1e-12)

    def test_sigmoid_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to approximately 1.0."""
        result = sigmoid(np.array([1000.0]))[0]
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_sigmoid_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to approximately 0.0."""
        result = sigmoid(np.array([-1000.0]))[0]
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_sigmoid_handles_infinities(self):
        """Positive infinity maps to 1.0 and negative infinity maps to 0.0."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]), rtol=1e-12)

    def test_sigmoid_nan_input_propagates_nan(self):
        """A NaN input must produce a NaN output."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_integer_input(self):
        """Integer arrays must be accepted and produce float results."""
        result = sigmoid(np.array([0, 1, -1]))
        expected = np.array([0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sigmoid_large_array(self):
        """A large array must be processed element-wise without error."""
        # An odd number of samples places an exact 0.0 at the midpoint.
        vector = np.linspace(-50.0, 50.0, 1001)
        result = sigmoid(vector)
        assert result.shape == vector.shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        assert vector[500] == 0.0
        np.testing.assert_allclose(result[500], 0.5, rtol=1e-12)
