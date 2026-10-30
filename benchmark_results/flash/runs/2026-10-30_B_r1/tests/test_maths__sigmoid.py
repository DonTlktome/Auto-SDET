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
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_sigmoid_of_zero_is_half(self):
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([0.5]), rtol=1e-12)

    def test_sigmoid_returns_ndarray(self):
        """The function must return a numpy array, not a scalar or list."""
        result = sigmoid(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_sigmoid_preserves_shape(self):
        """Output shape must match the input shape."""
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert sigmoid(vector).shape == vector.shape

    def test_sigmoid_does_not_mutate_input(self):
        """The input array must be left untouched."""
        vector = np.array([-2.0, 0.0, 2.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    def test_sigmoid_matches_scalar_reference_implementation(self):
        """Each element must equal 1 / (1 + exp(-x)) computed elementwise."""
        vector = np.array([-3.5, -0.25, 0.0, 0.75, 4.2])
        expected = np.array([1.0 / (1.0 + math.exp(-x)) for x in vector])
        np.testing.assert_allclose(sigmoid(vector), expected, rtol=1e-12)


class TestSigmoidEdgeCases:
    """Boundary and edge case tests for the sigmoid function."""

    def test_sigmoid_empty_array_returns_empty_array(self):
        """An empty input must yield an empty output of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_single_element_array(self):
        """A one-element array must return a one-element array."""
        result = sigmoid(np.array([2.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result, np.array([0.8807970779778823]), rtol=1e-12)

    def test_sigmoid_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to 1.0 without overflow."""
        result = sigmoid(np.array([1000.0]))
        np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-12)

    def test_sigmoid_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to 0.0 without overflow."""
        result = sigmoid(np.array([-1000.0]))
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-12)

    def test_sigmoid_output_is_bounded_between_zero_and_one(self):
        """All outputs must lie strictly within (0, 1) in the float-safe range.

        Note: for |x| beyond roughly 37, sigmoid(x) rounds to exactly 0.0 or
        1.0 in float64, so strict inequalities can only be asserted while
        exp(-x) remains representably distinct from 1.0 / 0.0.
        """
        vector = np.linspace(-20.0, 20.0, 101)
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_sigmoid_saturates_within_unit_interval_for_extreme_inputs(self):
        """For extreme inputs the output must still stay within [0, 1]."""
        vector = np.linspace(-50.0, 50.0, 101)
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_is_symmetric_around_origin(self):
        """sigmoid(-x) must equal 1 - sigmoid(x)."""
        vector = np.array([0.5, 1.5, 3.0, 7.25])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12
        )

    def test_sigmoid_is_monotonically_increasing(self):
        """The output must be strictly increasing for increasing inputs."""
        vector = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0.0)

    def test_sigmoid_handles_two_dimensional_input(self):
        """A 2-D array must be transformed elementwise, preserving shape."""
        vector = np.array([[-1.0, 0.0], [1.0, 2.0]])
        result = sigmoid(vector)
        assert result.shape == (2, 2)
        expected = np.array(
            [
                [1.0 / (1.0 + math.exp(1.0)), 0.5],
                [1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(-2.0))],
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sigmoid_handles_integer_input(self):
        """Integer arrays must be accepted and produce float results."""
        result = sigmoid(np.array([0, 1, -1]))
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(
            result,
            np.array([0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))]),
            rtol=1e-12,
        )

    def test_sigmoid_propagates_nan(self):
        """NaN inputs must propagate as NaN outputs."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_propagates_positive_infinity(self):
        """Positive infinity must map to 1.0."""
        result = sigmoid(np.array([np.inf]))
        np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-12)

    def test_sigmoid_propagates_negative_infinity(self):
        """Negative infinity must map to 0.0."""
        result = sigmoid(np.array([-np.inf]))
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-12)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, 0.5),
        (1.0, 0.7310585786300049),
        (-1.0, 0.2689414213699951),
        (2.0, 0.8807970779778823),
        (-2.0, 0.11920292202211755),
        (10.0, 0.9999546021312976),
        (-10.0, 4.5397868702434395e-05),
    ],
)
def test_sigmoid_parametrized_scalar_values(value, expected):
    """Parametrized check of sigmoid against known reference values."""
    result = sigmoid(np.array([value]))
    np.testing.assert_allclose(result, np.array([expected]), rtol=1e-12)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([0.0]),
        np.array([-1.0, 1.0, 2.0]),
        np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
        np.linspace(-5.0, 5.0, 11),
        np.zeros(4),
        np.ones(3),
    ],
)
def test_sigmoid_output_always_in_unit_interval(vector):
    """For any finite input, every output element must be in [0, 1]."""
    result = sigmoid(vector)
    assert result.shape == vector.shape
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
