"""Tests for the sigmoid implementation in maths/sigmoid.py."""

import math

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Positive test cases for the sigmoid function."""

    def test_doctest_example_values(self):
        """The documented example must produce the documented output."""
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_zero_returns_half(self):
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.5)

    def test_returns_ndarray(self):
        """The function must return a numpy array, not a scalar or list."""
        result = sigmoid(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_preserves_input_shape(self):
        """Output shape must match the input shape."""
        vector = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = sigmoid(vector)
        assert result.shape == vector.shape

    def test_matches_math_formula(self):
        """Each element must equal 1 / (1 + exp(-x)) computed elementwise."""
        vector = np.array([-3.5, -0.25, 0.0, 0.75, 4.2])
        result = sigmoid(vector)
        expected = np.array([1.0 / (1.0 + math.exp(-x)) for x in vector])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-1.0, 0.0, 1.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)


class TestSigmoidProperties:
    """Mathematical properties that the sigmoid function must satisfy."""

    @pytest.mark.parametrize("value", [-10.0, -2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
    def test_output_in_unit_interval(self, value):
        """Output must always lie strictly within (0, 1)."""
        result = sigmoid(np.array([value]))
        assert 0.0 < result[0] < 1.0

    @pytest.mark.parametrize("value", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_symmetry_identity(self, value):
        """sigmoid(x) + sigmoid(-x) must equal 1."""
        positive = sigmoid(np.array([value]))[0]
        negative = sigmoid(np.array([-value]))[0]
        assert positive + negative == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        """Larger inputs must produce larger outputs."""
        vector = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)

    def test_large_positive_saturates_near_one(self):
        """Very large positive inputs must approach 1."""
        result = sigmoid(np.array([100.0]))
        assert result[0] == pytest.approx(1.0)

    def test_large_negative_saturates_near_zero(self):
        """Very large negative inputs must approach 0."""
        result = sigmoid(np.array([-100.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)


class TestSigmoidEdgeCases:
    """Boundary and edge-case behaviour."""

    def test_empty_array_returns_empty_array(self):
        """An empty input must yield an empty output of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_single_element_array(self):
        """A one-element array must return a one-element array."""
        result = sigmoid(np.array([3.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))

    def test_integer_dtype_input(self):
        """Integer input arrays must be handled and produce float output."""
        result = sigmoid(np.array([0, 1, -1]))
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(
            result,
            np.array([0.5, 1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))]),
            rtol=1e-12,
        )

    def test_very_large_negative_does_not_raise(self):
        """Extreme negative values must not raise an overflow error."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_very_large_positive_does_not_raise(self):
        """Extreme positive values must not raise an overflow error."""
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0)

    def test_nan_propagates(self):
        """NaN input must propagate to NaN output."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_positive_infinity_maps_to_one(self):
        """Positive infinity must map to 1."""
        result = sigmoid(np.array([np.inf]))
        assert result[0] == pytest.approx(1.0)

    def test_negative_infinity_maps_to_zero(self):
        """Negative infinity must map to 0."""
        result = sigmoid(np.array([-np.inf]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_list_input_raises_attribute_error(self):
        """A plain Python list is not supported and must raise TypeError."""
        with pytest.raises(TypeError):
            sigmoid([1.0, 2.0, 3.0])

    def test_none_input_raises(self):
        """None is not a valid input and must raise an error."""
        with pytest.raises((TypeError, AttributeError)):
            sigmoid(None)
