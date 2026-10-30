"""Tests for the sigmoid function in maths/sigmoid.py."""

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Positive test cases for the sigmoid function."""

    def test_known_values(self):
        """Sigmoid of a mixed vector should match the documented example."""
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_allclose(result, expected, rtol=1e-7)

    def test_zero_returns_half(self):
        """Sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.array([0.5]))

    def test_returns_ndarray(self):
        """The function should return a numpy ndarray."""
        result = sigmoid(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, np.ndarray)

    def test_preserves_shape(self):
        """Output shape must equal input shape."""
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert sigmoid(vector).shape == vector.shape

    def test_2d_input_shape_preserved(self):
        """A 2-D input of shape (1, n) should yield a (1, n) output."""
        vector = np.array([[0.0, 1.0, -1.0]])
        result = sigmoid(vector)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result, np.array([[0.5, 0.73105858, 0.26894142]]), rtol=1e-7)

    def test_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-2.0, 0.0, 2.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)


class TestSigmoidEdgeCases:
    """Boundary and edge cases for the sigmoid function."""

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
    def test_scalar_values(self, value, expected):
        """Sigmoid of a single-element array should match the analytic value."""
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result, np.array([expected]), rtol=1e-9)

    def test_empty_array(self):
        """An empty input should produce an empty output."""
        result = sigmoid(np.array([]))
        assert result.shape == (0,)

    def test_large_positive_saturates_to_one(self):
        """Very large positive inputs should saturate near 1.0."""
        result = sigmoid(np.array([1000.0]))
        np.testing.assert_allclose(result, np.array([1.0]))

    def test_large_negative_underflows_to_zero(self):
        """Very large negative inputs should underflow to 0.0 (no exception)."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_output_range_bounded(self):
        """All outputs must lie strictly within [0, 1]."""
        vector = np.linspace(-50.0, 50.0, 101)
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_monotonic_increasing(self):
        """Sigmoid must be monotonically increasing."""
        vector = np.linspace(-10.0, 10.0, 200)
        result = sigmoid(vector)
        assert np.all(np.diff(result) >= 0.0)

    def test_symmetry_property(self):
        """sigmoid(-x) == 1 - sigmoid(x) for all x."""
        vector = np.array([-3.0, -1.5, -0.5, 0.5, 1.5, 3.0])
        np.testing.assert_allclose(sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12)

    def test_integer_input(self):
        """Integer arrays should be handled via numpy type promotion."""
        result = sigmoid(np.array([0, 1, -1]))
        expected = np.array([0.5, 0.7310585786300049, 0.2689414213699951])
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_list_input_rejected(self):
        """A plain Python list is not a supported input type for sigmoid()."""
        with pytest.raises(TypeError):
            sigmoid([0.0, 1.0])

    def test_list_converted_to_array(self):
        """Callers must convert lists to numpy arrays before calling sigmoid()."""
        result = sigmoid(np.array([0.0, 1.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.7310585786300049]), rtol=1e-9)

    def test_nan_propagates(self):
        """NaN input should propagate to NaN output."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_infinity_handling(self):
        """Positive infinity maps to 1.0 and negative infinity to 0.0."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]))
