"""Tests for the sigmoid implementation in maths/sigmoid.py."""

import math

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Positive test cases for the sigmoid function."""

    def test_matches_documented_doctest_example(self):
        """The documented example must produce the documented output."""
        result = sigmoid(np.array([-1.0, 1.0, 2.0]))
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_single_zero_returns_half(self):
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result, np.array([0.5]))

    def test_returns_numpy_array(self):
        """The function must return a numpy array, not a scalar or list."""
        result = sigmoid(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_preserves_input_shape(self):
        """Output shape must equal input shape for 1-D and 2-D inputs."""
        one_d = np.array([0.1, 0.2, 0.3, 0.4])
        assert sigmoid(one_d).shape == one_d.shape

        two_d = np.array([[0.1, 0.2], [0.3, 0.4]])
        assert sigmoid(two_d).shape == two_d.shape

    def test_does_not_mutate_input(self):
        """The input array must be left unchanged."""
        vector = np.array([-2.0, 0.0, 3.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    @pytest.mark.parametrize(
        "value",
        [-10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0],
    )
    def test_matches_math_formula(self, value):
        """Each element must equal 1 / (1 + exp(-x)) computed with math.exp."""
        expected = 1.0 / (1.0 + math.exp(-value))
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result, np.array([expected]), rtol=1e-12)

    @pytest.mark.parametrize(
        "vector",
        [
            np.array([-1.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([-5.0, 5.0]),
            np.array([1e-9, -1e-9]),
        ],
    )
    def test_outputs_are_within_unit_interval(self, vector):
        """All sigmoid outputs must lie in the open interval (0, 1)."""
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_is_monotonically_increasing(self):
        """Larger inputs must produce strictly larger outputs."""
        vector = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)

    def test_symmetry_property(self):
        """sigmoid(-x) must equal 1 - sigmoid(x)."""
        vector = np.array([0.5, 1.5, 2.5, 4.0])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12
        )


class TestSigmoidEdgeCases:
    """Boundary and edge-case tests for the sigmoid function."""

    def test_empty_array_returns_empty_array(self):
        """An empty input must yield an empty output of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to approximately 1.0."""
        result = sigmoid(np.array([1000.0]))
        np.testing.assert_allclose(result, np.array([1.0]))

    def test_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to approximately 0.0."""
        result = sigmoid(np.array([-1000.0]))
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-15)

    def test_extreme_negative_does_not_raise_overflow(self):
        """Extremely negative inputs must not raise an overflow error."""
        result = sigmoid(np.array([-1e4]))
        assert np.isfinite(result).all()
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_integer_dtype_input(self):
        """Integer arrays must be accepted and produce float results."""
        result = sigmoid(np.array([0, 1, -1]))
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(
            result,
            np.array([0.5, 1.0 / (1.0 + math.exp(-1)), 1.0 / (1.0 + math.exp(1))]),
            rtol=1e-12,
        )

    def test_two_dimensional_input(self):
        """2-D inputs must be transformed element-wise."""
        vector = np.array([[0.0, 1.0], [-1.0, 2.0]])
        result = sigmoid(vector)
        expected = 1.0 / (1.0 + np.exp(-vector))
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_nan_input_propagates_nan(self):
        """NaN inputs must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_positive_and_negative_infinity(self):
        """+inf maps to 1.0 and -inf maps to 0.0."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]))

    def test_large_array_vectorized(self):
        """A large array must be processed element-wise correctly."""
        # Use an odd number of points so the midpoint is exactly 0.0.
        vector = np.linspace(-50.0, 50.0, 1001)
        result = sigmoid(vector)
        assert result.shape == vector.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert vector[500] == 0.0
        np.testing.assert_allclose(result[500], 0.5, atol=1e-12)
