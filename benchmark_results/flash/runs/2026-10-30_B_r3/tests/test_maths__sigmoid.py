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


class TestSigmoidEdgeCases:
    """Boundary and edge case tests for the sigmoid function."""

    def test_empty_array(self):
        """An empty input must yield an empty output of the same shape."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_single_element_array(self):
        """A one-element array must return a one-element array."""
        result = sigmoid(np.array([2.0]))
        assert result.shape == (1,)
        assert result[0] == pytest.approx(0.8807970779778823)

    def test_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate near 1.0 without overflow."""
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0)
        assert result[0] <= 1.0

    def test_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate near 0.0 without overflow."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)
        assert result[0] >= 0.0

    def test_output_range_is_bounded(self):
        """All outputs must lie within [0, 1] for finite inputs."""
        vector = np.linspace(-20.0, 20.0, 101)
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_range_is_bounded_for_extreme_inputs(self):
        """Even far outside the linear region outputs never exceed [0, 1]."""
        vector = np.linspace(-50.0, 50.0, 101)
        result = sigmoid(vector)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetry_property(self):
        """sigmoid(-x) must equal 1 - sigmoid(x)."""
        vector = np.array([0.5, 1.5, 2.5, 7.0])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12
        )

    def test_monotonically_increasing(self):
        """The sigmoid function must be strictly increasing."""
        vector = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, 0.5),
            (1.0, 0.7310585786300049),
            (-1.0, 0.2689414213699951),
            (2.0, 0.8807970779778823),
            (-2.0, 0.11920292202211755),
        ],
    )
    def test_known_values(self, value, expected):
        """Parametrized check of well-known sigmoid values."""
        result = sigmoid(np.array([value]))
        assert result[0] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
    def test_accepts_various_dtypes(self, dtype):
        """Integer and floating point inputs must both be supported."""
        vector = np.array([-1, 0, 1], dtype=dtype)
        result = sigmoid(vector)
        assert result.shape == (3,)
        np.testing.assert_allclose(
            result.astype(np.float64),
            np.array([0.2689414213699951, 0.5, 0.7310585786300049]),
            rtol=1e-6,
        )

    def test_integer_input_returns_float_output(self):
        """Integer input must produce floating point output."""
        result = sigmoid(np.array([1, 2, 3]))
        assert np.issubdtype(result.dtype, np.floating)

    def test_nan_input_propagates_nan(self):
        """NaN input must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_infinite_inputs(self):
        """Positive and negative infinity must map to 1.0 and 0.0."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_list_input_raises_type_error(self):
        """A plain Python list is not supported and must raise TypeError."""
        with pytest.raises(TypeError):
            sigmoid([1.0, 2.0, 3.0])
