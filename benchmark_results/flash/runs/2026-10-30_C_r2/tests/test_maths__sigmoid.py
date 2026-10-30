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
        np.testing.assert_allclose(result, np.array([0.5]), rtol=1e-12)

    def test_sigmoid_returns_ndarray_of_same_shape(self):
        """The output must be a numpy array with the same shape as the input."""
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = sigmoid(vector)
        assert isinstance(result, np.ndarray)
        assert result.shape == vector.shape

    def test_sigmoid_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-2.0, 0.0, 2.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    def test_sigmoid_2d_input_preserves_shape(self):
        """A 2-D input of shape (1, n) must return a (1, n) array."""
        vector = np.array([[-1.0, 0.0, 1.0]])
        result = sigmoid(vector)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(
            result, np.array([[0.26894142, 0.5, 0.73105858]]), rtol=1e-8
        )


class TestSigmoidProperties:
    """Mathematical properties that the sigmoid function must satisfy."""

    @pytest.mark.parametrize(
        "value",
        [-10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0],
    )
    def test_sigmoid_matches_math_formula(self, value):
        """Each element must equal 1 / (1 + exp(-x)) computed with math.exp."""
        expected = 1.0 / (1.0 + math.exp(-value))
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "value",
        [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0],
    )
    def test_sigmoid_symmetry_identity(self, value):
        """sigmoid(x) + sigmoid(-x) must equal 1 for all finite x."""
        positive = sigmoid(np.array([value]))[0]
        negative = sigmoid(np.array([-value]))[0]
        np.testing.assert_allclose(positive + negative, 1.0, rtol=1e-12)

    @pytest.mark.parametrize(
        "vector",
        [
            np.array([-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0]),
            np.linspace(-20.0, 20.0, 101),
            np.array([1e-9, -1e-9, 1e-3, -1e-3]),
        ],
    )
    def test_sigmoid_output_is_between_zero_and_one(self, vector):
        """All outputs must lie in the closed interval [0, 1].

        The implementation is the plain ``1 / (1 + exp(-x))`` expression, which
        saturates to exactly 0.0 or 1.0 in floating point for large ``|x|``
        (for example x = 100.0), so the interval is closed rather than open.
        """
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @pytest.mark.parametrize(
        "vector",
        [
            np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]),
            np.linspace(-20.0, 20.0, 101),
            np.array([1e-9, -1e-9, 1e-3, -1e-3]),
        ],
    )
    def test_sigmoid_output_is_strictly_between_zero_and_one(self, vector):
        """For non-saturating inputs all outputs lie in the open interval (0, 1)."""
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_sigmoid_is_monotonically_increasing(self):
        """Larger inputs must produce larger (or equal) outputs."""
        vector = np.linspace(-10.0, 10.0, 200)
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0.0)


class TestSigmoidEdgeCases:
    """Boundary and edge-case behaviour."""

    def test_sigmoid_empty_array_returns_empty_array(self):
        """An empty input must return an empty array without raising."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_single_element_array(self):
        """A one-element array must return a one-element array."""
        result = sigmoid(np.array([2.0]))
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 0.8807970779778823, rtol=1e-12)

    def test_sigmoid_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to approximately 1.0."""
        result = sigmoid(np.array([1000.0]))
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_sigmoid_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to approximately 0.0."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_sigmoid_accepts_integer_array(self):
        """An integer dtype array must be handled and yield float results."""
        result = sigmoid(np.array([0, 1, -1]))
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_allclose(
            result, np.array([0.5, 0.73105858, 0.26894142]), rtol=1e-8
        )

    def test_sigmoid_accepts_numpy_array_from_python_list(self):
        """A plain Python list is accepted once coerced with np.asarray."""
        vector = np.asarray([0.0, 1.0])
        result = sigmoid(vector)
        np.testing.assert_allclose(
            result, np.array([0.5, 0.73105858]), rtol=1e-8
        )

    def test_sigmoid_raw_python_list_raises_type_error(self):
        """The source has no list coercion, so a bare list raises TypeError."""
        with pytest.raises(TypeError):
            sigmoid([0.0, 1.0])

    def test_sigmoid_accepts_scalar(self):
        """A scalar input must return a scalar-like result equal to 0.5 at 0."""
        result = sigmoid(0.0)
        assert float(result) == pytest.approx(0.5, rel=1e-12)

    def test_sigmoid_with_nan_propagates_nan(self):
        """NaN input must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_with_positive_infinity_returns_one(self):
        """Positive infinity must map to 1.0."""
        result = sigmoid(np.array([np.inf]))
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)

    def test_sigmoid_with_negative_infinity_returns_zero(self):
        """Negative infinity must map to 0.0."""
        result = sigmoid(np.array([-np.inf]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)
