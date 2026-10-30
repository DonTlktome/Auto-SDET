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
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        result = sigmoid(vector)
        assert isinstance(result, np.ndarray)
        assert result.shape == vector.shape

    def test_sigmoid_preserves_elementwise_order(self):
        """Sigmoid is monotonically increasing, so ordering must be preserved."""
        vector = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0)

    def test_sigmoid_matches_scalar_math_formula(self):
        """Each element must equal 1 / (1 + exp(-x)) computed with math.exp."""
        vector = np.array([-2.5, -0.5, 0.0, 0.5, 2.5])
        result = sigmoid(vector)
        expected = np.array([1.0 / (1.0 + math.exp(-x)) for x in vector])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_sigmoid_accepts_list_converted_to_array(self):
        """A Python list works once converted to a numpy array.

        The implementation is typed/documented to take an ``np.ndarray`` and
        performs array arithmetic directly, so plain lists must be converted
        explicitly by the caller.  Converting the list here checks that the
        function then behaves exactly as it does for any other array input.
        """
        result = sigmoid(np.array([0.0, 1.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.7310585786300049]))

    def test_sigmoid_requires_ndarray_input(self):
        """A plain Python list is not coerced and must raise TypeError."""
        with pytest.raises(TypeError):
            sigmoid([0.0, 1.0])

    def test_sigmoid_accepts_integer_array(self):
        """An integer array must be handled (implicit float conversion)."""
        result = sigmoid(np.array([0, 1, 2]))
        expected = np.array([0.5, 0.7310585786300049, 0.8807970779778823])
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestSigmoidEdgeCases:
    """Boundary and edge cases for the sigmoid function."""

    def test_sigmoid_of_empty_array_returns_empty_array(self):
        """An empty input must yield an empty output without raising."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_of_single_element_2d_array(self):
        """A 2-D input of shape (1, n) must keep its shape."""
        vector = np.array([[-1.0, 0.0, 1.0]])
        result = sigmoid(vector)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(
            result, np.array([[0.2689414213699951, 0.5, 0.7310585786300049]])
        )

    def test_sigmoid_output_is_bounded_between_zero_and_one(self):
        """All outputs must lie within [0, 1] for finite inputs.

        The bounds are inclusive: for large-magnitude inputs the exponential
        saturates/underflows in floating point, so sigmoid(x) can be exactly
        0.0 or exactly 1.0.
        """
        vector = np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_stays_strictly_inside_bounds_for_moderate_inputs(self):
        """For moderate finite inputs the output is strictly within (0, 1)."""
        vector = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_sigmoid_saturates_for_large_positive_input(self):
        """Very large positive inputs must saturate to 1.0."""
        result = sigmoid(np.array([1000.0]))
        np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-12)

    def test_sigmoid_saturates_for_large_negative_input(self):
        """Very large negative inputs must saturate to 0.0."""
        with np.errstate(over="ignore"):
            result = sigmoid(np.array([-1000.0]))
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-12)

    def test_sigmoid_is_symmetric_around_zero(self):
        """sigmoid(-x) must equal 1 - sigmoid(x)."""
        vector = np.array([0.5, 1.5, 4.0])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12
        )

    def test_sigmoid_does_not_mutate_input(self):
        """The input array must not be modified in place."""
        vector = np.array([-1.0, 0.0, 1.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

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
    def test_sigmoid_scalar_values(self, value, expected):
        """Parametrized check of sigmoid against known scalar values."""
        result = sigmoid(np.array([value]))
        np.testing.assert_allclose(result, np.array([expected]), rtol=1e-12)

    @pytest.mark.parametrize("shape", [(1,), (3,), (2, 2), (2, 3, 4)])
    def test_sigmoid_preserves_arbitrary_shapes(self, shape):
        """The output shape must always match the input shape."""
        vector = np.linspace(-2.0, 2.0, num=int(np.prod(shape))).reshape(shape)
        result = sigmoid(vector)
        assert result.shape == shape
        assert np.all(np.isfinite(result))
