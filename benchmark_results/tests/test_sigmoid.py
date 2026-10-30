"""
Test suite for the sigmoid function in maths/sigmoid.py.

Covers:
- Positive / happy-path tests
- Edge cases (zero, large values, empty array, multi-dimensional arrays)
- Boundary / numerical stability checks
- Input type and shape validation
"""

import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoidHappyPath:
    """Happy-path tests for the sigmoid function."""

    @pytest.mark.parametrize(
        "input_vec, expected",
        [
            (np.array([-1.0, 1.0, 2.0]), np.array([0.26894142, 0.73105858, 0.88079708])),
            (np.array([0.0]), np.array([0.5])),
            (np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
            (np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
             np.array([0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708])),
        ],
        ids=["mixed_signs", "single_zero", "all_zeros", "symmetric_range"],
    )
    def test_sigmoid_known_values(self, input_vec, expected):
        """sigmoid should return the expected values for known inputs."""
        result = sigmoid(input_vec)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_sigmoid_output_range(self):
        """Every output element must be strictly between 0 and 1."""
        vec = np.random.default_rng(42).uniform(-10, 10, size=100)
        result = sigmoid(vec)
        assert np.all(result > 0), "All outputs should be > 0"
        assert np.all(result < 1), "All outputs should be < 1"

    def test_sigmoid_monotonic(self):
        """sigmoid is monotonically increasing: x1 < x2 => sigmoid(x1) < sigmoid(x2)."""
        vec = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
        result = sigmoid(vec)
        assert np.all(np.diff(result) > 0), "sigmoid must be monotonically increasing"


class TestSigmoidEdgeCases:
    """Edge-case and boundary tests for the sigmoid function."""

    def test_sigmoid_empty_array(self):
        """sigmoid on an empty array should return an empty array."""
        vec = np.array([])
        result = sigmoid(vec)
        assert result.size == 0
        assert isinstance(result, np.ndarray)

    def test_sigmoid_single_element(self):
        """sigmoid on a single-element array should work correctly."""
        result = sigmoid(np.array([3.0]))
        expected = 1 / (1 + np.exp(-3.0))
        assert np.isclose(result[0], expected)

    @pytest.mark.parametrize(
        "value, expected_bound",
        [
            (100.0, 1.0),    # large positive → ~1
            (-100.0, 0.0),   # large negative → ~0
            (500.0, 1.0),    # very large positive
            (-500.0, 0.0),   # very large negative
        ],
        ids=["large_positive", "large_negative", "very_large_positive", "very_large_negative"],
    )
    def test_sigmoid_large_values(self, value, expected_bound):
        """sigmoid should handle large-magnitude inputs without overflow and saturate correctly."""
        result = sigmoid(np.array([value]))
        assert np.isclose(result[0], expected_bound, atol=1e-10)

    def test_sigmoid_symmetry(self):
        """sigmoid(-x) should equal 1 - sigmoid(x)."""
        x = np.array([0.5, 1.0, 2.0, 3.5])
        result_pos = sigmoid(x)
        result_neg = sigmoid(-x)
        np.testing.assert_array_almost_equal(result_neg, 1.0 - result_pos, decimal=10)

    def test_sigmoid_2d_array(self):
        """sigmoid should work on multi-dimensional arrays (e.g., 2D)."""
        vec = np.array([[-1.0, 0.0], [1.0, 2.0]])
        result = sigmoid(vec)
        assert result.shape == (2, 2)
        expected = 1 / (1 + np.exp(-vec))
        np.testing.assert_array_almost_equal(result, expected, decimal=8)

    def test_sigmoid_3d_array(self):
        """sigmoid should work on 3D arrays."""
        vec = np.ones((2, 3, 4), dtype=float) * 0.0
        result = sigmoid(vec)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_almost_equal(result, np.full((2, 3, 4), 0.5), decimal=8)

    def test_sigmoid_integer_input(self):
        """sigmoid should handle integer input arrays by promoting to float."""
        vec = np.array([-2, -1, 0, 1, 2])
        result = sigmoid(vec)
        expected = 1 / (1 + np.exp(-vec.astype(float)))
        np.testing.assert_array_almost_equal(result, expected, decimal=8)

    def test_sigmoid_identity_at_zero(self):
        """sigmoid(0) must be exactly 0.5."""
        result = sigmoid(np.array([0.0]))
        assert result[0] == 0.5


class TestSigmoidInputValidation:
    """Tests that verify behaviour with invalid or unusual inputs."""

    def test_sigmoid_returns_ndarray(self):
        """sigmoid must always return a numpy ndarray."""
        result = sigmoid(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)

    def test_sigmoid_preserves_shape(self):
        """sigmoid must preserve the shape of the input array."""
        shapes = [(5,), (1, 10), (3, 4), (2, 2, 2)]
        for shape in shapes:
            vec = np.random.default_rng(123).standard_normal(shape)
            result = sigmoid(vec)
            assert result.shape == shape, f"Shape mismatch for input shape {shape}"

    def test_sigmoid_does_not_mutate_input(self):
        """sigmoid must not modify the original input array."""
        vec = np.array([-1.0, 0.0, 1.0])
        original = vec.copy()
        _ = sigmoid(vec)
        np.testing.assert_array_equal(vec, original)