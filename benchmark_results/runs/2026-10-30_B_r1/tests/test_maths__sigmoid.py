import numpy as np
import pytest

from sigmoid import sigmoid


class TestSigmoid:
    """Test suite for the sigmoid function."""

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            (0.0, 0.5),
            (1.0, 0.7310585786300049),
            (-1.0, 0.2689414213699951),
            (2.0, 0.8807970779778823),
            (-2.0, 0.11920292202211755),
        ],
    )
    def test_sigmoid_scalar_values(self, input_value, expected):
        """Test sigmoid against known scalar reference values."""
        result = sigmoid(np.array([input_value]))
        assert result.shape == (1,)
        assert np.isclose(result[0], expected)

    def test_sigmoid_vector_happy_path(self):
        """Test sigmoid on a vector of mixed real values."""
        vector = np.array([-1.0, 1.0, 2.0])
        expected = np.array([0.26894142, 0.73105858, 0.88079708])
        result = sigmoid(vector)
        assert result.shape == vector.shape
        assert np.allclose(result, expected)

    def test_sigmoid_zero_vector(self):
        """Test sigmoid on a vector of zeros returns 0.5 for each element."""
        vector = np.zeros(5)
        result = sigmoid(vector)
        assert np.allclose(result, 0.5)

    def test_sigmoid_large_positive_value(self):
        """Test sigmoid saturates towards 1 for large positive inputs."""
        result = sigmoid(np.array([1000.0]))
        assert np.isclose(result[0], 1.0)

    def test_sigmoid_large_negative_value(self):
        """Test sigmoid saturates towards 0 for large negative inputs."""
        result = sigmoid(np.array([-1000.0]))
        assert np.isclose(result[0], 0.0)

    def test_sigmoid_output_range(self):
        """Test sigmoid output is always between 0 and 1."""
        vector = np.linspace(-50.0, 50.0, 1000)
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_symmetry_property(self):
        """Test the symmetry property sigmoid(-x) = 1 - sigmoid(x)."""
        vector = np.array([0.5, 1.5, -3.0, 7.0])
        result = sigmoid(vector)
        negated = sigmoid(-vector)
        assert np.allclose(result, 1.0 - negated)

    def test_sigmoid_preserves_shape(self):
        """Test sigmoid preserves the shape of the input array."""
        vector = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = sigmoid(vector)
        assert result.shape == vector.shape

    def test_sigmoid_empty_array(self):
        """Test sigmoid on an empty array returns an empty array."""
        vector = np.array([])
        result = sigmoid(vector)
        assert result.shape == (0,)
        assert result.size == 0

    def test_sigmoid_infinity_input(self):
        """Test sigmoid handles infinite inputs without error."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.0)
