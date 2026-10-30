import numpy as np
import pytest

from maths.sigmoid import sigmoid


class TestSigmoid:
    '''Test suite for the sigmoid function.'''

    @pytest.mark.parametrize(
        'input_vector, expected',
        [
            (np.array([0.0]), np.array([0.5])),
            (np.array([-1.0, 1.0, 2.0]), np.array([0.26894142, 0.73105858, 0.88079708])),
            (np.array([1.0]), np.array([0.73105858])),
            (np.array([-1.0]), np.array([0.26894142])),
        ],
    )
    def test_sigmoid_positive_cases(self, input_vector, expected):
        '''Test sigmoid returns correct values for typical real inputs.'''
        result = sigmoid(input_vector)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_sigmoid_zero_vector(self):
        '''Test sigmoid on a vector of zeros returns 0.5 for each element.'''
        result = sigmoid(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.5, 0.5]))

    def test_sigmoid_large_positive_value(self):
        '''Test sigmoid approaches 1 for a large positive input.'''
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_sigmoid_large_negative_value(self):
        '''Test sigmoid approaches 0 for a large negative input.'''
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_output_range(self):
        '''Test sigmoid output is always bounded between 0 and 1.'''
        inputs = np.linspace(-50.0, 50.0, 1000)
        result = sigmoid(inputs)
        assert np.all(result > 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_symmetry(self):
        '''Test the symmetry property: sigmoid(-x) = 1 - sigmoid(x).'''
        x = np.array([0.5, 1.5, -2.5, 3.0])
        result = sigmoid(-x)
        expected = 1.0 - sigmoid(x)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_sigmoid_preserves_shape(self):
        '''Test sigmoid preserves the shape of the input array.'''
        input_vector = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = sigmoid(input_vector)
        assert result.shape == input_vector.shape

    def test_sigmoid_empty_array(self):
        '''Test sigmoid on an empty array returns an empty array.'''
        result = sigmoid(np.array([]))
        assert result.size == 0
        assert result.shape == (0,)

    def test_sigmoid_integer_input(self):
        '''Test sigmoid accepts integer input and returns float output.'''
        result = sigmoid(np.array([0, 1, -1]))
        assert result.dtype == np.float64
        np.testing.assert_allclose(
            result, np.array([0.5, 0.73105858, 0.26894142]), rtol=1e-6, atol=1e-6
        )
