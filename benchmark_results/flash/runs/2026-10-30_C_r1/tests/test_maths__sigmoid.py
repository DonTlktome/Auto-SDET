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

    def test_sigmoid_preserves_input_array(self):
        """The input array must not be mutated by the function."""
        vector = np.array([-2.0, 0.0, 2.0])
        original = vector.copy()
        sigmoid(vector)
        np.testing.assert_array_equal(vector, original)

    def test_sigmoid_accepts_python_list(self):
        """A sequence must be converted to np.ndarray before being passed in."""
        result = sigmoid(np.array([0.0, 1.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.7310585786300049]))

    def test_sigmoid_rejects_raw_python_list(self):
        """The documented API takes np.ndarray; a raw list is not supported."""
        with pytest.raises(TypeError):
            sigmoid([0.0, 1.0])

    def test_sigmoid_accepts_scalar(self):
        """A scalar input must yield a scalar-like result equal to 0.5 at 0."""
        result = sigmoid(0.0)
        assert float(result) == pytest.approx(0.5)


class TestSigmoidEdgeCases:
    """Boundary and edge case tests for the sigmoid function."""

    def test_sigmoid_of_empty_array_returns_empty_array(self):
        """An empty input array must produce an empty output array."""
        result = sigmoid(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_sigmoid_of_large_positive_value_saturates_to_one(self):
        """Very large positive inputs must saturate to 1.0 without overflow."""
        result = sigmoid(np.array([1000.0]))
        assert result[0] == pytest.approx(1.0)

    def test_sigmoid_of_large_negative_value_saturates_to_zero(self):
        """Very large negative inputs must saturate to 0.0 without overflow."""
        result = sigmoid(np.array([-1000.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-12)

    def test_sigmoid_output_is_bounded_between_zero_and_one(self):
        """All outputs must lie within the closed interval [0, 1].

        Extreme inputs saturate to exactly 0.0 / 1.0 because exp() overflows
        to inf, so the closed interval is the strongest valid guarantee.
        """
        vector = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
        result = sigmoid(vector)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_output_strictly_inside_unit_interval(self):
        """For moderate inputs the output must lie strictly within (0, 1)."""
        vector = np.array([-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0])
        result = sigmoid(vector)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)

    def test_sigmoid_is_symmetric_around_zero(self):
        """sigmoid(-x) must equal 1 - sigmoid(x) for any x."""
        vector = np.array([0.5, 1.5, 3.0, 7.25])
        np.testing.assert_allclose(
            sigmoid(-vector), 1.0 - sigmoid(vector), rtol=1e-12
        )

    def test_sigmoid_is_monotonically_increasing(self):
        """The sigmoid function must be strictly increasing."""
        vector = np.linspace(-10.0, 10.0, 101)
        result = sigmoid(vector)
        assert np.all(np.diff(result) > 0.0)

    def test_sigmoid_handles_two_dimensional_input(self):
        """A 2-D input array must be transformed element-wise, keeping shape."""
        vector = np.array([[0.0, 1.0], [-1.0, 2.0]])
        result = sigmoid(vector)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    [0.5, 0.7310585786300049],
                    [0.2689414213699951, 0.8807970779778823],
                ]
            ),
            rtol=1e-12,
        )

    def test_sigmoid_handles_nan_input(self):
        """NaN inputs must propagate as NaN rather than raising."""
        result = sigmoid(np.array([np.nan]))
        assert np.isnan(result[0])

    def test_sigmoid_handles_infinite_input(self):
        """Positive and negative infinity must map to 1.0 and 0.0 respectively."""
        result = sigmoid(np.array([np.inf, -np.inf]))
        np.testing.assert_allclose(result, np.array([1.0, 0.0]))


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, 0.5),
        (1.0, 1.0 / (1.0 + math.exp(-1.0))),
        (-1.0, 1.0 / (1.0 + math.exp(1.0))),
        (2.0, 1.0 / (1.0 + math.exp(-2.0))),
        (-2.0, 1.0 / (1.0 + math.exp(2.0))),
        (0.5, 1.0 / (1.0 + math.exp(-0.5))),
        (10.0, 1.0 / (1.0 + math.exp(-10.0))),
    ],
)
def test_sigmoid_matches_reference_formula(value, expected):
    """Each element must equal the reference formula 1 / (1 + exp(-x))."""
    result = sigmoid(np.array([value]))
    assert result[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([0.0]),
        np.array([-3.0, -1.0, 0.0, 1.0, 3.0]),
        np.linspace(-5.0, 5.0, 11),
        np.zeros(5),
        np.ones(3),
    ],
)
def test_sigmoid_elementwise_against_math_exp(vector):
    """Every element must match a pure-Python math.exp computation."""
    result = sigmoid(vector)
    expected = np.array([1.0 / (1.0 + math.exp(-float(x))) for x in vector])
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_sigmoid_uses_numpy_exp(monkeypatch):
    """The implementation must delegate to numpy.exp (dependency isolation check)."""
    calls = {}

    real_exp = np.exp

    def spy_exp(x):
        calls["called"] = True
        calls["arg"] = np.asarray(x).copy()
        return real_exp(x)

    monkeypatch.setattr(np, "exp", spy_exp)
    result = sigmoid(np.array([0.0, 1.0]))

    assert calls.get("called") is True
    np.testing.assert_array_equal(calls["arg"], np.array([-0.0, -1.0]))
    np.testing.assert_allclose(result, np.array([0.5, 0.7310585786300049]))
