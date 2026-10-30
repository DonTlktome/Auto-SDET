"""
Pytest test suite for the fibonacci module.

Covers all public functions in maths/fibonacci.py:
- fib_iterative_yield
- fib_iterative
- fib_recursive
- fib_recursive_cached
- fib_memoization
- fib_binet
- matrix_pow_np
- fib_matrix_np
- time_func
"""

import numpy as np
import pytest

from maths.fibonacci import (
    fib_binet,
    fib_iterative,
    fib_iterative_yield,
    fib_matrix_np,
    fib_memoization,
    fib_recursive,
    fib_recursive_cached,
    matrix_pow_np,
    time_func,
)


# ---------------------------------------------------------------------------
# fib_iterative_yield
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_iterative_yield_positive(n, expected):
    """Test that fib_iterative_yield produces the correct sequence."""
    assert list(fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_iterative_yield(-1))


def test_fib_iterative_yield_is_iterator():
    """Test that fib_iterative_yield returns a lazy iterator."""
    gen = fib_iterative_yield(3)
    assert iter(gen) is gen
    assert next(gen) == 0


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_iterative_positive(n, expected):
    """Test that fib_iterative returns the correct list."""
    assert fib_iterative(n) == expected


def test_fib_iterative_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_iterative(-1)


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_recursive_positive(n, expected):
    """Test that fib_recursive returns the correct list."""
    assert fib_recursive(n) == expected


def test_fib_recursive_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_recursive(-1)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_recursive_cached_positive(n, expected):
    """Test that fib_recursive_cached returns the correct list."""
    assert fib_recursive_cached(n) == expected


def test_fib_recursive_cached_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_recursive_cached(-1)


# ---------------------------------------------------------------------------
# fib_memoization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_memoization_positive(n, expected):
    """Test that fib_memoization returns the correct list."""
    assert fib_memoization(n) == expected


def test_fib_memoization_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_memoization(-1)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
    ],
)
def test_fib_binet_positive(n, expected):
    """Test that fib_binet returns the correct list for small inputs."""
    assert fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1)


def test_fib_binet_too_large_raises():
    """Test that n >= 1475 raises ValueError."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_binet(1475)


def test_fib_binet_boundary_1474():
    """Test that n = 1474 (largest valid input) does not raise."""
    result = fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------

def test_matrix_pow_np_zero_power_identity():
    """Test that raising to power 0 returns the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    expected = np.array([[1, 0], [0, 1]], dtype=int)
    result = matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, expected)


def test_matrix_pow_np_one_power_same():
    """Test that raising to power 1 returns the original matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = matrix_pow_np(m, 1)
    np.testing.assert_array_equal(result, m)


def test_matrix_pow_np_positive_power():
    """Test that raising to a positive power produces the correct matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    expected = np.array([[8, 5], [5, 3]], dtype=int)
    result = matrix_pow_np(m, 5)
    np.testing.assert_array_equal(result, expected)


def test_matrix_pow_np_negative_power_raises():
    """Test that a negative power raises ValueError."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        matrix_pow_np(m, -1)


# ---------------------------------------------------------------------------
# fib_matrix_np
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fib_matrix_np_positive(n, expected):
    """Test that fib_matrix_np returns the correct n-th Fibonacci number."""
    assert fib_matrix_np(n) == expected


def test_fib_matrix_np_negative_raises():
    """Test that a negative input raises ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_matrix_np(-1)


def test_fib_matrix_np_large_value():
    """Test that fib_matrix_np handles a large index correctly."""
    assert fib_matrix_np(50) == 12586269025


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------

def test_time_func_returns_output(capsys):
    """Test that time_func returns the wrapped function's output."""
    def add(a, b):
        return a + b

    result = time_func(add, 2, 3)
    assert result == 5
    captured = capsys.readouterr()
    assert "add runtime:" in captured.out


def test_time_func_passes_kwargs(capsys):
    """Test that time_func forwards keyword arguments correctly."""
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}"

    result = time_func(greet, "World", greeting="Hi")
    assert result == "Hi, World"
    captured = capsys.readouterr()
    assert "greet runtime:" in captured.out