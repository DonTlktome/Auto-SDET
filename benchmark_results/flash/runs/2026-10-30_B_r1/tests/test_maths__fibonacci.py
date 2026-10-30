"""
Pytest test suite for maths/fibonacci.py

Covers:
    - time_func
    - fib_iterative_yield
    - fib_iterative
    - fib_recursive
    - fib_recursive_cached
    - fib_memoization
    - fib_binet
    - matrix_pow_np
    - fib_matrix_np
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make the target module importable regardless of the test runner's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fibonacci as fib_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

FIRST_11 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------


def test_time_func_returns_function_output():
    """time_func must return whatever the wrapped callable returns."""
    assert fib_mod.time_func(lambda a, b: a + b, 2, 3) == 5


def test_time_func_passes_kwargs():
    """time_func must forward keyword arguments to the wrapped callable."""

    def add(a, b=0):
        return a + b

    assert fib_mod.time_func(add, 1, b=41) == 42


def test_time_func_prints_millisecond_branch(capsys):
    """A sub-second runtime must be reported in milliseconds."""
    fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert "ms" in out


def test_time_func_prints_second_branch(capsys):
    """A runtime of at least one second must be reported in seconds."""
    # Force the measured duration to be 2.5 seconds.
    with patch.object(fib_mod, "time", side_effect=[0.0, 2.5]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "s" in out
    assert "ms" not in out
    assert "2.5000" in out


def test_time_func_uses_function_name(capsys):
    """The printed message must contain the wrapped function's name."""

    def my_named_func():
        return None

    fib_mod.time_func(my_named_func)
    assert "my_named_func" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# fib_iterative_yield
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_iterative_yield_values(n, expected):
    """fib_iterative_yield must produce the first n+1 Fibonacci numbers."""
    assert list(fib_mod.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_is_lazy_iterator():
    """fib_iterative_yield must return a lazy iterator, not a list."""
    gen = fib_mod.fib_iterative_yield(5)
    assert iter(gen) is gen
    assert next(gen) == 0
    assert next(gen) == 1


def test_fib_iterative_yield_negative_raises():
    """A negative n must raise ValueError when the generator is consumed."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_mod.fib_iterative_yield(-1))


def test_fib_iterative_yield_large_n():
    """A larger n must still yield the correct final value."""
    values = list(fib_mod.fib_iterative_yield(30))
    assert len(values) == 31
    assert values[-1] == 832040


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_iterative_values(n, expected):
    """fib_iterative must return the first n+1 Fibonacci numbers as a list."""
    result = fib_mod.fib_iterative(n)
    assert isinstance(result, list)
    assert result == expected


def test_fib_iterative_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_iterative(-1)


def test_fib_iterative_large_n():
    """fib_iterative must handle larger inputs correctly."""
    result = fib_mod.fib_iterative(50)
    assert len(result) == 51
    assert result[-1] == 12586269025


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_recursive_values(n, expected):
    """fib_recursive must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive(n) == expected


def test_fib_recursive_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive(-1)


def test_fib_recursive_matches_iterative():
    """fib_recursive and fib_iterative must agree for a moderate n."""
    assert fib_mod.fib_recursive(15) == fib_mod.fib_iterative(15)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_recursive_cached_values(n, expected):
    """fib_recursive_cached must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive_cached(n) == expected


def test_fib_recursive_cached_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive_cached(-1)


def test_fib_recursive_cached_large_n():
    """The cached variant must handle larger inputs quickly and correctly."""
    result = fib_mod.fib_recursive_cached(100)
    assert len(result) == 101
    assert result[-1] == 354224848179261915075


# ---------------------------------------------------------------------------
# fib_memoization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_memoization_values(n, expected):
    """fib_memoization must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_memoization(n) == expected


def test_fib_memoization_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_memoization(-1)


def test_fib_memoization_large_n():
    """The memoized variant must handle larger inputs correctly."""
    result = fib_mod.fib_memoization(60)
    assert len(result) == 61
    assert result[-1] == 1548008755920


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIRST_11),
    ],
)
def test_fib_binet_values(n, expected):
    """fib_binet must return the first n+1 Fibonacci numbers for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 2000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError because of float overflow."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474_is_allowed():
    """n = 1474 is the largest accepted value and must not raise."""
    result = fib_mod.fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


def test_fib_binet_matches_iterative_for_small_n():
    """Binet's formula must agree with the exact iterative version for small n."""
    assert fib_mod.fib_binet(30) == fib_mod.fib_iterative(30)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_power_zero_returns_identity():
    """Raising any matrix to the power 0 must return the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_matrix():
    """Raising a matrix to the power 1 must return the matrix itself."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    np.testing.assert_array_equal(fib_mod.matrix_pow_np(m, 1), m)


@pytest.mark.parametrize(
    "power, expected",
    [
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
        (10, [[89, 55], [55, 34]]),
    ],
)
def test_matrix_pow_np_values(power, expected):
    """matrix_pow_np must compute the correct matrix power."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, power)
    np.testing.assert_array_equal(result, np.array(expected, dtype=int))


def test_matrix_pow_np_negative_raises():
    """A negative power must raise ValueError."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        fib_mod.matrix_pow_np(m, -1)


def test_matrix_pow_np_identity_matrix_any_power():
    """The identity matrix raised to any power must remain the identity."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(
        fib_mod.matrix_pow_np(identity, 7), identity
    )


# ---------------------------------------------------------------------------
# fib_matrix_np
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fib_matrix_np_values(n, expected):
    """fib_matrix_np must return the n-th Fibonacci number."""
    assert fib_mod.fib_matrix_np(n) == expected


def test_fib_matrix_np_returns_int():
    """fib_matrix_np must return a plain Python int, not a numpy scalar."""
    result = fib_mod.fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(-1)


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np must agree with fib_iterative for a range of n."""
    for n in range(0, 25):
        assert fib_mod.fib_matrix_np(n) == fib_mod.fib_iterative(n)[-1]


def test_fib_matrix_np_uses_matrix_pow_np():
    """fib_matrix_np must delegate to matrix_pow_np (isolation via mock)."""
    fake_result = np.array([[13, 8], [8, 5]], dtype=int)
    with patch.object(
        fib_mod, "matrix_pow_np", MagicMock(return_value=fake_result)
    ) as mocked:
        assert fib_mod.fib_matrix_np(7) == 13
    mocked.assert_called_once()
    called_matrix, called_power = mocked.call_args.args
    np.testing.assert_array_equal(
        called_matrix, np.array([[1, 1], [1, 0]], dtype=int)
    )
    assert called_power == 6


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 12, 20])
def test_all_implementations_agree(n):
    """All list-returning implementations must produce identical results."""
    expected = fib_mod.fib_iterative(n)
    assert list(fib_mod.fib_iterative_yield(n)) == expected
    assert fib_mod.fib_recursive(n) == expected
    assert fib_mod.fib_recursive_cached(n) == expected
    assert fib_mod.fib_memoization(n) == expected
    assert fib_mod.fib_binet(n) == expected
    assert fib_mod.fib_matrix_np(n) == expected[-1]