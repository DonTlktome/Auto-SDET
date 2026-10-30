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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maths"))

import fibonacci as fib_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

EXPECTED_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Every function that returns the first n (0-indexed) Fibonacci numbers.
LIST_FUNCS = [
    fib_mod.fib_iterative,
    fib_mod.fib_recursive,
    fib_mod.fib_recursive_cached,
    fib_mod.fib_memoization,
    fib_mod.fib_binet,
]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------


def test_time_func_returns_output():
    """time_func must return whatever the wrapped callable returns."""
    assert fib_mod.time_func(lambda a, b: a + b, 2, 3) == 5


def test_time_func_passes_kwargs():
    """time_func must forward keyword arguments to the wrapped callable."""

    def add(a, b=0):
        return a + b

    assert fib_mod.time_func(add, 1, b=41) == 42


def test_time_func_prints_runtime(capsys):
    """time_func must print a runtime line containing the function name."""
    fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert "<lambda>" in out


@pytest.mark.parametrize("elapsed, expected_unit", [(0.0005, "ms"), (2.0, "s")])
def test_time_func_unit_selection(elapsed, expected_unit, capsys):
    """time_func prints milliseconds for sub-second runs and seconds otherwise."""
    fake_time = MagicMock(side_effect=[100.0, 100.0 + elapsed])
    with patch.object(fib_mod, "time", fake_time):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert expected_unit in out


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
        (10, EXPECTED_10),
    ],
)
def test_fib_iterative_yield_happy(n, expected):
    """fib_iterative_yield yields the first n+1 Fibonacci numbers."""
    assert list(fib_mod.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_is_lazy_iterator():
    """fib_iterative_yield must return a lazy iterator, not a materialised list."""
    gen = fib_mod.fib_iterative_yield(5)
    assert iter(gen) is gen
    assert next(gen) == 0
    assert next(gen) == 1


def test_fib_iterative_yield_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_mod.fib_iterative_yield(-1))


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, EXPECTED_10),
    ],
)
def test_fib_iterative_happy(n, expected):
    """fib_iterative returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_iterative(n) == expected


def test_fib_iterative_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_iterative(-1)


def test_fib_iterative_large_n_length():
    """The returned list must always have exactly n+1 elements."""
    result = fib_mod.fib_iterative(100)
    assert len(result) == 101
    assert result[-1] == 354224848179261915075


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, EXPECTED_10),
    ],
)
def test_fib_recursive_happy(n, expected):
    """fib_recursive returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive(n) == expected


def test_fib_recursive_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive(-1)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, EXPECTED_10),
    ],
)
def test_fib_recursive_cached_happy(n, expected):
    """fib_recursive_cached returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive_cached(n) == expected


def test_fib_recursive_cached_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive_cached(-1)


def test_fib_recursive_cached_handles_larger_n():
    """Caching must make moderately large n feasible."""
    result = fib_mod.fib_recursive_cached(50)
    assert len(result) == 51
    assert result[-1] == 12586269025


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
        (10, EXPECTED_10),
    ],
)
def test_fib_memoization_happy(n, expected):
    """fib_memoization returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_memoization(n) == expected


def test_fib_memoization_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_memoization(-1)


def test_fib_memoization_handles_larger_n():
    """Memoization must make moderately large n feasible."""
    result = fib_mod.fib_memoization(50)
    assert len(result) == 51
    assert result[-1] == 12586269025


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, EXPECTED_10),
    ],
)
def test_fib_binet_happy(n, expected):
    """fib_binet returns the first n+1 Fibonacci numbers for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 5000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474_ok():
    """n == 1474 is the largest accepted value and must not raise."""
    result = fib_mod.fib_binet(1474)
    assert len(result) == 1475


def test_fib_binet_matches_iterative_for_small_n():
    """Binet's formula must agree with the exact iterative version for small n."""
    assert fib_mod.fib_binet(30) == fib_mod.fib_iterative(30)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_power_zero_identity():
    """Raising any matrix to the power 0 yields the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_matrix():
    """Raising a matrix to the power 1 returns the matrix itself."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    np.testing.assert_array_equal(fib_mod.matrix_pow_np(m, 1), m)


@pytest.mark.parametrize(
    "power, expected",
    [
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
    ],
)
def test_matrix_pow_np_happy(power, expected):
    """matrix_pow_np computes the correct matrix power via binary exponentiation."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    np.testing.assert_array_equal(
        fib_mod.matrix_pow_np(m, power), np.array(expected, dtype=int)
    )


def test_matrix_pow_np_negative_raises():
    """A negative power must raise ValueError('power is negative')."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        fib_mod.matrix_pow_np(m, -1)


def test_matrix_pow_np_does_not_mutate_input():
    """matrix_pow_np must not modify the caller's matrix in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    fib_mod.matrix_pow_np(m, 7)
    np.testing.assert_array_equal(m, original)


# ---------------------------------------------------------------------------
# fib_matrix_np
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fib_matrix_np_happy(n, expected):
    """fib_matrix_np returns the n-th Fibonacci number."""
    assert fib_mod.fib_matrix_np(n) == expected


def test_fib_matrix_np_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(-1)


def test_fib_matrix_np_returns_int():
    """fib_matrix_np must return a plain Python int, not a numpy scalar."""
    result = fib_mod.fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_large_n():
    """fib_matrix_np must handle large n efficiently and exactly.

    The implementation uses numpy arrays with the platform default integer
    dtype (int64 on Linux).  fib(92) is the largest Fibonacci number that
    still fits in a signed 64-bit integer, so this is the largest n for
    which the exact value can be asserted.
    """
    assert fib_mod.fib_matrix_np(92) == 7540113804746346429


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", LIST_FUNCS)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 15])
def test_all_list_implementations_agree(func, n):
    """All list-returning implementations must agree with fib_iterative."""
    assert func(n) == fib_mod.fib_iterative(n)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 15])
def test_matrix_matches_iterative(n):
    """fib_matrix_np(n) must equal the last element of fib_iterative(n)."""
    assert fib_mod.fib_matrix_np(n) == fib_mod.fib_iterative(n)[-1]


@pytest.mark.parametrize("n", [0, 1, 5, 10])
def test_yield_matches_iterative(n):
    """fib_iterative_yield must produce the same sequence as fib_iterative."""
    assert list(fib_mod.fib_iterative_yield(n)) == fib_mod.fib_iterative(n)
