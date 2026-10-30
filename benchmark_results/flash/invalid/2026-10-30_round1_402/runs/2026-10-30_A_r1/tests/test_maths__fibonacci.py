"""
Tests for maths/fibonacci.py

Covers every public function:
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

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fibonacci as fib_mod  # noqa: E402
from fibonacci import (  # noqa: E402
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

# First 11 Fibonacci numbers (0-indexed): F(0) .. F(10)
FIB_0_TO_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Every function returning the first n (0-indexed) Fibonacci numbers.
LIST_FUNCS = [
    fib_iterative,
    fib_recursive,
    fib_recursive_cached,
    fib_memoization,
    fib_binet,
]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------


def test_time_func_returns_wrapped_output():
    """time_func must return whatever the wrapped callable returns."""
    assert time_func(lambda a, b: a + b, 2, 3) == 5


def test_time_func_passes_kwargs():
    """time_func must forward keyword arguments to the wrapped callable."""

    def add(a, b=0):
        return a + b

    assert time_func(add, 1, b=41) == 42


def test_time_func_prints_milliseconds_for_fast_calls(capsys):
    """A sub-second call must be reported in milliseconds (ms suffix)."""
    time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert out.strip().endswith("ms")


def test_time_func_prints_seconds_for_slow_calls(capsys):
    """A call taking >= 1 second must be reported in seconds (s suffix)."""
    # Simulate a slow function by patching the module's `time` clock.
    with patch.object(fib_mod, "time", side_effect=[0.0, 2.5]):
        time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert out.strip().endswith(" s")


def test_time_func_uses_function_name(capsys):
    """The printed message must contain the wrapped function's __name__."""

    def my_named_func():
        return None

    time_func(my_named_func)
    out = capsys.readouterr().out
    assert "my_named_func" in out


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
        (10, FIB_0_TO_10),
    ],
)
def test_fib_iterative_yield_values(n, expected):
    """fib_iterative_yield must yield the first n+1 Fibonacci numbers."""
    assert list(fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_returns_iterator():
    """fib_iterative_yield must return a lazy iterator, not a list."""
    result = fib_iterative_yield(5)
    assert iter(result) is result
    assert not isinstance(result, list)


def test_fib_iterative_yield_negative_raises():
    """A negative n must raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_iterative_yield(-1))


def test_fib_iterative_yield_is_lazy():
    """The generator must not raise until it is actually consumed."""
    gen = fib_iterative_yield(-1)  # no exception yet
    with pytest.raises(ValueError):
        next(gen)


# ---------------------------------------------------------------------------
# fib_iterative / fib_recursive / fib_recursive_cached / fib_memoization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", LIST_FUNCS)
@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIB_0_TO_10),
    ],
)
def test_list_functions_values(func, n, expected):
    """All list-returning implementations must agree on small inputs."""
    assert func(n) == expected


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_negative_raises(func):
    """Every list-returning implementation must reject negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        func(-1)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_length_is_n_plus_one(func):
    """The returned list must always contain exactly n + 1 elements."""
    for n in (0, 1, 3, 7):
        assert len(func(n)) == n + 1


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_are_strictly_increasing_after_first(func):
    """From index 2 onward the sequence must be non-decreasing and grow."""
    seq = func(20)
    assert seq[0] == 0
    assert seq[1] == 1
    assert all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    assert seq[-1] == 6765  # F(20)


def test_fib_iterative_large_value():
    """fib_iterative must produce exact big integers (no float overflow)."""
    assert fib_iterative(100)[-1] == 354224848179261915075


def test_fib_recursive_cached_large_value():
    """The cached recursive version must also handle large n exactly."""
    assert fib_recursive_cached(100)[-1] == 354224848179261915075


def test_fib_memoization_large_value():
    """The memoized version must handle large n exactly."""
    assert fib_memoization(100)[-1] == 354224848179261915075


def test_fib_recursive_returns_list_type():
    """fib_recursive must return a plain list of ints."""
    result = fib_recursive(5)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_fib_recursive_cached_returns_list_type():
    """fib_recursive_cached must return a plain list of ints."""
    result = fib_recursive_cached(5)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_fib_memoization_returns_list_type():
    """fib_memoization must return a plain list of ints."""
    result = fib_memoization(5)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_fib_iterative_returns_list_type():
    """fib_iterative must return a plain list of ints."""
    result = fib_iterative(5)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIB_0_TO_10),
    ],
)
def test_fib_binet_values(n, expected):
    """Binet's formula must match the exact sequence for small n."""
    assert fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """Negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 2000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError because floats overflow."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_binet(n)


def test_fib_binet_boundary_1474_is_allowed():
    """n = 1474 is the largest accepted input and must not raise."""
    result = fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


def test_fib_binet_matches_iterative_below_divergence():
    """Binet must agree with the exact iterative version for n <= 70."""
    assert fib_binet(70) == fib_iterative(70)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_power_zero_returns_identity():
    """Raising any matrix to the power 0 must yield the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_same_matrix():
    """Raising a matrix to the power 1 must return the matrix itself."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    np.testing.assert_array_equal(matrix_pow_np(m, 1), m)


@pytest.mark.parametrize(
    "power, expected",
    [
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
        (10, [[89, 55], [55, 34]]),
    ],
)
def test_matrix_pow_np_known_powers(power, expected):
    """Binary exponentiation must produce the correct Fibonacci matrix powers."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    np.testing.assert_array_equal(
        matrix_pow_np(m, power), np.array(expected, dtype=int)
    )


def test_matrix_pow_np_negative_power_raises():
    """A negative power must raise ValueError."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        matrix_pow_np(m, -1)


def test_matrix_pow_np_identity_matrix_any_power():
    """The identity matrix raised to any power stays the identity."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(matrix_pow_np(identity, 7), identity)


def test_matrix_pow_np_does_not_mutate_input():
    """matrix_pow_np must not modify the caller's matrix in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    matrix_pow_np(m, 5)
    np.testing.assert_array_equal(m, original)


def test_matrix_pow_np_returns_ndarray():
    """The result must be a numpy ndarray."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    assert isinstance(matrix_pow_np(m, 3), np.ndarray)


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
    assert fib_matrix_np(n) == expected


def test_fib_matrix_np_negative_raises():
    """A negative index must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_matrix_np(-1)


def test_fib_matrix_np_returns_int():
    """The result must be a Python int, not a numpy scalar."""
    result = fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np(n) must equal the last element of fib_iterative(n)."""
    for n in (0, 1, 2, 7, 15, 30):
        assert fib_matrix_np(n) == fib_iterative(n)[-1]


def test_fib_matrix_np_large_value():
    """Matrix exponentiation must be exact up to the int64 limit.

    ``matrix_pow_np`` performs arithmetic on numpy integer arrays, so results
    are bound by int64.  F(92) = 7540113804746346429 is the largest Fibonacci
    number that still fits in a signed 64-bit integer (F(93) already exceeds
    it), so this test asserts exactness up to -- but not beyond -- that bound.
    """
    assert fib_matrix_np(92) == 7540113804746346429
    assert fib_matrix_np(92) == fib_iterative(92)[-1]


def test_fib_matrix_np_calls_matrix_pow_np():
    """fib_matrix_np must delegate to matrix_pow_np with power n - 1."""
    with patch.object(
        fib_mod, "matrix_pow_np", wraps=fib_mod.matrix_pow_np
    ) as mock_pow:
        fib_matrix_np(6)
    mock_pow.assert_called_once()
    _, power = mock_pow.call_args[0]
    assert power == 5


def test_fib_matrix_np_does_not_call_matrix_pow_np_for_zero():
    """For n == 0 the early return must skip matrix exponentiation entirely."""
    with patch.object(fib_mod, "matrix_pow_np") as mock_pow:
        assert fib_matrix_np(0) == 0
    mock_pow.assert_not_called()


def test_fib_matrix_np_with_mocked_matrix_pow_np():
    """Isolate numpy work by mocking matrix_pow_np's return value."""
    fake = MagicMock()
    fake.__getitem__ = MagicMock(return_value=np.int64(42))
    with patch.object(fib_mod, "matrix_pow_np", return_value=fake):
        assert fib_matrix_np(3) == 42
    fake.__getitem__.assert_called_once_with((0, 0))
