"""
Pytest test suite for maths/fibonacci.py

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

All tests are self-contained and require no network access.
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


# ---------------------------------------------------------------------------
# Shared reference data
# ---------------------------------------------------------------------------

# First 21 Fibonacci numbers (0-indexed): F(0) .. F(20)
FIB_20 = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
    2584, 4181, 6765,
]

# Every function that returns the first n+1 Fibonacci numbers as a list.
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


def test_time_func_prints_millisecond_branch(capsys):
    """Fast calls (< 1 s) must be reported in milliseconds."""
    time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert "ms" in out


def test_time_func_prints_second_branch(capsys):
    """Slow calls (>= 1 s) must be reported in seconds."""
    # Simulate a 2.5 second elapsed time without actually sleeping.
    with patch.object(fib_mod, "time", side_effect=[100.0, 102.5]):
        time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert " s" in out
    assert "ms" not in out


def test_time_func_uses_function_name(capsys):
    """The printed message must contain the wrapped function's __name__."""

    def my_named_func():
        return None

    time_func(my_named_func)
    assert "my_named_func" in capsys.readouterr().out


def test_time_func_propagates_exceptions():
    """Exceptions raised by the wrapped callable must not be swallowed."""

    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        time_func(boom)


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
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        (20, FIB_20),
    ],
)
def test_fib_iterative_yield_values(n, expected):
    """fib_iterative_yield must yield the first n+1 Fibonacci numbers."""
    assert list(fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_returns_iterator():
    """The function must be a generator (lazy iterator), not a list."""
    result = fib_iterative_yield(5)
    assert hasattr(result, "__next__")
    assert hasattr(result, "__iter__")
    assert not isinstance(result, list)


def test_fib_iterative_yield_is_lazy():
    """Consuming only part of the generator must not compute everything."""
    gen = fib_iterative_yield(1000)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 1


def test_fib_iterative_yield_negative_raises():
    """A negative n must raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_iterative_yield(-1))


def test_fib_iterative_yield_negative_raises_on_consumption():
    """The ValueError surfaces when the generator is first advanced."""
    gen = fib_iterative_yield(-5)
    with pytest.raises(ValueError, match="n is negative"):
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
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        (20, FIB_20),
    ],
)
def test_list_functions_values(func, n, expected):
    """All list-returning implementations must agree on the sequence."""
    assert func(n) == expected


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_return_list(func):
    """Every list-returning implementation must return a real list."""
    assert isinstance(func(3), list)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_length_is_n_plus_one(func):
    """The result must contain exactly n + 1 elements."""
    for n in (0, 1, 7, 15):
        assert len(func(n)) == n + 1


@pytest.mark.parametrize("func", LIST_FUNCS)
@pytest.mark.parametrize("n", [-1, -2, -100])
def test_list_functions_negative_raises(func, n):
    """A negative n must raise ValueError for every implementation."""
    with pytest.raises(ValueError, match="n is negative"):
        func(n)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_agree_with_each_other(func):
    """All implementations must produce identical output for a mid-size n."""
    assert func(25) == fib_iterative(25)


def test_fib_iterative_large_n():
    """fib_iterative must handle a moderately large n correctly."""
    result = fib_iterative(100)
    assert len(result) == 101
    assert result[100] == 354224848179261915075


def test_fib_recursive_cached_large_n():
    """The cached recursive version must handle a large n without blowing up."""
    result = fib_recursive_cached(100)
    assert result[100] == 354224848179261915075


def test_fib_memoization_large_n():
    """The memoized version must handle a large n correctly."""
    result = fib_memoization(100)
    assert result[100] == 354224848179261915075


def test_fib_recursive_small_n_only():
    """The naive recursive version is exponential; test only a small n."""
    assert fib_recursive(12) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_fib_recursive_cached_is_repeatable():
    """Repeated calls must not leak state between invocations."""
    first = fib_recursive_cached(10)
    second = fib_recursive_cached(10)
    assert first == second


def test_fib_memoization_is_repeatable():
    """Repeated calls must not leak state between invocations."""
    assert fib_memoization(10) == fib_memoization(10)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        (20, FIB_20),
    ],
)
def test_fib_binet_values(n, expected):
    """Binet's formula must match the exact sequence for small n."""
    assert fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 2000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_binet(n)


def test_fib_binet_boundary_1474_is_allowed():
    """n == 1474 is the largest accepted value and must not raise."""
    result = fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


def test_fib_binet_negative_checked_before_size():
    """A negative n must report 'n is negative', not 'n is too large'."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1475)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_power_zero_returns_identity():
    """Raising any matrix to the power 0 must yield the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_matrix():
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
    """Binary exponentiation must match the known Fibonacci matrix powers."""
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
    for power in (0, 1, 2, 7, 64):
        np.testing.assert_array_equal(matrix_pow_np(identity, power), identity)


def test_matrix_pow_np_does_not_mutate_input():
    """The input matrix must not be modified in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    matrix_pow_np(m, 9)
    np.testing.assert_array_equal(m, original)


def test_matrix_pow_np_returns_ndarray():
    """The return type must be a numpy ndarray."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    assert isinstance(matrix_pow_np(m, 4), np.ndarray)


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
        (30, 832040),
    ],
)
def test_fib_matrix_np_values(n, expected):
    """fib_matrix_np must return the n-th Fibonacci number."""
    assert fib_matrix_np(n) == expected


def test_fib_matrix_np_returns_int():
    """The result must be a plain Python int, not a numpy scalar."""
    result = fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_negative_raises():
    """A negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_matrix_np(-1)


def test_fib_matrix_np_large_n():
    """Matrix exponentiation must return exact values up to the int64 limit.

    The implementation uses numpy's default integer dtype (int64), so results
    are only exact while they fit in a signed 64-bit integer. F(92) is the
    largest Fibonacci number that fits in int64 (it stays below 2**63 - 1);
    F(93) and beyond overflow, so we anchor the exact-value check at n == 92.
    """
    assert fib_matrix_np(92) == 7540113804746346429


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np(n) must equal the last element of fib_iterative(n)."""
    for n in range(0, 40):
        assert fib_matrix_np(n) == fib_iterative(n)[-1]


def test_fib_matrix_np_uses_matrix_pow_np():
    """fib_matrix_np must delegate to matrix_pow_np (isolation via mock)."""
    fake_result = np.array([[55, 0], [0, 0]], dtype=int)
    with patch.object(fib_mod, "matrix_pow_np", return_value=fake_result) as mocked:
        assert fib_matrix_np(10) == 55
    mocked.assert_called_once()
    called_matrix, called_power = mocked.call_args[0]
    np.testing.assert_array_equal(called_matrix, np.array([[1, 1], [1, 0]], dtype=int))
    assert called_power == 9


def test_fib_matrix_np_zero_skips_matrix_pow():
    """For n == 0 the matrix power helper must not be called at all."""
    with patch.object(fib_mod, "matrix_pow_np") as mocked:
        assert fib_matrix_np(0) == 0
    mocked.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


def test_all_implementations_agree():
    """Every implementation must agree on the first 20 Fibonacci numbers."""
    n = 20
    expected = FIB_20
    assert fib_iterative(n) == expected
    assert fib_recursive(n) == expected
    assert fib_recursive_cached(n) == expected
    assert fib_memoization(n) == expected
    assert fib_binet(n) == expected
    assert list(fib_iterative_yield(n)) == expected
    assert [fib_matrix_np(i) for i in range(n + 1)] == expected


def test_time_func_wraps_fibonacci(capsys):
    """time_func must work with the module's own functions and print timing."""
    result = time_func(fib_iterative, 5)
    assert result == [0, 1, 1, 2, 3, 5]
    assert "fib_iterative" in capsys.readouterr().out


def test_time_func_with_mock_callable(capsys):
    """time_func must accept any callable, including a MagicMock."""
    mock = MagicMock(return_value="mocked")
    mock.__name__ = "mock_func"
    assert time_func(mock, 1, key="value") == "mocked"
    mock.assert_called_once_with(1, key="value")
    assert "mock_func" in capsys.readouterr().out
