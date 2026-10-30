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

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the target module by file path so the test file is self-contained.
# ---------------------------------------------------------------------------
_SOURCE_PATH = Path(__file__).resolve().parent / "fibonacci.py"
if not _SOURCE_PATH.exists():
    # Fallback: try the absolute path provided by the task.
    _SOURCE_PATH = Path(
        r"D:\Documents\Study\Projects\algorithms-bench\maths\fibonacci.py"
    )

_spec = importlib.util.spec_from_file_location("fibonacci_under_test", _SOURCE_PATH)
fibonacci = importlib.util.module_from_spec(_spec)
sys.modules["fibonacci_under_test"] = fibonacci
_spec.loader.exec_module(fibonacci)


# ---------------------------------------------------------------------------
# Shared expected values
# ---------------------------------------------------------------------------
FIB_0_TO_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------
def test_time_func_returns_output():
    """time_func must return the wrapped function's return value unchanged."""
    result = fibonacci.time_func(lambda a, b: a + b, 2, 3)
    assert result == 5


def test_time_func_passes_kwargs():
    """time_func must forward keyword arguments to the wrapped function."""

    def add(a, b=0):
        return a + b

    assert fibonacci.time_func(add, 10, b=5) == 15


def test_time_func_prints_runtime(capsys):
    """time_func must print a runtime line containing the function name."""
    fibonacci.time_func(lambda: None)
    captured = capsys.readouterr()
    assert "runtime" in captured.out
    assert "<lambda>" in captured.out


def test_time_func_prints_seconds_branch(monkeypatch, capsys):
    """time_func must use the seconds format when elapsed time is >= 1 second."""
    times = iter([0.0, 2.5])
    monkeypatch.setattr(fibonacci, "time", lambda: next(times))
    fibonacci.time_func(lambda: 42)
    captured = capsys.readouterr()
    assert "s" in captured.out
    assert "ms" not in captured.out


def test_time_func_prints_milliseconds_branch(monkeypatch, capsys):
    """time_func must use the milliseconds format when elapsed time is < 1 second."""
    times = iter([0.0, 0.25])
    monkeypatch.setattr(fibonacci, "time", lambda: next(times))
    fibonacci.time_func(lambda: 42)
    captured = capsys.readouterr()
    assert "ms" in captured.out


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
def test_fib_iterative_yield_happy_path(n, expected):
    """fib_iterative_yield must yield the first n+1 Fibonacci numbers."""
    assert list(fibonacci.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_is_iterator():
    """fib_iterative_yield must return a lazy iterator, not a list."""
    gen = fibonacci.fib_iterative_yield(5)
    assert hasattr(gen, "__next__")
    assert hasattr(gen, "__iter__")


def test_fib_iterative_yield_negative_raises():
    """fib_iterative_yield must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fibonacci.fib_iterative_yield(-1))


def test_fib_iterative_yield_large_n():
    """fib_iterative_yield must produce n+1 values for a larger n."""
    values = list(fibonacci.fib_iterative_yield(50))
    assert len(values) == 51
    assert values[-1] == 12586269025


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
        (10, FIB_0_TO_10),
    ],
)
def test_fib_iterative_happy_path(n, expected):
    """fib_iterative must return the first n+1 Fibonacci numbers as a list."""
    result = fibonacci.fib_iterative(n)
    assert result == expected
    assert isinstance(result, list)


def test_fib_iterative_negative_raises():
    """fib_iterative must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_iterative(-1)


def test_fib_iterative_large_n():
    """fib_iterative must handle larger n correctly."""
    result = fibonacci.fib_iterative(30)
    assert len(result) == 31
    assert result[-1] == 832040


# ---------------------------------------------------------------------------
# fib_recursive
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
def test_fib_recursive_happy_path(n, expected):
    """fib_recursive must return the first n+1 Fibonacci numbers."""
    assert fibonacci.fib_recursive(n) == expected


def test_fib_recursive_negative_raises():
    """fib_recursive must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_recursive(-1)


def test_fib_recursive_matches_iterative():
    """fib_recursive must agree with fib_iterative for a moderate n."""
    assert fibonacci.fib_recursive(15) == fibonacci.fib_iterative(15)


# ---------------------------------------------------------------------------
# fib_recursive_cached
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
def test_fib_recursive_cached_happy_path(n, expected):
    """fib_recursive_cached must return the first n+1 Fibonacci numbers."""
    assert fibonacci.fib_recursive_cached(n) == expected


def test_fib_recursive_cached_negative_raises():
    """fib_recursive_cached must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_recursive_cached(-1)


def test_fib_recursive_cached_large_n():
    """fib_recursive_cached must handle larger n thanks to caching."""
    result = fibonacci.fib_recursive_cached(40)
    assert len(result) == 41
    assert result[-1] == 102334155


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
        (10, FIB_0_TO_10),
    ],
)
def test_fib_memoization_happy_path(n, expected):
    """fib_memoization must return the first n+1 Fibonacci numbers."""
    assert fibonacci.fib_memoization(n) == expected


def test_fib_memoization_negative_raises():
    """fib_memoization must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_memoization(-1)


def test_fib_memoization_large_n():
    """fib_memoization must handle larger n correctly."""
    result = fibonacci.fib_memoization(50)
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
        (10, FIB_0_TO_10),
    ],
)
def test_fib_binet_happy_path(n, expected):
    """fib_binet must return the first n+1 Fibonacci numbers for small n."""
    assert fibonacci.fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """fib_binet must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 2000])
def test_fib_binet_too_large_raises(n):
    """fib_binet must raise ValueError when n >= 1475 (float overflow limit)."""
    with pytest.raises(ValueError, match="n is too large"):
        fibonacci.fib_binet(n)


def test_fib_binet_boundary_1474_ok():
    """fib_binet must accept n == 1474 (just below the limit)."""
    result = fibonacci.fib_binet(1474)
    assert len(result) == 1475


def test_fib_binet_matches_iterative_small():
    """fib_binet must agree with fib_iterative for small n."""
    assert fibonacci.fib_binet(20) == fibonacci.fib_iterative(20)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------
def test_matrix_pow_np_power_zero_identity():
    """matrix_pow_np with power 0 must return the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fibonacci.matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_matrix():
    """matrix_pow_np with power 1 must return the original matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fibonacci.matrix_pow_np(m, 1)
    np.testing.assert_array_equal(result, m)


@pytest.mark.parametrize(
    "power, expected",
    [
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
        (10, [[89, 55], [55, 34]]),
    ],
)
def test_matrix_pow_np_happy_path(power, expected):
    """matrix_pow_np must compute the correct matrix power."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fibonacci.matrix_pow_np(m, power)
    np.testing.assert_array_equal(result, np.array(expected, dtype=int))


def test_matrix_pow_np_negative_raises():
    """matrix_pow_np must raise ValueError for negative power."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        fibonacci.matrix_pow_np(m, -1)


def test_matrix_pow_np_identity_matrix():
    """matrix_pow_np on the identity matrix must return the identity matrix."""
    identity = np.eye(2, dtype=int)
    result = fibonacci.matrix_pow_np(identity, 7)
    np.testing.assert_array_equal(result, identity)


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
def test_fib_matrix_np_happy_path(n, expected):
    """fib_matrix_np must return the n-th Fibonacci number."""
    result = fibonacci.fib_matrix_np(n)
    assert result == expected
    assert isinstance(result, int)


def test_fib_matrix_np_negative_raises():
    """fib_matrix_np must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fibonacci.fib_matrix_np(-1)


def test_fib_matrix_np_large_n():
    """fib_matrix_np must handle large n correctly while the result still fits
    in a signed 64-bit integer (numpy uses int64 matrices, so beyond
    fib(92) = 7540113804746346429 the intermediate products overflow).
    """
    n = 90
    expected = fibonacci.fib_iterative(n)[-1]
    assert expected == 2880067194370816120
    assert fibonacci.fib_matrix_np(n) == expected


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np must agree with fib_iterative for a range of n."""
    for n in range(0, 25):
        assert fibonacci.fib_matrix_np(n) == fibonacci.fib_iterative(n)[-1]


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 12, 20])
def test_all_implementations_agree(n):
    """All Fibonacci implementations must produce identical results for small n."""
    expected = fibonacci.fib_iterative(n)
    assert fibonacci.fib_recursive(n) == expected
    assert fibonacci.fib_recursive_cached(n) == expected
    assert fibonacci.fib_memoization(n) == expected
    assert fibonacci.fib_binet(n) == expected
    assert list(fibonacci.fib_iterative_yield(n)) == expected
    assert fibonacci.fib_matrix_np(n) == expected[-1]
