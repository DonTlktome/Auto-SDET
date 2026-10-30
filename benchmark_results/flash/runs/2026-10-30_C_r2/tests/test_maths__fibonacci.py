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


# ---------------------------------------------------------------------------
# Shared reference data
# ---------------------------------------------------------------------------

# First 21 Fibonacci numbers (0-indexed): F(0) .. F(20)
FIB_20 = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
    987, 1597, 2584, 4181, 6765,
]

# Functions returning the first n (0-indexed) Fibonacci numbers as a list.
LIST_FUNCS = [
    fib_mod.fib_iterative,
    fib_mod.fib_recursive,
    fib_mod.fib_recursive_cached,
    fib_mod.fib_memoization,
    fib_mod.fib_binet,
]

# fib_matrix_np uses numpy arrays with dtype=int, i.e. int64 on most platforms.
# Results above INT64_MAX (F(93)) overflow, so only n <= 92 is compared exactly.
INT64_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------


def test_time_func_returns_output():
    """time_func must return whatever the wrapped callable returns."""
    assert fib_mod.time_func(lambda: 42) == 42


def test_time_func_passes_args_and_kwargs():
    """time_func must forward positional and keyword arguments."""

    def add(a, b, c=0):
        return a + b + c

    assert fib_mod.time_func(add, 1, 2, c=3) == 6


def test_time_func_prints_milliseconds_branch(capsys):
    """When elapsed time is < 1s, the runtime is reported in milliseconds."""
    with patch.object(fib_mod, "time", side_effect=[10.0, 10.25]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "ms" in out
    assert "250.0000 ms" in out


def test_time_func_prints_seconds_branch(capsys):
    """When elapsed time is >= 1s, the runtime is reported in seconds."""
    with patch.object(fib_mod, "time", side_effect=[10.0, 12.5]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "s" in out
    assert "2.5000 s" in out


def test_time_func_uses_function_name(capsys):
    """The printed message must contain the wrapped function's __name__."""

    def my_named_func():
        return None

    with patch.object(fib_mod, "time", side_effect=[0.0, 0.001]):
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
        (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        (20, FIB_20),
    ],
)
def test_fib_iterative_yield_happy_path(n, expected):
    """fib_iterative_yield yields the first n+1 Fibonacci numbers."""
    assert list(fib_mod.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_returns_iterator():
    """fib_iterative_yield must return a lazy iterator, not a list."""
    result = fib_mod.fib_iterative_yield(5)
    assert not isinstance(result, list)
    assert iter(result) is result


def test_fib_iterative_yield_is_lazy():
    """Consuming only part of the generator must not exhaust it eagerly."""
    gen = fib_mod.fib_iterative_yield(10)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 1


@pytest.mark.parametrize("n", [-1, -5, -100])
def test_fib_iterative_yield_negative_raises(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_mod.fib_iterative_yield(n))


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
def test_list_functions_happy_path(func, n, expected):
    """All list-returning implementations agree on the first n+1 numbers."""
    assert func(n) == expected


@pytest.mark.parametrize("func", LIST_FUNCS)
@pytest.mark.parametrize("n", [-1, -2, -50])
def test_list_functions_negative_raises(func, n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        func(n)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_return_list_of_ints(func):
    """Every implementation must return a list of plain ints."""
    result = func(10)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_length_is_n_plus_one(func):
    """The returned list must contain exactly n + 1 elements."""
    for n in (0, 1, 3, 7, 15):
        assert len(func(n)) == n + 1


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_are_strictly_increasing_after_first(func):
    """From index 2 onward the sequence is non-decreasing and grows."""
    seq = func(15)
    assert seq[0] == 0
    assert seq[1] == 1
    for a, b in zip(seq[1:], seq[2:]):
        assert b >= a


def test_fib_iterative_large_n():
    """fib_iterative must handle a moderately large n correctly."""
    seq = fib_mod.fib_iterative(100)
    assert len(seq) == 101
    assert seq[100] == 354224848179261915075


def test_fib_recursive_cached_large_n():
    """fib_recursive_cached must handle a moderately large n correctly."""
    seq = fib_mod.fib_recursive_cached(100)
    assert seq[100] == 354224848179261915075


def test_fib_memoization_large_n():
    """fib_memoization must handle a moderately large n correctly."""
    seq = fib_mod.fib_memoization(100)
    assert seq[100] == 354224848179261915075


def test_fib_recursive_small_n_only():
    """fib_recursive is exponential; verify it on a small input."""
    assert fib_mod.fib_recursive(12) == FIB_20[:13]


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
    ],
)
def test_fib_binet_happy_path(n, expected):
    """fib_binet matches the exact sequence for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_matches_iterative_up_to_70():
    """Binet's formula is exact (per docstring) up to about n = 70."""
    assert fib_mod.fib_binet(70) == fib_mod.fib_iterative(70)


@pytest.mark.parametrize("n", [-1, -10])
def test_fib_binet_negative_raises(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(n)


@pytest.mark.parametrize("n", [1475, 1476, 5000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474_allowed():
    """n = 1474 is the largest accepted input and must not raise."""
    result = fib_mod.fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "power, expected",
    [
        (0, [[1, 0], [0, 1]]),
        (1, [[1, 1], [1, 0]]),
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
        (10, [[89, 55], [55, 34]]),
    ],
)
def test_matrix_pow_np_happy_path(power, expected):
    """matrix_pow_np raises the Fibonacci matrix to the given power."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, power)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(expected, dtype=int))


def test_matrix_pow_np_identity_matrix():
    """Raising the identity matrix to any power returns the identity."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    result = fib_mod.matrix_pow_np(identity, 7)
    np.testing.assert_array_equal(result, identity)


def test_matrix_pow_np_does_not_mutate_input():
    """The input matrix must not be modified in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    fib_mod.matrix_pow_np(m, 6)
    np.testing.assert_array_equal(m, original)


@pytest.mark.parametrize("power", [-1, -2, -100])
def test_matrix_pow_np_negative_power_raises(power):
    """Negative powers must raise ValueError('power is negative')."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        fib_mod.matrix_pow_np(m, power)


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
        (50, 12586269025),
        # Largest exactly representable values: the numpy matrices are int64,
        # so n > 92 would overflow (e.g. F(100) does not fit in int64).
        (90, 2880067194370816120),
        (92, 7540113804746346429),
    ],
)
def test_fib_matrix_np_happy_path(n, expected):
    """fib_matrix_np returns the n-th Fibonacci number (within int64 range)."""
    assert fib_mod.fib_matrix_np(n) == expected
    assert expected <= INT64_MAX


def test_fib_matrix_np_returns_int():
    """The return value must be a plain Python int, not a numpy scalar."""
    result = fib_mod.fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np agrees with fib_iterative for a range of n."""
    seq = fib_mod.fib_iterative(40)
    for n in range(0, 41, 5):
        assert fib_mod.fib_matrix_np(n) == seq[n]


def test_fib_matrix_np_matches_iterative_up_to_int64_limit():
    """fib_matrix_np stays exact right up to the int64 boundary (n = 92)."""
    seq = fib_mod.fib_iterative(92)
    for n in (80, 85, 90, 91, 92):
        assert fib_mod.fib_matrix_np(n) == seq[n]


@pytest.mark.parametrize("n", [-1, -2, -100])
def test_fib_matrix_np_negative_raises(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(n)


def test_fib_matrix_np_zero_short_circuits():
    """n == 0 must return 0 without calling matrix_pow_np."""
    with patch.object(fib_mod, "matrix_pow_np", MagicMock()) as mocked:
        assert fib_mod.fib_matrix_np(0) == 0
    mocked.assert_not_called()


def test_fib_matrix_np_calls_matrix_pow_np_with_n_minus_one():
    """For n > 0, matrix_pow_np is invoked with power == n - 1."""
    with patch.object(
        fib_mod, "matrix_pow_np", wraps=fib_mod.matrix_pow_np
    ) as mocked:
        fib_mod.fib_matrix_np(7)
    mocked.assert_called_once()
    assert mocked.call_args.args[1] == 6


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 12, 25])
def test_all_implementations_agree(n):
    """Every implementation must produce the same sequence for the same n."""
    reference = fib_mod.fib_iterative(n)
    assert list(fib_mod.fib_iterative_yield(n)) == reference
    assert fib_mod.fib_recursive(n) == reference
    assert fib_mod.fib_recursive_cached(n) == reference
    assert fib_mod.fib_memoization(n) == reference
    assert fib_mod.fib_binet(n) == reference
    assert fib_mod.fib_matrix_np(n) == reference[n]
