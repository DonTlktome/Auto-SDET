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

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fibonacci as fib_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

EXPECTED_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
EXPECTED_5 = [0, 1, 1, 2, 3, 5]

# All functions that return the first n (0-indexed) Fibonacci numbers as a list.
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


def test_time_func_seconds_branch(capsys):
    """When elapsed time is >= 1s, the message must be reported in seconds."""
    fake_clock = MagicMock(side_effect=[0.0, 2.5])
    with patch.object(fib_mod, "time", fake_clock):
        fib_mod.time_func(lambda: "ok")
    out = capsys.readouterr().out
    assert "s" in out
    assert "2.5000 s" in out


def test_time_func_milliseconds_branch(capsys):
    """When elapsed time is < 1s, the message must be reported in milliseconds."""
    fake_clock = MagicMock(side_effect=[0.0, 0.25])
    with patch.object(fib_mod, "time", fake_clock):
        fib_mod.time_func(lambda: "ok")
    out = capsys.readouterr().out
    assert "250.0000 ms" in out


def test_time_func_propagates_exceptions():
    """Exceptions raised by the wrapped callable must propagate unchanged."""

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fib_mod.time_func(boom)


# ---------------------------------------------------------------------------
# fib_iterative_yield
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, (0,)),
        (1, (0, 1)),
        (2, (0, 1, 1)),
        (5, (0, 1, 1, 2, 3, 5)),
        (10, tuple(EXPECTED_10)),
    ],
)
def test_fib_iterative_yield_values(n, expected):
    """fib_iterative_yield must yield the first n+1 Fibonacci numbers."""
    assert tuple(fib_mod.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_is_iterator():
    """fib_iterative_yield must return a lazy iterator, not a list."""
    gen = fib_mod.fib_iterative_yield(5)
    assert iter(gen) is gen
    assert not isinstance(gen, list)


def test_fib_iterative_yield_lazy():
    """The generator must not compute values until iterated."""
    gen = fib_mod.fib_iterative_yield(1000)
    assert next(gen) == 0
    assert next(gen) == 1


@pytest.mark.parametrize("n", [-1, -5, -100])
def test_fib_iterative_yield_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_mod.fib_iterative_yield(n))


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, EXPECTED_5),
        (10, EXPECTED_10),
    ],
)
def test_fib_iterative_values(n, expected):
    """fib_iterative must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_iterative(n) == expected


def test_fib_iterative_returns_list():
    """fib_iterative must return a plain list of ints."""
    result = fib_mod.fib_iterative(3)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_fib_iterative_large():
    """fib_iterative must handle moderately large n correctly."""
    result = fib_mod.fib_iterative(50)
    assert len(result) == 51
    assert result[-1] == 12586269025


@pytest.mark.parametrize("n", [-1, -10])
def test_fib_iterative_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_iterative(n)


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, EXPECTED_5),
        (10, EXPECTED_10),
    ],
)
def test_fib_recursive_values(n, expected):
    """fib_recursive must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive(n) == expected


@pytest.mark.parametrize("n", [-1, -3])
def test_fib_recursive_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive(n)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, EXPECTED_5),
        (10, EXPECTED_10),
    ],
)
def test_fib_recursive_cached_values(n, expected):
    """fib_recursive_cached must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive_cached(n) == expected


def test_fib_recursive_cached_large():
    """The cached variant must handle larger n without blowing the stack."""
    result = fib_mod.fib_recursive_cached(100)
    assert len(result) == 101
    assert result[-1] == 354224848179261915075


@pytest.mark.parametrize("n", [-1, -2])
def test_fib_recursive_cached_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive_cached(n)


# ---------------------------------------------------------------------------
# fib_memoization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, EXPECTED_5),
        (10, EXPECTED_10),
    ],
)
def test_fib_memoization_values(n, expected):
    """fib_memoization must return the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_memoization(n) == expected


def test_fib_memoization_large():
    """fib_memoization must handle larger n correctly."""
    result = fib_mod.fib_memoization(60)
    assert len(result) == 61
    assert result[-1] == 1548008755920


@pytest.mark.parametrize("n", [-1, -7])
def test_fib_memoization_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_memoization(n)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, EXPECTED_5),
        (10, EXPECTED_10),
    ],
)
def test_fib_binet_values(n, expected):
    """fib_binet must return the first n+1 Fibonacci numbers for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_matches_iterative_small():
    """Binet's formula must agree with the exact iterative version for n <= 70."""
    assert fib_mod.fib_binet(70) == fib_mod.fib_iterative(70)


@pytest.mark.parametrize("n", [-1, -100])
def test_fib_binet_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(n)


@pytest.mark.parametrize("n", [1475, 1476, 5000])
def test_fib_binet_too_large(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474():
    """n == 1474 is the largest accepted input and must not raise."""
    result = fib_mod.fib_binet(1474)
    assert len(result) == 1475


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_zero_is_identity():
    """Raising a matrix to the power 0 must return the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_one_is_same():
    """Raising a matrix to the power 1 must return the same matrix."""
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
    np.testing.assert_array_equal(
        fib_mod.matrix_pow_np(m, power), np.array(expected, dtype=int)
    )


def test_matrix_pow_np_identity_input():
    """The identity matrix raised to any power must remain the identity."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(
        fib_mod.matrix_pow_np(identity, 7), identity
    )


def test_matrix_pow_np_does_not_mutate_input():
    """matrix_pow_np must not modify the input matrix in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    fib_mod.matrix_pow_np(m, 4)
    np.testing.assert_array_equal(m, original)


@pytest.mark.parametrize("power", [-1, -2, -100])
def test_matrix_pow_np_negative_power(power):
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
    ],
)
def test_fib_matrix_np_values(n, expected):
    """fib_matrix_np must return the n-th Fibonacci number."""
    assert fib_mod.fib_matrix_np(n) == expected


def test_fib_matrix_np_returns_int():
    """fib_matrix_np must return a Python int, not a numpy scalar."""
    result = fib_mod.fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_large():
    """fib_matrix_np must handle large n efficiently and exactly."""
    # The numpy int64 matrix implementation overflows beyond n ~ 92, so we
    # verify exactness only up to the largest n that fits in int64.
    assert fib_mod.fib_matrix_np(90) == 2880067194370816120


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np must agree with fib_iterative for a range of n."""
    for n in range(0, 30):
        assert fib_mod.fib_matrix_np(n) == fib_mod.fib_iterative(n)[-1]


@pytest.mark.parametrize("n", [-1, -10])
def test_fib_matrix_np_negative(n):
    """Negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(n)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10, 15, 20])
def test_all_list_implementations_agree(n):
    """All list-returning implementations must produce identical results."""
    results = [func(n) for func in LIST_FUNCS]
    for result in results[1:]:
        assert result == results[0]


@pytest.mark.parametrize("n", [0, 1, 5, 10, 25])
def test_matrix_matches_list_implementations(n):
    """fib_matrix_np(n) must equal the last element of the list implementations."""
    expected = fib_mod.fib_iterative(n)[-1]
    assert fib_mod.fib_matrix_np(n) == expected
