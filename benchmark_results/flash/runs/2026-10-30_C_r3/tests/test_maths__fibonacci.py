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

FIRST_11 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Every function that returns the first n (0-indexed) Fibonacci numbers.
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


def test_time_func_returns_function_output():
    """time_func must return whatever the wrapped callable returns."""
    assert time_func(lambda a, b: a + b, 2, 3) == 5


def test_time_func_passes_kwargs():
    """time_func must forward keyword arguments to the wrapped callable."""

    def add(a, b=0):
        return a + b

    assert time_func(add, 1, b=41) == 42


def test_time_func_prints_milliseconds_when_fast(capsys):
    """A sub-second call must be reported in milliseconds (ms suffix)."""
    time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert "ms" in out


def test_time_func_prints_seconds_when_slow(capsys):
    """A call taking >= 1 second must be reported in seconds (s suffix)."""
    # Simulate a slow call by patching the module-level `time` function.
    with patch.object(fib_mod, "time", side_effect=[100.0, 102.5]):
        time_func(lambda: "done")
    out = capsys.readouterr().out
    assert "runtime: 2.5000 s" in out


def test_time_func_uses_function_name_in_output(capsys):
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
        (10, FIRST_11),
    ],
)
def test_fib_iterative_yield_happy_path(n, expected):
    """fib_iterative_yield yields the first n+1 Fibonacci numbers."""
    assert list(fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_returns_iterator():
    """The function must be a generator (lazy iterator), not a list."""
    result = fib_iterative_yield(5)
    assert hasattr(result, "__next__")
    assert hasattr(result, "__iter__")
    assert not isinstance(result, list)


def test_fib_iterative_yield_is_lazy():
    """Consuming only part of the generator must not raise or over-produce."""
    gen = fib_iterative_yield(100)
    assert next(gen) == 0
    assert next(gen) == 1
    assert next(gen) == 1


def test_fib_iterative_yield_negative_raises():
    """A negative n must raise ValueError with the documented message."""
    with pytest.raises(ValueError, match="n is negative"):
        list(fib_iterative_yield(-1))


def test_fib_iterative_yield_negative_raises_on_first_next():
    """The ValueError is raised lazily, on the first iteration step."""
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
        (10, FIRST_11),
    ],
)
def test_list_functions_happy_path(func, n, expected):
    """All list-returning implementations agree on small inputs."""
    assert func(n) == expected


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_negative_raises(func):
    """All list-returning implementations reject negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        func(-1)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_length_is_n_plus_one(func):
    """The returned list always has exactly n + 1 elements."""
    for n in (0, 1, 3, 7):
        assert len(func(n)) == n + 1


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_list_functions_are_consistent_with_each_other(func):
    """Every implementation must produce identical output for n = 20."""
    assert func(20) == fib_iterative(20)


def test_fib_iterative_large_value():
    """fib_iterative must compute exact big integers (n = 100)."""
    result = fib_iterative(100)
    assert len(result) == 101
    assert result[100] == 354224848179261915075


def test_fib_iterative_returns_new_list_each_call():
    """Repeated calls must not share mutable state."""
    first = fib_iterative(5)
    first.append(999)
    assert fib_iterative(5) == [0, 1, 1, 2, 3, 5]


def test_fib_recursive_zero_boundary():
    """fib_recursive(0) is the smallest valid input and returns [0]."""
    assert fib_recursive(0) == [0]


def test_fib_recursive_cached_repeated_calls_are_stable():
    """The internal functools.cache must not leak state between calls."""
    assert fib_recursive_cached(10) == FIRST_11
    assert fib_recursive_cached(10) == FIRST_11
    assert fib_recursive_cached(0) == [0]


def test_fib_memoization_prefilled_cache_boundary():
    """fib_memoization relies on a prefilled cache; verify indices 0..2."""
    assert fib_memoization(2) == [0, 1, 1]


def test_fib_memoization_large_value():
    """fib_memoization must compute exact big integers (n = 50)."""
    assert fib_memoization(50)[50] == 12586269025


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
def test_fib_binet_happy_path(n, expected):
    """Binet's formula matches the exact sequence for small n."""
    assert fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """Negative n must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 2000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_binet(n)


def test_fib_binet_upper_boundary_is_allowed():
    """n = 1474 is the largest accepted input and must not raise."""
    result = fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


def test_fib_binet_negative_checked_before_size():
    """A negative n must report 'n is negative', not 'n is too large'."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_binet(-1475)


def test_fib_binet_returns_ints():
    """Binet's formula rounds floats, so results must be Python ints."""
    result = fib_binet(10)
    assert all(isinstance(x, int) for x in result)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


@pytest.fixture()
def fib_matrix():
    """The Fibonacci Q-matrix [[1, 1], [1, 0]] as an integer ndarray."""
    return np.array([[1, 1], [1, 0]], dtype=int)


def test_matrix_pow_np_power_zero_returns_identity(fib_matrix):
    """Raising any matrix to the power 0 yields the identity matrix."""
    result = matrix_pow_np(fib_matrix, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_returns_same_matrix(fib_matrix):
    """Raising a matrix to the power 1 returns the matrix itself."""
    np.testing.assert_array_equal(matrix_pow_np(fib_matrix, 1), fib_matrix)


@pytest.mark.parametrize(
    "power, expected",
    [
        (2, [[2, 1], [1, 1]]),
        (3, [[3, 2], [2, 1]]),
        (5, [[8, 5], [5, 3]]),
        (10, [[89, 55], [55, 34]]),
    ],
)
def test_matrix_pow_np_happy_path(fib_matrix, power, expected):
    """Binary exponentiation matches known powers of the Q-matrix."""
    np.testing.assert_array_equal(
        matrix_pow_np(fib_matrix, power), np.array(expected, dtype=int)
    )


def test_matrix_pow_np_negative_power_raises(fib_matrix):
    """A negative power must raise ValueError."""
    with pytest.raises(ValueError, match="power is negative"):
        matrix_pow_np(fib_matrix, -1)


def test_matrix_pow_np_identity_matrix_any_power():
    """The identity matrix raised to any power is the identity matrix."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(matrix_pow_np(identity, 7), identity)


def test_matrix_pow_np_does_not_mutate_input(fib_matrix):
    """The input matrix must not be modified in place."""
    original = fib_matrix.copy()
    matrix_pow_np(fib_matrix, 6)
    np.testing.assert_array_equal(fib_matrix, original)


def test_matrix_pow_np_returns_ndarray(fib_matrix):
    """The return type must be a numpy ndarray."""
    assert isinstance(matrix_pow_np(fib_matrix, 4), np.ndarray)


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
def test_fib_matrix_np_happy_path(n, expected):
    """fib_matrix_np returns the n-th Fibonacci number."""
    assert fib_matrix_np(n) == expected


def test_fib_matrix_np_negative_raises():
    """A negative index must raise ValueError."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_matrix_np(-1)


def test_fib_matrix_np_zero_short_circuits():
    """n = 0 must return 0 without invoking matrix exponentiation."""
    with patch.object(fib_mod, "matrix_pow_np") as mock_pow:
        assert fib_matrix_np(0) == 0
    mock_pow.assert_not_called()


def test_fib_matrix_np_calls_matrix_pow_with_n_minus_one():
    """For n >= 1, matrix_pow_np must be called with power n - 1."""
    with patch.object(
        fib_mod, "matrix_pow_np", return_value=np.array([[13, 8], [8, 5]], dtype=int)
    ) as mock_pow:
        result = fib_matrix_np(7)
    mock_pow.assert_called_once()
    assert mock_pow.call_args[0][1] == 6
    assert result == 13


def test_fib_matrix_np_returns_python_int():
    """The result must be a plain Python int, not a numpy scalar."""
    result = fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_matches_iterative_for_large_n():
    """fib_matrix_np must agree with fib_iterative for a large index.

    NOTE: fib_matrix_np builds its Q-matrix with numpy `int` (int64 on
    typical platforms), so it silently overflows once the Fibonacci
    number exceeds 2**63 - 1, i.e. for n > 92.  The largest index that
    still fits in a signed 64-bit integer is n = 92
    (fib(92) = 7540113804746346429), so n = 90 is used here.
    """
    n = 90
    assert fib_matrix_np(n) == fib_iterative(n)[n]


def test_fib_matrix_np_uses_mocked_matrix_pow():
    """Isolate numpy work by mocking matrix_pow_np entirely."""
    fake = MagicMock(return_value=np.array([[1, 1], [1, 0]], dtype=int))
    with patch.object(fib_mod, "matrix_pow_np", fake):
        assert fib_matrix_np(1) == 1
    fake.assert_called_once()
