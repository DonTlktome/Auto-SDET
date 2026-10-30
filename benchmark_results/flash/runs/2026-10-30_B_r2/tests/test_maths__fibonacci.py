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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_SEARCH_ROOTS = [
    _HERE,
    *list(_HERE.parents)[:3],
    Path.cwd(),
    *list(Path.cwd().parents)[:3],
]

# Make the target module importable regardless of the test runner's CWD/layout.
for _root in _SEARCH_ROOTS:
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

fib_mod = None
try:  # preferred: the module lives inside a "maths" package
    from maths import fibonacci as fib_mod  # noqa: E402
except ImportError:
    fib_mod = None

if fib_mod is None:  # fallback: locate fibonacci.py on disk directly

    def _find_fibonacci_file():
        """Locate fibonacci.py on disk for flat (non-package) layouts."""
        candidates = []
        for root in _SEARCH_ROOTS:
            candidates.append(root / "maths" / "fibonacci.py")
            candidates.append(root / "fibonacci.py")
        tests_path = Path(__file__).resolve()
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() != tests_path:
                return candidate
        raise ImportError(
            "Could not locate fibonacci.py; searched: "
            + ", ".join(str(c) for c in candidates)
        )

    _FIB_PATH = _find_fibonacci_file()
    _SPEC = importlib.util.spec_from_file_location("fibonacci", _FIB_PATH)
    fib_mod = importlib.util.module_from_spec(_SPEC)
    sys.modules.setdefault("fibonacci", fib_mod)
    _SPEC.loader.exec_module(fib_mod)


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


@pytest.mark.parametrize("delta, expected_unit", [(0.0005, "ms"), (2.0, "s")])
def test_time_func_unit_selection(delta, expected_unit, capsys):
    """time_func prints milliseconds for sub-second runtimes and seconds otherwise."""
    with patch.object(fib_mod, "time", side_effect=[100.0, 100.0 + delta]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert expected_unit in out


def test_time_func_propagates_exceptions():
    """Exceptions raised by the wrapped callable must propagate unchanged."""

    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        fib_mod.time_func(boom)


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
def test_fib_iterative_yield_happy_path(n, expected):
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


def test_fib_iterative_yield_negative_raises_on_iteration():
    """The ValueError is raised lazily, i.e. only when the generator is consumed."""
    gen = fib_mod.fib_iterative_yield(-1)
    with pytest.raises(ValueError, match="n is negative"):
        next(gen)


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
        (10, EXPECTED_10),
    ],
)
def test_fib_iterative_happy_path(n, expected):
    """fib_iterative returns the first n+1 Fibonacci numbers as a list."""
    result = fib_mod.fib_iterative(n)
    assert result == expected
    assert isinstance(result, list)


def test_fib_iterative_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_iterative(-1)


def test_fib_iterative_large_n_length():
    """The returned list must always have exactly n+1 elements."""
    assert len(fib_mod.fib_iterative(200)) == 201


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
def test_fib_recursive_happy_path(n, expected):
    """fib_recursive returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive(n) == expected


def test_fib_recursive_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive(-1)


def test_fib_recursive_matches_iterative():
    """Recursive and iterative implementations must agree."""
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
        (10, EXPECTED_10),
    ],
)
def test_fib_recursive_cached_happy_path(n, expected):
    """fib_recursive_cached returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_recursive_cached(n) == expected


def test_fib_recursive_cached_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_recursive_cached(-1)


def test_fib_recursive_cached_handles_larger_n():
    """Caching makes larger inputs feasible; result must match the iterative one."""
    assert fib_mod.fib_recursive_cached(60) == fib_mod.fib_iterative(60)


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
def test_fib_memoization_happy_path(n, expected):
    """fib_memoization returns the first n+1 Fibonacci numbers."""
    assert fib_mod.fib_memoization(n) == expected


def test_fib_memoization_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_memoization(-1)


def test_fib_memoization_matches_iterative():
    """Memoized and iterative implementations must agree."""
    assert fib_mod.fib_memoization(40) == fib_mod.fib_iterative(40)


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
def test_fib_binet_happy_path(n, expected):
    """fib_binet returns the first n+1 Fibonacci numbers for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 5000])
def test_fib_binet_too_large_raises(n):
    """n >= 1475 must raise ValueError('n is too large')."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474_is_allowed():
    """n == 1474 is the largest accepted input and must not raise."""
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


@pytest.fixture
def fib_matrix():
    """The Fibonacci Q-matrix [[1, 1], [1, 0]] as an integer numpy array."""
    return np.array([[1, 1], [1, 0]], dtype=int)


def test_matrix_pow_np_power_zero_identity(fib_matrix):
    """Raising any matrix to the power 0 yields the identity matrix."""
    result = fib_mod.matrix_pow_np(fib_matrix, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_identity(fib_matrix):
    """Raising a matrix to the power 1 returns the matrix itself."""
    np.testing.assert_array_equal(fib_mod.matrix_pow_np(fib_matrix, 1), fib_matrix)


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
    """Binary exponentiation must produce the correct matrix power."""
    result = fib_mod.matrix_pow_np(fib_matrix, power)
    np.testing.assert_array_equal(result, np.array(expected, dtype=int))


def test_matrix_pow_np_negative_power_raises(fib_matrix):
    """A negative power must raise ValueError('power is negative')."""
    with pytest.raises(ValueError, match="power is negative"):
        fib_mod.matrix_pow_np(fib_matrix, -1)


def test_matrix_pow_np_identity_input():
    """The identity matrix raised to any power is still the identity."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(fib_mod.matrix_pow_np(identity, 7), identity)


def test_matrix_pow_np_does_not_mutate_input(fib_matrix):
    """matrix_pow_np must not modify the caller's matrix in place."""
    original = fib_matrix.copy()
    fib_mod.matrix_pow_np(fib_matrix, 6)
    np.testing.assert_array_equal(fib_matrix, original)


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
    result = fib_mod.fib_matrix_np(n)
    assert result == expected
    assert isinstance(result, int)


def test_fib_matrix_np_negative_raises():
    """A negative n must raise ValueError('n is negative')."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(-1)


def test_fib_matrix_np_matches_iterative():
    """The matrix method must agree with the iterative implementation."""
    n = 50
    assert fib_mod.fib_matrix_np(n) == fib_mod.fib_iterative(n)[-1]


def test_fib_matrix_np_large_n():
    """The matrix method must stay exact for large n within the int64 range.

    The implementation uses numpy's default integer dtype (int64), so the
    largest exactly representable Fibonacci number is F(92) = 7540113804746346429.
    F(93) already overflows a signed 64-bit integer, hence F(92) is used here as
    the exactness boundary to verify no floating-point precision is involved.
    """
    expected = 7540113804746346429  # F(92), largest Fibonacci fitting in int64
    assert fib_mod.fib_matrix_np(92) == expected
    assert fib_mod.fib_matrix_np(92) == fib_mod.fib_iterative(92)[-1]


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", LIST_FUNCS)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 12])
def test_all_list_implementations_agree(func, n):
    """Every list-returning implementation must produce identical output."""
    assert func(n) == fib_mod.fib_iterative(n)


@pytest.mark.parametrize("func", LIST_FUNCS)
def test_all_list_implementations_reject_negative(func):
    """Every list-returning implementation must reject negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        func(-5)


def test_yield_and_list_implementations_agree():
    """The generator version must match the list version."""
    assert list(fib_mod.fib_iterative_yield(12)) == fib_mod.fib_iterative(12)


# ---------------------------------------------------------------------------
# Isolation / mocking sanity check
# ---------------------------------------------------------------------------


def test_time_func_with_magicmock():
    """time_func must work with a MagicMock callable and return its value."""
    mock = MagicMock(return_value=[0, 1, 1])
    mock.__name__ = "mock_fib"
    assert fib_mod.time_func(mock, 2) == [0, 1, 1]
    mock.assert_called_once_with(2)
