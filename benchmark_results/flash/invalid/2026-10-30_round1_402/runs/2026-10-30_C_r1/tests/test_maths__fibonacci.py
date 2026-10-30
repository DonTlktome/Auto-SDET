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

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the target module regardless of the test runner's cwd / import mode.
# ---------------------------------------------------------------------------

def _load_fib_module():
    """Import the fibonacci module, falling back to a direct file load."""
    here = Path(__file__).resolve().parent

    # Collect every plausible root and make it importable.
    search_roots = []
    for root in (here, Path.cwd(), *here.parents):
        if root not in search_roots:
            search_roots.append(root)
    for root in search_roots:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    # 1) Try normal imports (package form first, then bare module).
    for module_name in ("maths.fibonacci", "fibonacci"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

    # 2) Fallback: locate fibonacci.py on disk near the test file / cwd.
    candidates = []
    seen = set()

    def _add(candidate: Path) -> None:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(candidate)

    for root in search_roots:
        _add(root / "fibonacci.py")
        _add(root / "maths" / "fibonacci.py")
        _add(root / "math" / "fibonacci.py")
        if root.is_dir():
            try:
                children = sorted(root.iterdir())
            except OSError:  # pragma: no cover - unreadable directory
                children = []
            for child in children:
                if child.is_dir():
                    _add(child / "fibonacci.py")

    for candidate in candidates:
        spec = importlib.util.spec_from_file_location("fibonacci", candidate)
        if spec is None or spec.loader is None:  # pragma: no cover
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["fibonacci"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - try the next candidate
            sys.modules.pop("fibonacci", None)
            continue
        return module

    raise ModuleNotFoundError("Could not locate fibonacci.py")


fib_mod = _load_fib_module()


# ---------------------------------------------------------------------------
# Shared reference data
# ---------------------------------------------------------------------------

# First 11 Fibonacci numbers (0-indexed): F(0) .. F(10)
FIB_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

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

    assert fib_mod.time_func(add, 10, b=5) == 15


def test_time_func_prints_runtime(capsys):
    """time_func must print a runtime line containing the function name."""
    fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "runtime:" in out
    assert "<lambda>" in out


def test_time_func_prints_seconds_when_slow(capsys):
    """time_func must report seconds when the elapsed time exceeds 1 second."""
    # Force a >1s elapsed time by patching the module's `time` symbol.
    with patch.object(fib_mod, "time", side_effect=[0.0, 2.5]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "s" in out
    assert "2.5000 s" in out


def test_time_func_prints_milliseconds_when_fast(capsys):
    """time_func must report milliseconds when the elapsed time is under 1s."""
    with patch.object(fib_mod, "time", side_effect=[0.0, 0.002]):
        fib_mod.time_func(lambda: None)
    out = capsys.readouterr().out
    assert "ms" in out
    assert "2.0000 ms" in out


def test_time_func_propagates_exceptions():
    """time_func must not swallow exceptions raised by the wrapped callable."""

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
        (10, tuple(FIB_10)),
    ],
)
def test_fib_iterative_yield_happy_path(n, expected):
    """fib_iterative_yield yields the first n+1 Fibonacci numbers."""
    assert tuple(fib_mod.fib_iterative_yield(n)) == expected


def test_fib_iterative_yield_is_lazy_iterator():
    """fib_iterative_yield must return a lazy iterator, not a materialized list."""
    gen = fib_mod.fib_iterative_yield(5)
    assert iter(gen) is gen
    assert next(gen) == 0
    assert next(gen) == 1


def test_fib_iterative_yield_negative_raises():
    """fib_iterative_yield must raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        # The generator body only runs on first iteration.
        list(fib_mod.fib_iterative_yield(-1))


# ---------------------------------------------------------------------------
# fib_iterative / fib_recursive / fib_recursive_cached / fib_memoization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", LIST_FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 1, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIB_10),
    ],
)
def test_list_functions_happy_path(func, n, expected):
    """All list-returning implementations agree on the first n+1 numbers."""
    assert func(n) == expected


@pytest.mark.parametrize("func", LIST_FUNCS, ids=lambda f: f.__name__)
def test_list_functions_negative_raises(func):
    """All list-returning implementations raise ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        func(-1)


@pytest.mark.parametrize("func", LIST_FUNCS, ids=lambda f: f.__name__)
def test_list_functions_length(func):
    """All list-returning implementations return exactly n+1 elements."""
    assert len(func(7)) == 8


@pytest.mark.parametrize("func", LIST_FUNCS, ids=lambda f: f.__name__)
def test_list_functions_agree_with_iterative(func):
    """Every implementation must match fib_iterative for a mid-size n."""
    assert func(20) == fib_mod.fib_iterative(20)


def test_fib_iterative_large_n():
    """fib_iterative must handle a moderately large n correctly."""
    result = fib_mod.fib_iterative(50)
    assert len(result) == 51
    assert result[-1] == 12586269025


def test_fib_recursive_cached_large_n():
    """fib_recursive_cached must remain fast/correct for larger n."""
    result = fib_mod.fib_recursive_cached(40)
    assert result[-1] == 102334155


def test_fib_memoization_large_n():
    """fib_memoization must remain correct for larger n."""
    result = fib_mod.fib_memoization(40)
    assert result[-1] == 102334155


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, [0]),
        (1, [0, 1]),
        (5, [0, 1, 1, 2, 3, 5]),
        (10, FIB_10),
    ],
)
def test_fib_binet_happy_path(n, expected):
    """fib_binet returns the first n+1 Fibonacci numbers for small n."""
    assert fib_mod.fib_binet(n) == expected


def test_fib_binet_negative_raises():
    """fib_binet raises ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_binet(-1)


@pytest.mark.parametrize("n", [1475, 1476, 5000])
def test_fib_binet_too_large_raises(n):
    """fib_binet raises ValueError when n >= 1475 (float overflow limit)."""
    with pytest.raises(ValueError, match="n is too large"):
        fib_mod.fib_binet(n)


def test_fib_binet_boundary_1474_ok():
    """fib_binet accepts the largest allowed value, n = 1474."""
    result = fib_mod.fib_binet(1474)
    assert len(result) == 1475
    assert result[0] == 0
    assert result[1] == 1


def test_fib_binet_matches_iterative_below_divergence():
    """fib_binet agrees with fib_iterative for n well below the divergence point."""
    assert fib_mod.fib_binet(60) == fib_mod.fib_iterative(60)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------


def test_matrix_pow_np_power_zero_identity():
    """Raising a matrix to the power 0 yields the identity matrix."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, 0)
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))


def test_matrix_pow_np_power_one_identity_matrix():
    """Raising a matrix to the power 1 returns the matrix itself."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, 1)
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
    """matrix_pow_np computes the correct matrix power via binary exponentiation."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    result = fib_mod.matrix_pow_np(m, power)
    np.testing.assert_array_equal(result, np.array(expected, dtype=int))


def test_matrix_pow_np_negative_raises():
    """matrix_pow_np raises ValueError for a negative power."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    with pytest.raises(ValueError, match="power is negative"):
        fib_mod.matrix_pow_np(m, -1)


def test_matrix_pow_np_does_not_mutate_input():
    """matrix_pow_np must not modify the input matrix in place."""
    m = np.array([[1, 1], [1, 0]], dtype=int)
    original = m.copy()
    fib_mod.matrix_pow_np(m, 7)
    np.testing.assert_array_equal(m, original)


def test_matrix_pow_np_identity_input():
    """Raising the identity matrix to any power returns the identity matrix."""
    identity = np.array([[1, 0], [0, 1]], dtype=int)
    result = fib_mod.matrix_pow_np(identity, 42)
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
    """fib_matrix_np returns the n-th Fibonacci number."""
    assert fib_mod.fib_matrix_np(n) == expected


def test_fib_matrix_np_returns_int():
    """fib_matrix_np must return a plain Python int, not a numpy scalar."""
    result = fib_mod.fib_matrix_np(10)
    assert isinstance(result, int)
    assert not isinstance(result, np.integer)


def test_fib_matrix_np_negative_raises():
    """fib_matrix_np raises ValueError for negative n."""
    with pytest.raises(ValueError, match="n is negative"):
        fib_mod.fib_matrix_np(-1)


def test_fib_matrix_np_large_n():
    """fib_matrix_np handles large n correctly within numpy's integer range."""
    # fib_matrix_np relies on numpy integer matrices (dtype=int).  F(93) already
    # exceeds the int64 range, so F(92) is the largest exactly representable
    # Fibonacci number for this implementation.
    assert fib_mod.fib_matrix_np(92) == 7540113804746346429


def test_fib_matrix_np_wraps_beyond_integer_range():
    """Beyond the numpy integer range fib_matrix_np results wrap modulo 2**bits."""
    int_dtype = np.array([[1, 1], [1, 0]], dtype=int).dtype
    modulus = 2 ** np.iinfo(int_dtype).bits
    expected = fib_mod.fib_iterative(100)[-1] % modulus
    assert fib_mod.fib_matrix_np(100) == expected


def test_fib_matrix_np_matches_iterative():
    """fib_matrix_np agrees with fib_iterative for a range of n values."""
    for n in range(0, 25):
        assert fib_mod.fib_matrix_np(n) == fib_mod.fib_iterative(n)[-1]


def test_fib_matrix_np_uses_matrix_pow_np():
    """fib_matrix_np delegates to matrix_pow_np (isolation via mock)."""
    fake_result = np.array([[55, 34], [34, 21]], dtype=int)
    with patch.object(
        fib_mod, "matrix_pow_np", MagicMock(return_value=fake_result)
    ) as mocked:
        assert fib_mod.fib_matrix_np(10) == 55
    mocked.assert_called_once()
    called_matrix, called_power = mocked.call_args.args
    np.testing.assert_array_equal(called_matrix, np.array([[1, 1], [1, 0]], dtype=int))
    assert called_power == 9


def test_fib_matrix_np_zero_skips_matrix_pow():
    """fib_matrix_np(0) returns 0 without calling matrix_pow_np."""
    with patch.object(fib_mod, "matrix_pow_np") as mocked:
        assert fib_mod.fib_matrix_np(0) == 0
    mocked.assert_not_called()
