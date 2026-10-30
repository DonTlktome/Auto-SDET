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

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  The source file lives in a directory named
# "maths" which is not a package, so we load it directly from its path.
# ---------------------------------------------------------------------------
SOURCE_PATH = Path(__file__).resolve().parent / "fibonacci.py"
if not SOURCE_PATH.exists():
    # Fallback: try the absolute path provided by the task.
    SOURCE_PATH = Path(
        r"D:\Documents\Study\Projects\algorithms-bench\maths\fibonacci.py"
    )


def _load_module():
    spec = importlib.util.spec_from_file_location("fibonacci_under_test", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fibonacci_under_test"] = module
    spec.loader.exec_module(module)
    return module


fib = _load_module()


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
FIB_0_TO_10 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------
class TestTimeFunc:
    """Tests for the time_func helper."""

    def test_returns_function_output(self):
        """time_func must return whatever the wrapped function returns."""
        assert fib.time_func(lambda a, b: a + b, 2, 3) == 5

    def test_passes_kwargs(self):
        """time_func must forward keyword arguments to the wrapped function."""

        def add(a, b=0):
            return a + b

        assert fib.time_func(add, 1, b=41) == 42

    def test_prints_runtime(self, capsys):
        """time_func must print a runtime line mentioning the function name."""
        fib.time_func(lambda: None)
        captured = capsys.readouterr()
        assert "runtime:" in captured.out
        assert "<lambda>" in captured.out

    @pytest.mark.parametrize("elapsed, expected_unit", [(0.5, "ms"), (2.0, "s")])
    def test_prints_correct_unit(self, capsys, elapsed, expected_unit):
        """time_func prints milliseconds for sub-second runs and seconds otherwise."""
        fake_time = MagicMock(side_effect=[100.0, 100.0 + elapsed])
        with patch.object(fib, "time", fake_time):
            fib.time_func(lambda: None)
        captured = capsys.readouterr()
        assert expected_unit in captured.out


# ---------------------------------------------------------------------------
# fib_iterative_yield
# ---------------------------------------------------------------------------
class TestFibIterativeYield:
    """Tests for the generator-based iterative Fibonacci implementation."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, FIB_0_TO_10),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Generator yields the first n+1 Fibonacci numbers."""
        assert list(fib.fib_iterative_yield(n)) == expected

    def test_returns_iterator(self):
        """The function must return a lazy iterator, not a list."""
        result = fib.fib_iterative_yield(5)
        assert hasattr(result, "__next__")
        assert not isinstance(result, list)

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            list(fib.fib_iterative_yield(-1))

    def test_large_n_length(self):
        """The generator must produce exactly n+1 values."""
        assert len(list(fib.fib_iterative_yield(100))) == 101


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------
class TestFibIterative:
    """Tests for the list-based iterative Fibonacci implementation."""

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
    def test_positive_cases(self, n, expected):
        """Returns the first n+1 Fibonacci numbers as a list."""
        assert fib.fib_iterative(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_iterative(-1)

    def test_returns_list(self):
        """Return type must be a list of ints."""
        result = fib.fib_iterative(5)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_large_n(self):
        """Large n must still produce the correct length and last value."""
        result = fib.fib_iterative(200)
        assert len(result) == 201
        assert result[-1] == result[-2] + result[-3]


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------
class TestFibRecursive:
    """Tests for the naive recursive Fibonacci implementation."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, FIB_0_TO_10),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Returns the first n+1 Fibonacci numbers via recursion."""
        assert fib.fib_recursive(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_recursive(-1)

    def test_matches_iterative(self):
        """Recursive and iterative implementations must agree."""
        assert fib.fib_recursive(15) == fib.fib_iterative(15)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------
class TestFibRecursiveCached:
    """Tests for the cached recursive Fibonacci implementation."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, FIB_0_TO_10),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Returns the first n+1 Fibonacci numbers via cached recursion."""
        assert fib.fib_recursive_cached(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_recursive_cached(-1)

    def test_handles_larger_n(self):
        """Caching must make larger inputs feasible."""
        result = fib.fib_recursive_cached(100)
        assert len(result) == 101
        assert result == fib.fib_iterative(100)


# ---------------------------------------------------------------------------
# fib_memoization
# ---------------------------------------------------------------------------
class TestFibMemoization:
    """Tests for the memoization-based Fibonacci implementation."""

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
    def test_positive_cases(self, n, expected):
        """Returns the first n+1 Fibonacci numbers using memoization."""
        assert fib.fib_memoization(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_memoization(-1)

    def test_matches_iterative(self):
        """Memoized and iterative implementations must agree."""
        assert fib.fib_memoization(50) == fib.fib_iterative(50)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------
class TestFibBinet:
    """Tests for the Binet's-formula Fibonacci implementation."""

    @pytest.mark.parametrize(
        "n, expected",
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, FIB_0_TO_10),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Returns the first n+1 Fibonacci numbers via Binet's formula."""
        assert fib.fib_binet(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_binet(-1)

    @pytest.mark.parametrize("n", [1475, 1476, 2000])
    def test_too_large_raises(self, n):
        """n >= 1475 must raise ValueError due to float overflow."""
        with pytest.raises(ValueError, match="n is too large"):
            fib.fib_binet(n)

    def test_boundary_1474_allowed(self):
        """n == 1474 is the largest accepted value."""
        result = fib.fib_binet(1474)
        assert len(result) == 1475

    def test_matches_iterative_small(self):
        """Binet's formula must agree with iteration for small n."""
        assert fib.fib_binet(30) == fib.fib_iterative(30)


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------
class TestMatrixPowNp:
    """Tests for the numpy matrix exponentiation helper."""

    @pytest.fixture
    def fib_matrix(self):
        """The standard Fibonacci transition matrix."""
        return np.array([[1, 1], [1, 0]], dtype=int)

    def test_power_zero_identity(self, fib_matrix):
        """Raising to power 0 returns the identity matrix."""
        result = fib.matrix_pow_np(fib_matrix, 0)
        np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]], dtype=int))

    def test_power_one_unchanged(self, fib_matrix):
        """Raising to power 1 returns the original matrix."""
        result = fib.matrix_pow_np(fib_matrix, 1)
        np.testing.assert_array_equal(result, fib_matrix)

    @pytest.mark.parametrize(
        "power, expected",
        [
            (2, [[2, 1], [1, 1]]),
            (3, [[3, 2], [2, 1]]),
            (5, [[8, 5], [5, 3]]),
            (10, [[89, 55], [55, 34]]),
        ],
    )
    def test_positive_powers(self, fib_matrix, power, expected):
        """Matrix powers match the known Fibonacci transition matrices."""
        result = fib.matrix_pow_np(fib_matrix, power)
        np.testing.assert_array_equal(result, np.array(expected, dtype=int))

    def test_negative_power_raises(self, fib_matrix):
        """Negative power must raise ValueError."""
        with pytest.raises(ValueError, match="power is negative"):
            fib.matrix_pow_np(fib_matrix, -1)

    def test_identity_matrix_any_power(self):
        """The identity matrix raised to any power is itself."""
        identity = np.array([[1, 0], [0, 1]], dtype=int)
        result = fib.matrix_pow_np(identity, 7)
        np.testing.assert_array_equal(result, identity)

    def test_returns_ndarray(self, fib_matrix):
        """Return type must be a numpy ndarray."""
        assert isinstance(fib.matrix_pow_np(fib_matrix, 3), np.ndarray)


# ---------------------------------------------------------------------------
# fib_matrix_np
# ---------------------------------------------------------------------------
class TestFibMatrixNp:
    """Tests for the matrix-exponentiation Fibonacci implementation."""

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
    def test_positive_cases(self, n, expected):
        """Returns the n-th Fibonacci number."""
        assert fib.fib_matrix_np(n) == expected

    def test_negative_raises(self):
        """Negative n must raise ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib.fib_matrix_np(-1)

    def test_returns_int(self):
        """Return type must be a plain Python int."""
        result = fib.fib_matrix_np(10)
        assert isinstance(result, int)
        assert not isinstance(result, np.integer)

    def test_matches_iterative(self):
        """Matrix method must agree with the iterative implementation.

        NOTE: ``fib_matrix_np`` keeps its intermediate values in numpy's
        fixed-width integer dtype, so it silently overflows once the Fibonacci
        number exceeds ``2**63 - 1`` (n >= 93).  The comparison is therefore
        limited to inputs whose result still fits into that dtype.
        """
        n = 90
        assert fib.fib_matrix_np(n) == fib.fib_iterative(n)[-1]

    def test_large_n(self):
        """Largest n whose value still fits in numpy's integer dtype works."""
        n = 92
        expected = fib.fib_iterative(n)[-1]
        # Guard: keep the test meaningful only while the value is in range.
        assert expected <= np.iinfo(np.int64).max
        assert fib.fib_matrix_np(n) == expected
