import numpy as np
import pytest

from maths.fibonacci import (
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
# fib_iterative_yield
# ---------------------------------------------------------------------------
class TestFibIterativeYield:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert list(fib_iterative_yield(n)) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            list(fib_iterative_yield(-1))

    def test_returns_iterator(self):
        """Verify the function returns an iterator (lazy generator)."""
        result = fib_iterative_yield(3)
        assert iter(result) is result


# ---------------------------------------------------------------------------
# fib_iterative
# ---------------------------------------------------------------------------
class TestFibIterative:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert fib_iterative(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_iterative(-1)

    def test_large_n_consistency(self):
        """Verify a larger n produces a list of the correct length."""
        result = fib_iterative(20)
        assert len(result) == 21
        assert result[-1] == 6765


# ---------------------------------------------------------------------------
# fib_recursive
# ---------------------------------------------------------------------------
class TestFibRecursive:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert fib_recursive(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_recursive(-1)


# ---------------------------------------------------------------------------
# fib_recursive_cached
# ---------------------------------------------------------------------------
class TestFibRecursiveCached:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert fib_recursive_cached(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_recursive_cached(-1)


# ---------------------------------------------------------------------------
# fib_memoization
# ---------------------------------------------------------------------------
class TestFibMemoization:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert fib_memoization(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_memoization(-1)


# ---------------------------------------------------------------------------
# fib_binet
# ---------------------------------------------------------------------------
class TestFibBinet:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, [0]),
            (1, [0, 1]),
            (5, [0, 1, 1, 2, 3, 5]),
            (10, [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct Fibonacci sequences for valid non-negative n."""
        assert fib_binet(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_binet(-1)

    def test_too_large_raises_value_error(self):
        """Verify n >= 1475 raises ValueError due to float overflow."""
        with pytest.raises(ValueError, match="n is too large"):
            fib_binet(1475)

    def test_boundary_1474_does_not_raise(self):
        """Verify n = 1474 (just below the limit) does not raise."""
        result = fib_binet(1474)
        assert len(result) == 1475


# ---------------------------------------------------------------------------
# matrix_pow_np
# ---------------------------------------------------------------------------
class TestMatrixPowNp:
    def test_power_zero_returns_identity(self):
        """Verify raising to power 0 returns the identity matrix."""
        m = np.array([[1, 1], [1, 0]], dtype=int)
        expected = np.array([[1, 0], [0, 1]], dtype=int)
        np.testing.assert_array_equal(matrix_pow_np(m, 0), expected)

    def test_power_one_returns_same_matrix(self):
        """Verify raising to power 1 returns the original matrix."""
        m = np.array([[1, 1], [1, 0]], dtype=int)
        np.testing.assert_array_equal(matrix_pow_np(m, 1), m)

    def test_power_five(self):
        """Verify raising to power 5 produces the expected matrix."""
        m = np.array([[1, 1], [1, 0]], dtype=int)
        expected = np.array([[8, 5], [5, 3]], dtype=int)
        np.testing.assert_array_equal(matrix_pow_np(m, 5), expected)

    def test_negative_power_raises_value_error(self):
        """Verify a negative power raises ValueError."""
        m = np.array([[1, 1], [1, 0]], dtype=int)
        with pytest.raises(ValueError, match="power is negative"):
            matrix_pow_np(m, -1)


# ---------------------------------------------------------------------------
# fib_matrix_np
# ---------------------------------------------------------------------------
class TestFibMatrixNp:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, 0),
            (1, 1),
            (5, 5),
            (10, 55),
        ],
    )
    def test_positive_cases(self, n, expected):
        """Verify correct n-th Fibonacci number for valid non-negative n."""
        assert fib_matrix_np(n) == expected

    def test_negative_raises_value_error(self):
        """Verify a negative n raises ValueError."""
        with pytest.raises(ValueError, match="n is negative"):
            fib_matrix_np(-1)

    def test_large_n_matches_iterative(self):
        """Verify matrix result matches iterative result for a larger n."""
        n = 50
        assert fib_matrix_np(n) == fib_iterative(n)[-1]


# ---------------------------------------------------------------------------
# time_func
# ---------------------------------------------------------------------------
class TestTimeFunc:
    def test_returns_function_output(self):
        """Verify time_func returns the wrapped function's output."""
        result = time_func(lambda: 42)
        assert result == 42

    def test_passes_args_and_kwargs(self):
        """Verify time_func forwards positional and keyword arguments."""
        def add(a, b=0):
            return a + b

        assert time_func(add, 2, b=3) == 5

    def test_prints_runtime(self, capsys):
        """Verify time_func prints a runtime message."""
        time_func(lambda: None)
        captured = capsys.readouterr()
        assert "runtime:" in captured.out