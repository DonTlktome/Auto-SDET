import pytest
from calculator import add, subtract, multiply, divide, sqrt, Calculator


# ----------------------------------------------------------------
# Tests for add
# ----------------------------------------------------------------
class TestAdd:
    """Tests for add() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (2, 3, 5),
            (-2, 3, 1),
            (0, 0, 0),
            (1.5, 2.5, 4.0),
            (-1.1, -2.2, -3.3),
            (1e10, 1e10, 2e10),
        ],
    )
    def test_add_positive_and_edge_cases(self, a, b, expected):
        """Test add() with positive, negative, zero, float and large numbers."""
        result = add(a, b)
        assert result == pytest.approx(expected)

    def test_add_with_zero(self):
        """Edge case: adding zero does not change value."""
        assert add(5, 0) == 5
        assert add(0, 5) == 5


# ----------------------------------------------------------------
# Tests for subtract
# ----------------------------------------------------------------
class TestSubtract:
    """Tests for subtract() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (10, 5, 5),
            (5, 10, -5),
            (-3, -3, 0),
            (0, 0, 0),
            (1.5, 0.5, 1.0),
        ],
    )
    def test_subtract_positive_and_edge(self, a, b, expected):
        """Test subtract() with various inputs including zero and negatives."""
        result = subtract(a, b)
        assert result == pytest.approx(expected)

    def test_subtract_symmetric(self):
        """Subtracting a number from itself yields zero."""
        assert subtract(7, 7) == 0


# ----------------------------------------------------------------
# Tests for multiply
# ----------------------------------------------------------------
class TestMultiply:
    """Tests for multiply() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (3, 4, 12),
            (-3, 4, -12),
            (0, 10, 0),
            (0, 0, 0),
            (2.5, 2, 5.0),
        ],
    )
    def test_multiply_positive_and_edge(self, a, b, expected):
        """Test multiply() with positive, negative, zero, and floats."""
        result = multiply(a, b)
        assert result == pytest.approx(expected)

    def test_multiply_by_one(self):
        """Edge case: multiplication by 1 returns the same number."""
        assert multiply(42, 1) == 42


# ----------------------------------------------------------------
# Tests for divide
# ----------------------------------------------------------------
class TestDivide:
    """Tests for divide() function."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (6, 2, 3),
            (6, -2, -3),
            (0, 5, 0),
            (2.5, 0.5, 5.0),
        ],
    )
    def test_divide_happy_path(self, a, b, expected):
        """Test divide() with valid non-zero divisors."""
        result = divide(a, b)
        assert result == pytest.approx(expected)

    def test_divide_by_zero_raises(self):
        """Edge case: division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)


# ----------------------------------------------------------------
# Tests for sqrt
# ----------------------------------------------------------------
class TestSqrt:
    """Tests for sqrt() function."""

    @pytest.mark.parametrize(
        "x, expected",
        [
            (4, 2),
            (0, 0),
            (2.25, 1.5),
            (0.25, 0.5),
        ],
    )
    def test_sqrt_positive(self, x, expected):
        """Test sqrt() with non-negative numbers."""
        result = sqrt(x)
        assert result == pytest.approx(expected)

    def test_sqrt_of_one(self):
        """Edge case: sqrt(1) = 1."""
        assert sqrt(1) == 1

    def test_sqrt_negative_raises(self):
        """Edge case: sqrt of negative number raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
            sqrt(-9)


# ----------------------------------------------------------------
# Tests for Calculator class
# ----------------------------------------------------------------
class TestCalculator:
    """Tests for the Calculator class."""

    @pytest.fixture
    def calc(self):
        """Provide a fresh Calculator instance for each test."""
        return Calculator()

    # ------ compute ------
    @pytest.mark.parametrize(
        "operation, a, b, expected_result, expected_history_entry",
        [
            ("add", 3, 4, 7, "add(3, 4) = 7"),
            ("subtract", 10, 3, 7, "subtract(10, 3) = 7"),
            ("multiply", 2, 5, 10, "multiply(2, 5) = 10"),
            ("divide", 12, 4, 3, "divide(12, 4) = 3.0"),
            ("sqrt", 9, 0, 3, "sqrt(9, 0) = 3.0"),  # b ignored, but captured as 0 in history
        ],
    )
    def test_compute_valid_operations(
        self, calc, operation, a, b, expected_result, expected_history_entry
    ):
        """Positive test: compute() returns correct result and records history entry."""
        result = calc.compute(operation, a, b)
        assert result == pytest.approx(expected_result)
        history = calc.get_history()
        assert len(history) == 1
        assert history[0] == expected_history_entry

    def test_compute_unknown_operation_raises(self, calc):
        """Edge case: unknown operation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation: invalid"):
            calc.compute("invalid", 1, 2)

    def test_compute_divide_by_zero_propagates_and_no_history(self, calc):
        """
        Edge case: compute divide by zero raises ValueError and
        does NOT record history.
        """
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.compute("divide", 5, 0)
        assert calc.get_history() == []

    def test_compute_sqrt_negative_propagates_and_no_history(self, calc):
        """
        Edge case: compute sqrt of negative raises ValueError and
        does NOT record history.
        """
        with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
            calc.compute("sqrt", -4)
        assert calc.get_history() == []

    def test_compute_multiple_operations_history(self, calc):
        """Positive test: performing multiple computations accumulates history."""
        calc.compute("add", 2, 3)
        calc.compute("multiply", 4, 5)
        history = calc.get_history()
        assert len(history) == 2
        assert history == ["add(2, 3) = 5", "multiply(4, 5) = 20"]

    # ------ get_history ------
    def test_get_history_empty(self, calc):
        """Edge case: get_history on a fresh calculator returns empty list."""
        hist = calc.get_history()
        assert hist == []

    def test_get_history_returns_copy(self, calc):
        """Positive test: get_history returns a copy, not the internal list."""
        calc.compute("add", 1, 1)
        hist = calc.get_history()
        hist.append("malicious")
        assert calc.get_history() == ["add(1, 1) = 2"]

    # ------ clear_history ------
    def test_clear_history(self, calc):
        """Positive test: clear_history removes all entries."""
        calc.compute("add", 1, 2)
        calc.clear_history()
        assert calc.get_history() == []

    def test_clear_history_on_empty(self, calc):
        """Edge case: clear_history on already empty history does nothing."""
        calc.clear_history()  # no exception
        assert calc.get_history() == []