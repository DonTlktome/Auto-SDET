"""
Test file for calculator.py
"""
import pytest
from calculator import add, subtract, multiply, divide, sqrt, Calculator


def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(2.5, 3.5) == 6.0


def test_subtract():
    """Test subtraction function."""
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2
    assert subtract(0, 0) == 0
    assert subtract(2.5, 1.5) == 1.0


def test_multiply():
    """Test multiplication function."""
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0
    assert multiply(2.5, 4) == 10.0


def test_divide():
    """Test division function."""
    assert divide(6, 3) == 2
    assert divide(5, 2) == 2.5
    assert divide(0, 5) == 0
    
    # Test division by zero raises ValueError
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(5, 0)


def test_sqrt():
    """Test square root function."""
    assert sqrt(4) == 2
    assert sqrt(0) == 0
    assert sqrt(2.25) == 1.5
    
    # Test negative input raises ValueError
    with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
        sqrt(-1)


class TestCalculator:
    """Test Calculator class."""
    
    def setup_method(self):
        """Setup method to create a fresh calculator for each test."""
        self.calc = Calculator()
    
    def test_compute_add(self):
        """Test compute method with addition."""
        result = self.calc.compute("add", 2, 3)
        assert result == 5
        assert len(self.calc.history) == 1
        assert "add(2, 3) = 5" in self.calc.history[0]
    
    def test_compute_subtract(self):
        """Test compute method with subtraction."""
        result = self.calc.compute("subtract", 5, 3)
        assert result == 2
        assert "subtract(5, 3) = 2" in self.calc.history[0]
    
    def test_compute_multiply(self):
        """Test compute method with multiplication."""
        result = self.calc.compute("multiply", 2, 3)
        assert result == 6
        assert "multiply(2, 3) = 6" in self.calc.history[0]
    
    def test_compute_divide(self):
        """Test compute method with division."""
        result = self.calc.compute("divide", 6, 3)
        assert result == 2
        assert "divide(6, 3) = 2" in self.calc.history[0]
        
        # Test division by zero
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.compute("divide", 5, 0)
    
    def test_compute_sqrt(self):
        """Test compute method with square root."""
        result = self.calc.compute("sqrt", 4)
        assert result == 2
        assert "sqrt(4, 0) = 2" in self.calc.history[0]
        
        # Test negative input
        with pytest.raises(ValueError, match="Cannot compute square root of negative number"):
            self.calc.compute("sqrt", -1)
    
    def test_compute_invalid_operation(self):
        """Test compute method with invalid operation."""
        with pytest.raises(ValueError, match="Unknown operation"):
            self.calc.compute("invalid", 1, 2)
    
    def test_get_history(self):
        """Test get_history method."""
        self.calc.compute("add", 1, 2)
        self.calc.compute("subtract", 5, 3)
        
        history = self.calc.get_history()
        assert len(history) == 2
        assert "add(1, 2) = 3" in history[0]
        assert "subtract(5, 3) = 2" in history[1]
    
    def test_clear_history(self):
        """Test clear_history method."""
        self.calc.compute("add", 1, 2)
        self.calc.compute("subtract", 5, 3)
        
        assert len(self.calc.history) == 2
        self.calc.clear_history()
        assert len(self.calc.history) == 0
        assert len(self.calc.get_history()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])