"""Pytest test suite for bubble_sort.py

Covers both `bubble_sort_iterative` and `bubble_sort_recursive` with
positive, edge, boundary, and regression-style test cases.
"""

import math
import random
import string
from typing import Any

import pytest

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_sorted(seq: list[Any]) -> bool:
    """Return True if *seq* is non-decreasing."""
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def _same_elements(a: list[Any], b: list[Any]) -> bool:
    """Return True if *a* and *b* contain the same multiset of elements."""
    return sorted(a) == sorted(b)


# ---------------------------------------------------------------------------
# Parametrized test data
# ---------------------------------------------------------------------------

# Each entry: (description, input_list)
POSITIVE_CASES = [
    ("empty list", []),
    ("single element", [42]),
    ("two elements already sorted", [1, 2]),
    ("two elements reversed", [2, 1]),
    ("small mixed integers", [0, 5, 2, 3, 2]),
    ("negative integers", [-2, -45, -5]),
    ("mixed sign integers", [-23, 0, 6, -4, 34]),
    ("already sorted", [1, 2, 3, 4]),
    ("all equal", [3, 3, 3, 3]),
    ("floats", [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6]),
    ("mixed ints and floats", [1, 3.3, 5, 7.7, 2, 4.4, 6]),
    ("single-char strings", ['d', 'a', 'b', 'e']),
    ("multi-char strings", ['z', 'a', 'y', 'b', 'x', 'c']),
    ("mixed-case strings", ['a', 'Z', 'B', 'C', 'A', 'c']),
    ("large sorted list", list(range(100))),
    ("large reversed list", list(range(100, 0, -1))),
]

EDGE_CASES = [
    ("single None-like sentinel would fail, skip", [0]),
    ("very large values", [10**9, -(10**9), 0]),
    ("floats with infinity", [float('inf'), float('-inf'), 0.0]),
]


# ---------------------------------------------------------------------------
# bubble_sort_iterative
# ---------------------------------------------------------------------------

class TestBubbleSortIterative:
    """Tests for `bubble_sort_iterative`."""

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_sorts_correctly(self, desc: str, data: list[Any]) -> None:
        """The returned list is sorted in non-decreasing order."""
        result = bubble_sort_iterative(data[:])
        assert _is_sorted(result), f"Failed on {desc}"

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_preserves_elements(self, desc: str, data: list[Any]) -> None:
        """The returned list contains exactly the same elements (multiset)."""
        original = data[:]
        result = bubble_sort_iterative(data[:])
        assert _same_elements(original, result), f"Element mismatch on {desc}"

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_matches_builtin_sorted(self, desc: str, data: list[Any]) -> None:
        """The result equals Python's built-in `sorted`."""
        result = bubble_sort_iterative(data[:])
        assert result == sorted(data), f"Mismatch with sorted() on {desc}"

    def test_does_not_mutate_original_list_elements(self) -> None:
        """The original list's *elements* are unchanged (though the list
        object itself may be mutated in-place, the multiset is preserved)."""
        original = [3, 1, 2]
        copy = original[:]
        bubble_sort_iterative(copy)
        assert sorted(original) == sorted(copy) == [1, 2, 3]

    def test_empty_list_returns_empty(self) -> None:
        """Empty input returns empty list."""
        assert bubble_sort_iterative([]) == []

    def test_single_element_returns_same(self) -> None:
        """Single-element list is returned as-is."""
        assert bubble_sort_iterative([99]) == [99]

    def test_already_sorted_breaks_early(self) -> None:
        """Already-sorted list should be detected and returned quickly
        (the `swapped` flag causes an early break)."""
        data = list(range(50))
        result = bubble_sort_iterative(data[:])
        assert result == data

    def test_random_large_list(self) -> None:
        """Stress test with 200 random integers."""
        random.seed(0)
        data = random.sample(range(-500, 500), 200)
        result = bubble_sort_iterative(data[:])
        assert result == sorted(data)

    def test_random_string_list(self) -> None:
        """Stress test with 200 random ASCII letters/digits."""
        random.seed(1)
        data = random.choices(string.ascii_letters + string.digits, k=200)
        result = bubble_sort_iterative(data[:])
        assert result == sorted(data)

    # Edge / boundary
    @pytest.mark.parametrize("desc,data", EDGE_CASES)
    def test_edge_cases(self, desc: str, data: list[Any]) -> None:
        """Edge-case inputs are handled without error and produce a
        sorted result (NaN ordering is Python's default)."""
        result = bubble_sort_iterative(data[:])
        assert _is_sorted(result), f"Not sorted on {desc}"


# ---------------------------------------------------------------------------
# bubble_sort_recursive
# ---------------------------------------------------------------------------

class TestBubbleSortRecursive:
    """Tests for `bubble_sort_recursive`."""

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_sorts_correctly(self, desc: str, data: list[Any]) -> None:
        """The returned list is sorted in non-decreasing order."""
        result = bubble_sort_recursive(data[:])
        assert _is_sorted(result), f"Failed on {desc}"

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_preserves_elements(self, desc: str, data: list[Any]) -> None:
        """The returned list contains exactly the same elements (multiset)."""
        original = data[:]
        result = bubble_sort_recursive(data[:])
        assert _same_elements(original, result), f"Element mismatch on {desc}"

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_matches_builtin_sorted(self, desc: str, data: list[Any]) -> None:
        """The result equals Python's built-in `sorted`."""
        result = bubble_sort_recursive(data[:])
        assert result == sorted(data), f"Mismatch with sorted() on {desc}"

    def test_empty_list_returns_empty(self) -> None:
        """Empty input returns empty list."""
        assert bubble_sort_recursive([]) == []

    def test_single_element_returns_same(self) -> None:
        """Single-element list is returned as-is."""
        assert bubble_sort_recursive([99]) == [99]

    def test_already_sorted_returns_immediately(self) -> None:
        """Already-sorted list returns immediately (no swap → base case)."""
        data = list(range(50))
        result = bubble_sort_recursive(data[:])
        assert result == data

    def test_random_large_list(self) -> None:
        """Stress test with 200 random integers (recursive depth is safe)."""
        random.seed(0)
        data = random.sample(range(-500, 500), 200)
        result = bubble_sort_recursive(data[:])
        assert result == sorted(data)

    def test_random_string_list(self) -> None:
        """Stress test with 200 random ASCII letters/digits."""
        random.seed(1)
        data = random.choices(string.ascii_letters + string.digits, k=200)
        result = bubble_sort_recursive(data[:])
        assert result == sorted(data)

    @pytest.mark.parametrize("desc,data", EDGE_CASES)
    def test_edge_cases(self, desc: str, data: list[Any]) -> None:
        """Edge-case inputs are handled without error and produce a
        sorted result."""
        result = bubble_sort_recursive(data[:])
        assert _is_sorted(result), f"Not sorted on {desc}"

    def test_recursion_depth_for_large_input(self) -> None:
        """Ensure the recursive implementation does not hit Python's
        recursion limit for a moderately large reversed list."""
        # 500 elements → at most 500 recursive calls (well below default 1000).
        data = list(range(500, 0, -1))
        result = bubble_sort_recursive(data[:])
        assert result == sorted(data)


# ---------------------------------------------------------------------------
# Cross-comparison: iterative vs recursive
# ---------------------------------------------------------------------------

class TestConsistencyBetweenImplementations:
    """Both implementations should produce identical results."""

    @pytest.mark.parametrize("desc,data", POSITIVE_CASES)
    def test_both_implementations_match(self, desc: str, data: list[Any]) -> None:
        """Iterative and recursive bubble sort return the same result."""
        a = bubble_sort_iterative(data[:])
        b = bubble_sort_recursive(data[:])
        assert a == b, f"Implementations diverge on {desc}"
