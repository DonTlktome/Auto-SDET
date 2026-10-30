"""
Pytest test suite for quick_sort.py

Covers all public functions with positive, edge-case, and property-based tests.
Uses mocking to isolate the random pivot selection for deterministic testing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quick_sort import quick_sort


class TestQuickSort:
    """Tests for the quick_sort function."""

    # ------------------------------------------------------------------
    # Positive / happy-path tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("input_list", "expected"),
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([3, 1, 2], [1, 2, 3]),
            ([42], [42]),
            ([], []),
            ([-2, 5, 0, -45], [-45, -2, 0, 5]),
            ([3.5, 1.0, 2.5], [1.0, 2.5, 3.5]),
        ],
    )
    def test_sort_various_lists(self, input_list: list, expected: list) -> None:
        """quick_sort should return the list sorted in ascending order."""
        # Copy the input because quick_sort mutates it (pop) and returns a new list
        result = quick_sort(input_list[:])
        assert result == expected

    def test_sort_large_list(self) -> None:
        """quick_sort should correctly sort a large list of integers."""
        large = list(range(1000, 0, -1))
        expected = list(range(1, 1001))
        result = quick_sort(large[:])
        assert result == expected

    # ------------------------------------------------------------------
    # Edge-case / boundary tests
    # ------------------------------------------------------------------

    def test_empty_list(self) -> None:
        """quick_sort on an empty list should return an empty list."""
        assert quick_sort([]) == []

    def test_single_element(self) -> None:
        """quick_sort on a single-element list should return the same list."""
        assert quick_sort([99]) == [99]

    def test_two_elements_sorted(self) -> None:
        """quick_sort on an already-sorted two-element list."""
        assert quick_sort([1, 2]) == [1, 2]

    def test_two_elements_reversed(self) -> None:
        """quick_sort on a reversed two-element list."""
        assert quick_sort([2, 1]) == [1, 2]

    def test_all_equal_elements(self) -> None:
        """quick_sort on a list where all elements are equal."""
        assert quick_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_negative_numbers(self) -> None:
        """quick_sort should handle negative numbers correctly."""
        assert quick_sort([-5, -1, -3, -2, -4]) == [-5, -4, -3, -2, -1]

    def test_mixed_positive_negative(self) -> None:
        """quick_sort should handle a mix of positive and negative numbers."""
        result = quick_sort([3, -1, 0, 2, -2])
        assert result == [-2, -1, 0, 2, 3]

    def test_duplicates(self) -> None:
        """quick_sort should preserve duplicate values correctly."""
        result = quick_sort([4, 2, 4, 1, 2, 3, 1])
        assert result == [1, 1, 2, 2, 3, 4, 4]

    def test_strings(self) -> None:
        """quick_sort should work with strings (comparable items)."""
        result = quick_sort(["banana", "apple", "cherry", "date"])
        assert result == ["apple", "banana", "cherry", "date"]

    def test_floats(self) -> None:
        """quick_sort should work with floating-point numbers."""
        result = quick_sort([3.14, 1.41, 2.71, 0.0, -1.0])
        assert result == [-1.0, 0.0, 1.41, 2.71, 3.14]

    # ------------------------------------------------------------------
    # Tests with mocked pivot to ensure deterministic partitioning paths
    # ------------------------------------------------------------------

    def test_pivot_first_element(self) -> None:
        """quick_sort with pivot forced to index 0 (first element)."""
        with patch("quick_sort.randrange", return_value=0):
            result = quick_sort([3, 1, 4, 1, 5, 9, 2, 6])
        assert result == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_pivot_last_element(self) -> None:
        """quick_sort with pivot forced to the last index."""
        input_list = [9, 7, 5, 3, 1]
        with patch("quick_sort.randrange", side_effect=lambda n: n - 1):
            result = quick_sort(input_list[:])
        assert result == [1, 3, 5, 7, 9]

    def test_pivot_middle_element(self) -> None:
        """quick_sort with pivot forced to the middle index."""
        input_list = [5, 1, 3, 2, 4]
        with patch("quick_sort.randrange", side_effect=lambda n: n // 2):  # pivot = 3
            result = quick_sort(input_list[:])
        assert result == [1, 2, 3, 4, 5]

    # ------------------------------------------------------------------
    # Property-based / idempotency tests
    # ------------------------------------------------------------------

    def test_idempotent(self) -> None:
        """Sorting an already-sorted list should return the same order."""
        sorted_list = [1, 2, 3, 4, 5]
        result = quick_sort(sorted_list[:])
        assert result == sorted_list

    def test_original_unmodified_after_copy(self) -> None:
        """The caller's list should not be mutated if a copy is passed."""
        original = [3, 1, 2]
        copy = original[:]
        quick_sort(copy)
        assert original == [3, 1, 2]

    def test_length_preserved(self) -> None:
        """quick_sort should return a list of the same length."""
        input_list = [9, 4, 7, 1, 3, 8, 2, 6, 5]
        result = quick_sort(input_list[:])
        assert len(result) == len(input_list)

    def test_elements_preserved(self) -> None:
        """quick_sort should preserve all original elements (multiset equality)."""
        input_list = [5, 3, 5, 1, 3, 2]
        result = quick_sort(input_list[:])
        assert sorted(input_list) == result
        assert set(input_list) == set(result)
