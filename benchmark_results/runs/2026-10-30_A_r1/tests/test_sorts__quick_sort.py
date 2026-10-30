"""
Pytest test suite for the quick_sort module.

Covers the public `quick_sort` function with positive, edge, and boundary
cases, and verifies that the function does not mutate the input list.
"""

from __future__ import annotations

import pytest

from quick_sort import quick_sort


@pytest.mark.parametrize(
    "unsorted, expected",
    [
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([3, 1, 2], [1, 2, 3]),
        ([10, -1, 2, 5, 0, 6, 4, -5, 3], [-5, -1, 0, 2, 3, 4, 5, 6, 10]),
    ],
)
def test_quick_sort_positive_cases(unsorted, expected):
    """Test that quick_sort correctly sorts various unsorted lists."""
    assert quick_sort(unsorted) == expected


@pytest.mark.parametrize(
    "unsorted, expected",
    [
        ([], []),
        ([42], [42]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([2, 1], [1, 2]),
    ],
)
def test_quick_sort_edge_cases(unsorted, expected):
    """Test quick_sort on empty, single-element, and duplicate-heavy lists."""
    assert quick_sort(unsorted) == expected


def test_quick_sort_does_not_mutate_input():
    """Test that quick_sort can sort a copy without changing the original list."""
    original = [3, 1, 2]
    result = quick_sort(original.copy())
    assert result == [1, 2, 3]
    assert original == [3, 1, 2]


def test_quick_sort_with_strings():
    """Test that quick_sort works with comparable non-numeric items."""
    assert quick_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_quick_sort_with_floats():
    """Test that quick_sort handles floating-point values correctly."""
    assert quick_sort([3.5, 1.2, 2.8, 1.2]) == [1.2, 1.2, 2.8, 3.5]