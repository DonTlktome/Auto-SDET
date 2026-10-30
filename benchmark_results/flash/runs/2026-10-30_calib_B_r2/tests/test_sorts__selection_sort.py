"""Tests for the selection_sort module."""

import random

import pytest

from sorts.selection_sort import selection_sort


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------

def test_selection_sort_sorts_unsorted_list():
    """A typical unsorted list of positive integers is sorted ascending."""
    assert selection_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_selection_sort_returns_same_object():
    """The function sorts in place and returns the very same list object."""
    data = [4, 1, 3]
    result = selection_sort(data)
    assert result is data


def test_selection_sort_negative_numbers():
    """Negative integers are ordered correctly (docstring example)."""
    assert selection_sort([-2, -5, -45]) == [-45, -5, -2]


def test_selection_sort_mixed_signs():
    """A mix of negative, zero and positive values is sorted correctly."""
    assert selection_sort([3, -1, 0, -7, 8, 2]) == [-7, -1, 0, 2, 3, 8]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                      # empty list
        ([42], [42]),                  # single element
        ([1, 2], [1, 2]),              # already sorted, two elements
        ([2, 1], [1, 2]),              # reversed, two elements
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),   # already sorted
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),   # reverse sorted
        ([7, 7, 7, 7], [7, 7, 7, 7]),         # all duplicates
        ([0, 0, -1, 0], [-1, 0, 0, 0]),       # duplicates with negative
        ([2, 1, 2, 1, 2], [1, 1, 2, 2, 2]),   # interleaved duplicates
    ],
)
def test_selection_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, sorted, reversed, duplicates) sort correctly."""
    assert selection_sort(collection) == expected


def test_selection_sort_empty_list_returns_empty_list():
    """An empty list is returned unchanged."""
    data = []
    assert selection_sort(data) == []
    assert data == []


def test_selection_sort_single_element_unchanged():
    """A one-element list is returned as-is."""
    assert selection_sort([99]) == [99]


def test_selection_sort_does_not_change_length():
    """Sorting never adds or removes elements."""
    data = [5, 3, 9, 1, 3]
    original_length = len(data)
    selection_sort(data)
    assert len(data) == original_length


def test_selection_sort_preserves_multiset_of_values():
    """Sorting preserves the exact multiset of values (no data loss)."""
    data = [4, -2, 4, 0, 7, -2, 1]
    expected_sorted = sorted(data)
    assert selection_sort(data) == expected_sorted


def test_selection_sort_large_random_list_matches_builtin_sorted():
    """A larger pseudo-random list (seeded) matches Python's built-in sorted()."""
    rng = random.Random(12345)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    expected = sorted(data)
    assert selection_sort(data) == expected


def test_selection_sort_extreme_integer_values():
    """Very large and very small integers are handled without overflow issues."""
    data = [10**18, -(10**18), 0, 10**18 - 1]
    assert selection_sort(data) == [-(10**18), 0, 10**18 - 1, 10**18]


def test_selection_sort_does_not_mutate_input_when_already_sorted():
    """An already sorted list keeps its original ordering."""
    data = [1, 2, 3, 4, 5]
    selection_sort(data)
    assert data == [1, 2, 3, 4, 5]