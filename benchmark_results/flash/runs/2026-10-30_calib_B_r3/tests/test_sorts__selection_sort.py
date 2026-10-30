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
    assert data == [1, 3, 4]


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
        ([], []),                 # empty list
        ([42], [42]),             # single element
        ([1, 2], [1, 2]),         # already sorted pair
        ([2, 1], [1, 2]),         # reversed pair
        ([5, 5, 5], [5, 5, 5]),   # all duplicates
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # already sorted
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),  # reverse sorted
        ([0, 0, -1, 0], [-1, 0, 0, 0]),      # zeros and a negative
    ],
)
def test_selection_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, duplicates, sorted, reversed) sort correctly."""
    assert selection_sort(collection) == expected


def test_selection_sort_empty_list_returns_empty_list():
    """An empty list is returned unchanged."""
    data = []
    assert selection_sort(data) == []
    assert data == []


def test_selection_sort_single_element_unchanged():
    """A one-element list is returned unchanged."""
    assert selection_sort([7]) == [7]


def test_selection_sort_does_not_add_or_remove_elements():
    """Sorting preserves the multiset of elements (length and contents)."""
    data = [9, 3, 9, 1, 3, 1]
    result = selection_sort(data)
    assert len(result) == len(data)
    assert sorted(result) == sorted(data)


def test_selection_sort_large_random_list():
    """A larger pseudo-random list is sorted identically to Python's sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(data)
    assert selection_sort(data) == expected


def test_selection_sort_does_not_mutate_input_when_copied():
    """Sorting a copy leaves the original list untouched."""
    original = [3, 1, 2]
    copy = list(original)
    selection_sort(copy)
    assert original == [3, 1, 2]
    assert copy == [1, 2, 3]


def test_selection_sort_duplicates_are_stable_in_value():
    """Duplicate values all remain present after sorting."""
    data = [2, 1, 2, 1, 2]
    assert selection_sort(data) == [1, 1, 2, 2, 2]