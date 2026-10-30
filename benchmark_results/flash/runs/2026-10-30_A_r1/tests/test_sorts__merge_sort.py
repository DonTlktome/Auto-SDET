"""Tests for the merge_sort module."""

import random

import pytest

from sorts.merge_sort import merge_sort


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------

def test_merge_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_merge_sort_returns_new_list_and_does_not_mutate_input():
    """The input list must not be modified by the sort."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    result = merge_sort(original)
    assert result == [1, 2, 3, 4]
    assert original == snapshot


def test_merge_sort_handles_duplicates():
    """Duplicate values are preserved (stable multiset of elements)."""
    assert merge_sort([3, 1, 3, 1, 3]) == [1, 1, 3, 3, 3]


def test_merge_sort_handles_floats():
    """Floating point values are sorted correctly."""
    assert merge_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_merge_sort_handles_strings():
    """Strings are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_matches_builtin_sorted_on_random_data():
    """Result matches Python's built-in sorted() for random input."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert merge_sort(data) == sorted(data)


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                       # empty list
        ([42], [42]),                   # single element
        ([1, 2], [1, 2]),               # already sorted pair
        ([2, 1], [1, 2]),               # reversed pair
        ([-2, -5, -45], [-45, -5, -2]), # all negative
        ([0, 0, 0], [0, 0, 0]),         # all equal
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),  # fully reversed
        (list(range(10)), list(range(10))),  # already sorted
    ],
)
def test_merge_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, reversed, negatives, equal) sort correctly."""
    assert merge_sort(collection) == expected


def test_merge_sort_empty_list_returns_empty_list():
    """An empty input yields an empty list."""
    assert merge_sort([]) == []


def test_merge_sort_single_element_returns_same_content():
    """A one-element list is returned unchanged in content."""
    assert merge_sort([7]) == [7]


def test_merge_sort_large_input():
    """A large input is sorted correctly (exercises deep recursion)."""
    data = list(range(1000, 0, -1))
    assert merge_sort(data) == list(range(1, 1001))


def test_merge_sort_odd_length_input():
    """Odd-length inputs split unevenly and still sort correctly."""
    assert merge_sort([9, 1, 8, 2, 7]) == [1, 2, 7, 8, 9]


def test_merge_sort_does_not_mutate_nested_slices():
    """The original list object identity is preserved for the input container."""
    data = [3, 1, 2]
    result = merge_sort(data)
    assert result is not data
    assert data == [3, 1, 2]