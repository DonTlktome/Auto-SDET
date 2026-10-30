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


def test_merge_sort_returns_list_type():
    """The return value is a list instance."""
    result = merge_sort([3, 1, 2])
    assert isinstance(result, list)


def test_merge_sort_does_not_mutate_input():
    """The original input collection must not be modified."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    merge_sort(original)
    assert original == snapshot


def test_merge_sort_matches_builtin_sorted_on_random_data():
    """Result matches Python's built-in sorted() for random data."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert merge_sort(data) == sorted(data)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                       # empty list
        ([42], [42]),                   # single element
        ([1, 2], [1, 2]),               # already sorted pair
        ([2, 1], [1, 2]),               # reversed pair
        ([1, 1, 1, 1], [1, 1, 1, 1]),   # all duplicates
        ([-2, -5, -45], [-45, -5, -2]), # all negative
        ([0, -1, 1, -2, 2], [-2, -1, 0, 1, 2]),  # mixed signs
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),      # reverse sorted
    ],
)
def test_merge_sort_edge_cases(collection, expected):
    """Boundary and degenerate inputs are sorted correctly."""
    assert merge_sort(collection) == expected


def test_merge_sort_empty_list_returns_empty_list():
    """An empty input yields an empty list."""
    assert merge_sort([]) == []


def test_merge_sort_single_element_returns_same_element():
    """A one-element list is returned unchanged."""
    assert merge_sort([7]) == [7]


def test_merge_sort_odd_length_list():
    """Odd-length lists are split correctly around the midpoint."""
    assert merge_sort([9, 1, 8, 2, 7]) == [1, 2, 7, 8, 9]


def test_merge_sort_even_length_list():
    """Even-length lists are split correctly around the midpoint."""
    assert merge_sort([4, 3, 2, 1]) == [1, 2, 3, 4]


def test_merge_sort_large_list():
    """A large list is sorted correctly (exercises recursion depth)."""
    data = list(range(1000, 0, -1))
    assert merge_sort(data) == list(range(1, 1001))


def test_merge_sort_strings():
    """Comparable non-numeric items (strings) are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_floats():
    """Floating point values are sorted correctly."""
    assert merge_sort([2.5, -1.0, 0.0, 3.14]) == [-1.0, 0.0, 2.5, 3.14]


def test_merge_sort_tuples():
    """Tuples are sorted by their natural ordering."""
    assert merge_sort([(2, "b"), (1, "a"), (2, "a")]) == [(1, "a"), (2, "a"), (2, "b")]


def test_merge_sort_stability_of_equal_keys():
    """Equal elements keep their relative order (merge sort is stable)."""
    data = [(1, "first"), (0, "x"), (1, "second"), (0, "y")]
    result = merge_sort(data)
    assert result == [(0, "x"), (0, "y"), (1, "first"), (1, "second")]


def test_merge_sort_tuple_input_raises_attribute_error():
    """A tuple is immutable, but merge_sort requires a mutable collection."""
    with pytest.raises(AttributeError):
        merge_sort((3, 1, 2))


def test_merge_sort_raises_on_incomparable_items():
    """Mixing incomparable types raises a TypeError."""
    with pytest.raises(TypeError):
        merge_sort([1, "a", 2])
