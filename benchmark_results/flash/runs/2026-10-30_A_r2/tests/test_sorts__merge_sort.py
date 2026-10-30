"""
Pytest test suite for sorts/merge_sort.py

Covers the public `merge_sort` function with happy-path, edge-case,
boundary, and property-style tests. No network access or third-party
dependencies are required.
"""

import random

import pytest

from sorts.merge_sort import merge_sort


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

def test_merge_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_merge_sort_returns_list_type():
    """The return value is always a list instance."""
    result = merge_sort([3, 1, 2])
    assert isinstance(result, list)


def test_merge_sort_does_not_mutate_input():
    """The original input collection must be left untouched."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    merge_sort(original)
    assert original == snapshot


def test_merge_sort_sorts_strings():
    """Comparable non-numeric items (strings) are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_sorts_floats():
    """Floating point values are sorted correctly."""
    assert merge_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_merge_sort_sorts_tuples_of_comparable_items():
    """Tuples are compared element-wise and sorted accordingly."""
    assert merge_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                 # empty list
        ([42], [42]),             # single element
        ([1, 2], [1, 2]),         # already sorted, two elements
        ([2, 1], [1, 2]),         # reversed, two elements
        ([5, 5, 5], [5, 5, 5]),   # all duplicates
        ([-2, -5, -45], [-45, -5, -2]),  # all negative
        ([0], [0]),               # single zero
        ([3, 1, 2], [1, 2, 3]),   # odd length
        ([4, 3, 2, 1], [1, 2, 3, 4]),  # even length, reversed
    ],
)
def test_merge_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, duplicates, negatives) sort correctly."""
    assert merge_sort(collection) == expected


def test_merge_sort_empty_list_returns_empty_list():
    """An empty input yields an empty list (not None)."""
    result = merge_sort([])
    assert result == []
    assert isinstance(result, list)


def test_merge_sort_single_element_returns_same_value():
    """A one-element list is returned with the same single value."""
    assert merge_sort([7]) == [7]


def test_merge_sort_already_sorted_is_stable_output():
    """An already sorted list is returned unchanged."""
    data = list(range(10))
    assert merge_sort(data) == data


def test_merge_sort_reverse_sorted():
    """A reverse-sorted list is fully reversed into ascending order."""
    data = list(range(10, 0, -1))
    assert merge_sort(data) == list(range(1, 11))


def test_merge_sort_large_values():
    """Very large and very small integers are handled without overflow issues."""
    data = [10**18, -(10**18), 0, 10**17, -(10**17)]
    assert merge_sort(data) == [-(10**18), -(10**17), 0, 10**17, 10**18]


def test_merge_sort_duplicates_preserved_count():
    """Duplicate values are all retained (no elements lost)."""
    data = [2, 1, 2, 1, 2, 1]
    result = merge_sort(data)
    assert result == [1, 1, 1, 2, 2, 2]
    assert len(result) == len(data)


# ---------------------------------------------------------------------------
# Property-style / randomized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_merge_sort_matches_builtin_sorted(seed):
    """For random inputs, merge_sort output equals Python's built-in sorted()."""
    rng = random.Random(seed)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert merge_sort(data) == sorted(data)


def test_merge_sort_output_is_non_decreasing():
    """The output sequence is monotonically non-decreasing."""
    rng = random.Random(7)
    data = [rng.randint(-50, 50) for _ in range(150)]
    result = merge_sort(data)
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_merge_sort_preserves_multiset_of_elements():
    """Sorting preserves the multiset of elements (same counts)."""
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    result = merge_sort(data)
    assert sorted(result) == sorted(data)
    assert len(result) == len(data)


def test_merge_sort_rejects_tuple_input():
    """The function is documented for *mutable* ordered collections.

    A tuple is immutable; the internal merge() relies on list.pop(0), so a
    tuple input raises AttributeError rather than being sorted.
    """
    with pytest.raises(AttributeError):
        merge_sort((3, 1, 2))


def test_merge_sort_does_not_raise_on_incomparable_types():
    """Mixed incomparable types raise TypeError from comparison, not silently pass."""
    with pytest.raises(TypeError):
        merge_sort([1, "a", 2])
