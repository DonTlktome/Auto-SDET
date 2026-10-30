"""
Pytest test suite for sorts/merge_sort.py

Covers the public `merge_sort` function with happy-path, edge-case,
boundary, and property-based style tests. All tests are pure-Python and
require no network access.
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


def test_merge_sort_returns_new_list_and_does_not_mutate_input():
    """The input list must not be modified by the sort operation."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    result = merge_sort(original)
    assert result == [1, 2, 3, 4]
    assert original == snapshot


def test_merge_sort_handles_negative_numbers():
    """Negative values are ordered correctly (doctest example)."""
    assert merge_sort([-2, -5, -45]) == [-45, -5, -2]


def test_merge_sort_handles_floats():
    """Floating point values are sorted in ascending order."""
    assert merge_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_merge_sort_handles_strings():
    """Strings are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_handles_tuples_of_comparable_items():
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
        ([5, 5, 5, 5], [5, 5, 5, 5]),  # all duplicates
        ([3, 2, 1], [1, 2, 3]),   # fully reversed
        ([-1, 0, 1], [-1, 0, 1]), # already sorted with negatives
    ],
)
def test_merge_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, duplicates, reversed) sort correctly."""
    assert merge_sort(collection) == expected


def test_merge_sort_empty_list_returns_empty_list():
    """An empty list yields an empty list (doctest example)."""
    assert merge_sort([]) == []


def test_merge_sort_single_element_returns_same_object():
    """A list of length <= 1 is returned as-is (identity preserved)."""
    single = [7]
    result = merge_sort(single)
    assert result == [7]
    assert result is single


def test_merge_sort_odd_length_list():
    """Odd-length lists exercise the mid_index floor division branch."""
    assert merge_sort([9, 1, 8, 2, 7]) == [1, 2, 7, 8, 9]


def test_merge_sort_even_length_list():
    """Even-length lists split evenly at the midpoint."""
    assert merge_sort([9, 1, 8, 2, 7, 3]) == [1, 2, 3, 7, 8, 9]


def test_merge_sort_large_list_matches_builtin_sorted():
    """A large randomized list produces the same result as sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert merge_sort(list(data)) == sorted(data)


def test_merge_sort_is_stable_for_equal_keys():
    """Merge sort is stable: equal keys keep their original relative order."""
    data = [(1, "first"), (0, "x"), (1, "second"), (1, "third")]
    result = merge_sort(data)
    assert [item[1] for item in result if item[0] == 1] == ["first", "second", "third"]


def test_merge_sort_raises_on_incomparable_types():
    """Mixing incomparable types raises TypeError during merging."""
    with pytest.raises(TypeError):
        merge_sort([1, "a", 2])


def test_merge_sort_requires_a_mutable_sequence():
    """The merge step calls list.pop(0), so immutable inputs like tuples are
    not supported and raise AttributeError (the docstring documents a
    *mutable* ordered collection as the required input type)."""
    with pytest.raises(AttributeError):
        merge_sort((3, 1, 2))


def test_merge_sort_returns_a_list_for_list_input():
    """Sorting a list-like mutable sequence returns a list."""
    result = merge_sort([3, 1, 2])
    assert result == [1, 2, 3]
    assert isinstance(result, list)
