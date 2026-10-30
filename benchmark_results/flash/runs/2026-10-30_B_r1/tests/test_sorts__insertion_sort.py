"""Tests for the insertion_sort module.

The module under test provides a single public function, ``insertion_sort``,
which sorts a mutable sequence in place and returns it.
"""

from __future__ import annotations

import random
import string

import pytest

from sorts.insertion_sort import insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_insertion_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_insertion_sort_returns_same_object():
    """The function sorts in place and returns the very same object."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_insertion_sort_matches_builtin_sorted_on_random_ints():
    """Random integer collections are sorted identically to ``sorted``."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


def test_insertion_sort_matches_builtin_sorted_on_random_strings():
    """Random string collections are sorted identically to ``sorted``."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


def test_insertion_sort_empty_collection():
    """An empty collection is returned unchanged."""
    collection: list[int] = []
    assert insertion_sort(collection) == []
    assert collection == []


def test_insertion_sort_single_element():
    """A one-element collection is already sorted and left untouched."""
    assert insertion_sort([42]) == [42]


def test_insertion_sort_two_elements_swapped():
    """The smallest non-trivial case: two elements out of order."""
    assert insertion_sort([2, 1]) == [1, 2]


def test_insertion_sort_already_sorted():
    """An already sorted collection stays sorted."""
    assert insertion_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_insertion_sort_reverse_sorted():
    """A reverse sorted collection is fully reversed into ascending order."""
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_insertion_sort_all_equal_elements():
    """Duplicate-only collections are handled without reordering issues."""
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_insertion_sort_negative_numbers():
    """Negative numbers are ordered correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_insertion_sort_mixed_sign_numbers():
    """A mix of negative, zero and positive values is ordered correctly."""
    assert insertion_sort([0, -1, 5, -10, 3]) == [-10, -1, 0, 3, 5]


def test_insertion_sort_floats():
    """Floating point values are ordered correctly."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_insertion_sort_strings():
    """String values are ordered lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_insertion_sort_preserves_length_and_multiset():
    """Sorting never adds or removes elements."""
    collection = [4, 1, 4, 2, 9, 1]
    result = insertion_sort(collection)
    assert len(result) == 6
    assert sorted(result) == sorted([4, 1, 4, 2, 9, 1])


def test_insertion_sort_works_on_mutable_sequence_subclass():
    """Any MutableSequence (e.g. a list subclass) is supported."""
    class MyList(list):
        pass

    collection = MyList([3, 1, 2])
    result = insertion_sort(collection)
    assert isinstance(result, MyList)
    assert list(result) == [1, 2, 3]


def test_insertion_sort_does_not_mutate_input_when_already_sorted():
    """Sorting an already sorted list leaves its contents identical."""
    collection = [1, 2, 3]
    snapshot = list(collection)
    insertion_sort(collection)
    assert collection == snapshot


# ---------------------------------------------------------------------------
# Parametrized coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2]),
        ([2, 1], [1, 2]),
        ([3, 3, 3], [3, 3, 3]),
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([-2, -5, -45], [-45, -5, -2]),
        (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),
        ([10, -10, 0], [-10, 0, 10]),
        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]),
        ([6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]),
    ],
)
def test_insertion_sort_parametrized(collection, expected):
    """Various collections are sorted to the expected ascending order."""
    assert insertion_sort(collection) == expected


@pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 50])
def test_insertion_sort_matches_sorted_for_various_sizes(size):
    """For a range of collection sizes the result equals ``sorted``."""
    rng = random.Random(size)
    collection = [rng.randint(-100, 100) for _ in range(size)]
    assert insertion_sort(collection) == sorted(collection)


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------


def test_insertion_sort_raises_on_immutable_sequence():
    """Passing an immutable sequence (tuple) raises TypeError on assignment."""
    with pytest.raises(TypeError):
        insertion_sort((3, 1, 2))


def test_insertion_sort_raises_on_incomparable_elements():
    """Elements that cannot be compared raise TypeError."""
    with pytest.raises(TypeError):
        insertion_sort([1, "a"])