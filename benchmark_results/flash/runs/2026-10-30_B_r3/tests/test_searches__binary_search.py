#!/usr/bin/env python3

"""
Pytest test suite for searches/binary_search.py

Covers:
    - bisect_left
    - bisect_right
    - insort_left
    - insort_right
    - binary_search
    - binary_search_std_lib
    - binary_search_with_duplicates
    - binary_search_by_recursion
    - exponential_search
    - the module-level `searches` tuple
"""

import bisect as stdlib_bisect
import random

import pytest

from searches.binary_search import (
    binary_search,
    binary_search_by_recursion,
    binary_search_std_lib,
    binary_search_with_duplicates,
    bisect_left,
    bisect_right,
    exponential_search,
    insort_left,
    insort_right,
    searches,
)


SORTED = [0, 5, 7, 10, 15]


# ---------------------------------------------------------------------------
# bisect_left
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], 20, 5),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], -1, 0),
    ],
)
def test_bisect_left_happy_path(collection, item, expected):
    """bisect_left returns the first index whose value is >= item."""
    assert bisect_left(collection, item) == expected


@pytest.mark.parametrize(
    ("collection", "item", "lo", "hi", "expected"),
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, -1, 2),
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi slice bounds, including hi=-1 meaning len()."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bisect_left([], 42) == 0


def test_bisect_left_duplicates_returns_first_index():
    """bisect_left returns the leftmost index among equal elements."""
    assert bisect_left([1, 2, 2, 2, 3], 2) == 1


def test_bisect_left_matches_stdlib():
    """bisect_left agrees with the standard library implementation."""
    collection = sorted(random.Random(0).sample(range(1000), 200))
    for item in range(-5, 1005, 7):
        assert bisect_left(collection, item) == stdlib_bisect.bisect_left(
            collection, item
        )


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], 5, 2),
        ([0, 5, 7, 10, 15], -10, 0),
        ([0, 5, 7, 10, 15], 100, 5),
    ],
)
def test_bisect_right_happy_path(collection, item, expected):
    """bisect_right returns the first index whose value is > item."""
    assert bisect_right(collection, item) == expected


@pytest.mark.parametrize(
    ("collection", "item", "lo", "hi", "expected"),
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, -1, 2),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi slice bounds."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bisect_right([], 1) == 0


def test_bisect_right_duplicates_returns_after_last():
    """bisect_right returns the index just past the last equal element."""
    assert bisect_right([1, 2, 2, 2, 3], 2) == 4


def test_bisect_right_matches_stdlib():
    """bisect_right agrees with the standard library implementation."""
    collection = sorted(random.Random(1).sample(range(1000), 200))
    for item in range(-5, 1005, 7):
        assert bisect_right(collection, item) == stdlib_bisect.bisect_right(
            collection, item
        )


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_inserts_in_middle():
    """insort_left inserts a new value at the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the item before existing equal items (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Build the item as a *fresh* tuple: equal tuple literals inside a single
    # code object are folded into one constant, which would make the identity
    # assertions below meaningless (and falsely fail).
    item = tuple([5, 5])
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]
    assert item is not collection[2]


def test_insort_left_appends_at_end():
    """insort_left appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_left_into_empty_collection():
    """insort_left on an empty collection produces a single-element list."""
    collection = []
    insort_left(collection, 3)
    assert collection == [3]


def test_insort_left_with_bounds():
    """insort_left respects lo/hi bounds, which may break global ordering."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------


def test_insort_right_inserts_in_middle():
    """insort_right inserts a new value at the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_right(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_inserts_after_duplicates():
    """insort_right places the item after existing equal items (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Fresh tuple object, see test_insort_left_inserts_before_duplicates.
    item = tuple([5, 5])
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is not collection[1]
    assert item is collection[2]


def test_insort_right_appends_at_end():
    """insort_right appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_right_into_empty_collection():
    """insort_right on an empty collection produces a single-element list."""
    collection = []
    insort_right(collection, 3)
    assert collection == [3]


def test_insort_right_with_bounds():
    """insort_right respects lo/hi bounds, which may break global ordering."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 7, 2),
        ([0, 5, 7, 10, 15], 10, 3),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -1, -1),
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert binary_search(collection, item) == expected


def test_binary_search_empty_collection():
    """binary_search on an empty collection returns -1."""
    assert binary_search([], 1) == -1


def test_binary_search_single_element_found():
    """binary_search finds the only element of a one-element collection."""
    assert binary_search([42], 42) == 0


def test_binary_search_single_element_not_found():
    """binary_search returns -1 for a one-element collection without the item."""
    assert binary_search([42], 7) == -1


def test_binary_search_unsorted_raises():
    """binary_search raises ValueError when the collection is not ascending."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_valid_index():
    """binary_search returns an index that actually holds the searched item."""
    collection = [1, 2, 2, 2, 3]
    index = binary_search(collection, 2)
    assert collection[index] == 2


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -3, -1),
        ([0, 5, 7, 10, 15], 99, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib on an empty collection returns -1."""
    assert binary_search_std_lib([], 5) == -1


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_duplicates_returns_first_index():
    """binary_search_std_lib returns the leftmost index of duplicated items."""
    assert binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sorted sequence, e.g. a range."""
    assert binary_search_std_lib(range(10), 7) == 7
    assert binary_search_std_lib(range(10), 10) == -1


# ---------------------------------------------------------------------------
# binary_search_with_duplicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, [0]),
        ([0, 5, 7, 10, 15], 15, [4]),
        ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
        ([1, 2, 2, 2, 3], 4, []),
        ([1, 2, 2, 2, 3], 1, [0]),
        ([1, 2, 2, 2, 3], 3, [4]),
        ([1, 2, 2, 2, 3], 0, []),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index where item occurs."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates returns [] for an empty collection."""
    assert binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_all_equal():
    """binary_search_with_duplicates returns all indexes when all values match."""
    assert binary_search_with_duplicates([7, 7, 7, 7], 7) == [0, 1, 2, 3]


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "left", "right", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0, 4, 0),
        ([0, 5, 7, 10, 15], 15, 0, 4, 4),
        ([0, 5, 7, 10, 15], 5, 0, 4, 1),
        ([0, 5, 7, 10, 15], 6, 0, 4, -1),
        ([0, 5, 7, 10, 15], 10, 0, 4, 3),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the item index within [left, right]."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


def test_binary_search_by_recursion_default_bounds():
    """binary_search_by_recursion defaults right to len(collection) - 1."""
    assert binary_search_by_recursion(SORTED, 15) == 4
    assert binary_search_by_recursion(SORTED, 6) == -1


def test_binary_search_by_recursion_empty_collection():
    """binary_search_by_recursion returns -1 for an empty collection."""
    assert binary_search_by_recursion([], 1) == -1


def test_binary_search_by_recursion_right_less_than_left():
    """binary_search_by_recursion returns -1 when right < left."""
    assert binary_search_by_recursion(SORTED, 5, 3, 1) == -1


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_by_recursion([9, 8, 7], 8)


def test_binary_search_by_recursion_restricted_range():
    """binary_search_by_recursion only searches inside the given bounds."""
    assert binary_search_by_recursion(SORTED, 0, 1, 4) == -1
    assert binary_search_by_recursion(SORTED, 15, 1, 4) == 4


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], 8, -1),
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_empty_collection():
    """exponential_search returns -1 for an empty collection."""
    assert exponential_search([], 3) == -1


def test_exponential_search_single_element():
    """exponential_search handles a one-element collection."""
    assert exponential_search([9], 9) == 0
    assert exponential_search([9], 10) == -1


def test_exponential_search_unsorted_raises():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        exponential_search([4, 3, 2, 1], 3)


def test_exponential_search_large_collection():
    """exponential_search finds items in a large collection."""
    collection = list(range(0, 2000, 2))
    assert exponential_search(collection, 0) == 0
    assert exponential_search(collection, 1998) == 999
    assert exponential_search(collection, 1000) == 500
    assert exponential_search(collection, 1001) == -1


# ---------------------------------------------------------------------------
# Cross-implementation consistency & module attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "search",
    [
        binary_search,
        binary_search_std_lib,
        binary_search_by_recursion,
        exponential_search,
    ],
)
def test_all_searches_agree_on_sorted_collection(search):
    """Every search implementation agrees on found/not-found results."""
    collection = sorted(random.Random(42).sample(range(500), 100))
    for item in collection:
        index = search(collection, item)
        assert index != -1
        assert collection[index] == item
    present = set(collection)
    absent_items = [
        value for value in range(collection[0], 500) if value not in present
    ]
    assert len(absent_items) >= 3
    for item in absent_items[:3]:
        assert search(collection, item) == -1


def test_searches_tuple_contents():
    """The module-level `searches` tuple lists the four search functions."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )


def test_searches_tuple_is_iterable_and_callable():
    """Each entry of `searches` is callable and finds a known value."""
    for search in searches:
        assert callable(search)
        assert search([0, 5, 7, 10, 15], 10) == 3
