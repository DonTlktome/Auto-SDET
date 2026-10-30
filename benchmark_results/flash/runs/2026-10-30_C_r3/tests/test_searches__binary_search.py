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

import bisect
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
    """bisect_left respects the lo/hi slice bounds (hi=-1 means len)."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bisect_left([], 42) == 0


def test_bisect_left_matches_stdlib():
    """bisect_left agrees with the standard library implementation."""
    collection = sorted(random.Random(0).sample(range(1000), 100))
    for item in range(-5, 1005, 7):
        assert bisect_left(collection, item) == bisect.bisect_left(collection, item)


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -1, 0),
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
    """bisect_right respects the lo/hi slice bounds."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bisect_right([], 42) == 0


def test_bisect_right_duplicates():
    """bisect_right skips past all duplicates of the item."""
    assert bisect_right([1, 2, 2, 2, 3], 2) == 4
    assert bisect_left([1, 2, 2, 2, 3], 2) == 1


def test_bisect_right_matches_stdlib():
    """bisect_right agrees with the standard library implementation."""
    collection = sorted(random.Random(1).sample(range(1000), 100))
    for item in range(-5, 1005, 7):
        assert bisect_right(collection, item) == bisect.bisect_right(collection, item)


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_happy_path():
    """insort_left inserts the item in sorted position and returns None."""
    collection = [0, 5, 7, 10, 15]
    assert insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the new item before existing equal items."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (-10, [-10, 0, 5, 7, 10, 15]),
        (20, [0, 5, 7, 10, 15, 20]),
        (0, [0, 0, 5, 7, 10, 15]),
    ],
)
def test_insort_left_boundaries(item, expected):
    """insort_left handles items below, above and equal to the extremes."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, item)
    assert collection == expected


def test_insort_left_with_bounds():
    """insort_left honours lo/hi bounds, which may break global ordering."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_empty_collection():
    """insort_left on an empty list appends the item."""
    collection = []
    insort_left(collection, 3)
    assert collection == [3]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------


def test_insort_right_happy_path():
    """insort_right inserts the item in sorted position and returns None."""
    collection = [0, 5, 7, 10, 15]
    assert insort_right(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_inserts_after_duplicates():
    """insort_right places the new item after existing equal items."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[2]


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (-10, [-10, 0, 5, 7, 10, 15]),
        (20, [0, 5, 7, 10, 15, 20]),
        (15, [0, 5, 7, 10, 15, 15]),
    ],
)
def test_insort_right_boundaries(item, expected):
    """insort_right handles items below, above and equal to the extremes."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, item)
    assert collection == expected


def test_insort_right_with_bounds():
    """insort_right honours lo/hi bounds, which may break global ordering."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_empty_collection():
    """insort_right on an empty list appends the item."""
    collection = []
    insort_right(collection, 3)
    assert collection == [3]


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [(0, 0), (15, 4), (5, 1), (7, 2), (10, 3), (6, -1), (-1, -1), (100, -1)],
)
def test_binary_search_happy_path(item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert binary_search(SORTED, item) == expected


def test_binary_search_empty_collection():
    """binary_search on an empty collection returns -1."""
    assert binary_search([], 1) == -1


def test_binary_search_single_element():
    """binary_search works on a one-element collection."""
    assert binary_search([42], 42) == 0
    assert binary_search([42], 41) == -1


def test_binary_search_unsorted_raises():
    """binary_search raises ValueError for a non-ascending collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_index():
    """binary_search returns an index pointing at the item when duplicates exist."""
    collection = [1, 2, 2, 2, 3]
    index = binary_search(collection, 2)
    assert collection[index] == 2


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [(0, 0), (15, 4), (5, 1), (7, 2), (10, 3), (6, -1), (-1, -1), (100, -1)],
)
def test_binary_search_std_lib_happy_path(item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert binary_search_std_lib(SORTED, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib on an empty collection returns -1."""
    assert binary_search_std_lib([], 1) == -1


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib accepts any sequence, e.g. a range object."""
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
        ([1, 2, 2, 2, 3], 0, []),
        ([2, 2, 2], 2, [0, 1, 2]),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index of the item."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates on an empty collection returns []."""
    assert binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_with_duplicates([2, 1], 1)


def test_binary_search_with_duplicates_returns_list_type():
    """binary_search_with_duplicates always returns a list."""
    result = binary_search_with_duplicates([1, 1], 1)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [(0, 0), (15, 4), (5, 1), (7, 2), (10, 3), (6, -1), (100, -1)],
)
def test_binary_search_by_recursion_happy_path(item, expected):
    """binary_search_by_recursion finds the index or returns -1."""
    assert binary_search_by_recursion(SORTED, item, 0, 4) == expected


def test_binary_search_by_recursion_default_bounds():
    """binary_search_by_recursion computes right bound when right is negative."""
    assert binary_search_by_recursion(SORTED, 15) == 4
    assert binary_search_by_recursion(SORTED, 6) == -1


def test_binary_search_by_recursion_empty_collection():
    """binary_search_by_recursion on an empty collection returns -1."""
    assert binary_search_by_recursion([], 1) == -1


def test_binary_search_by_recursion_right_less_than_left():
    """binary_search_by_recursion returns -1 when right < left."""
    assert binary_search_by_recursion(SORTED, 0, 3, 1) == -1


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_by_recursion([9, 8, 7], 8, 0, 2)


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [(0, 0), (15, 4), (5, 1), (7, 2), (10, 3), (6, -1), (100, -1)],
)
def test_exponential_search_happy_path(item, expected):
    """exponential_search finds the index or returns -1."""
    assert exponential_search(SORTED, item) == expected


def test_exponential_search_empty_collection():
    """exponential_search on an empty collection returns -1."""
    assert exponential_search([], 1) == -1


def test_exponential_search_single_element():
    """exponential_search works on a one-element collection."""
    assert exponential_search([7], 7) == 0
    assert exponential_search([7], 8) == -1


def test_exponential_search_unsorted_raises():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        exponential_search([3, 2, 1], 2)


def test_exponential_search_large_collection():
    """exponential_search finds items in a large collection."""
    collection = list(range(0, 2000, 2))
    assert exponential_search(collection, 1998) == 999
    assert exponential_search(collection, 1999) == -1


# ---------------------------------------------------------------------------
# Cross-implementation consistency & module attributes
# ---------------------------------------------------------------------------


def _random_sorted_collection():
    """Return a deterministic ascending collection of 50 distinct integers."""
    rng = random.Random(42)
    return sorted(rng.sample(range(500), 50))


@pytest.mark.parametrize("search", [binary_search, binary_search_std_lib])
def test_iterative_searches_handle_targets_below_minimum(search):
    """The iterative searches report -1 for every target below the minimum."""
    collection = _random_sorted_collection()
    for item in range(-2, collection[0], 3):
        assert search(collection, item) == -1


@pytest.mark.parametrize(
    "search",
    [binary_search, binary_search_std_lib, exponential_search, binary_search_by_recursion],
)
def test_all_searches_agree(search):
    """All search implementations agree on a random sorted collection.

    Note: the recursion-based implementations (`binary_search_by_recursion` and
    `exponential_search`, which delegates to it) treat any ``right == -1`` as
    "use the last index".  A target strictly smaller than ``collection[0]``
    therefore drives the recursive call to ``right == -1`` and the search never
    terminates, so the target range starts at ``collection[0]`` here.  The
    below-minimum case is exercised for the iterative implementations in
    ``test_iterative_searches_handle_targets_below_minimum``.
    """
    collection = _random_sorted_collection()
    for item in range(collection[0], 502, 3):
        expected = collection.index(item) if item in collection else -1
        assert search(collection, item) == expected


def test_searches_tuple_contents():
    """The module-level `searches` tuple exposes the four search functions."""
    assert isinstance(searches, tuple)
    assert set(searches) == {
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    }
