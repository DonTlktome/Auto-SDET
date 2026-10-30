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

import bisect as std_bisect
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
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], 20, 5),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], -1, 0),
        ([1, 2, 2, 2, 3], 2, 1),
    ],
)
def test_bisect_left_happy_path(collection, item, expected):
    """bisect_left returns the first index whose value is >= item."""
    assert bisect_left(collection, item) == expected


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi slice bounds, including hi=-1 meaning len()."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bisect_left([], 42) == 0


def test_bisect_left_matches_stdlib():
    """bisect_left agrees with the standard library implementation."""
    collection = sorted(random.Random(0).sample(range(1000), 100))
    for item in range(-5, 1005, 7):
        assert bisect_left(collection, item) == std_bisect.bisect_left(collection, item)


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -1, 0),
        ([1, 2, 2, 2, 3], 2, 4),
    ],
)
def test_bisect_right_happy_path(collection, item, expected):
    """bisect_right returns the first index whose value is > item."""
    assert bisect_right(collection, item) == expected


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi slice bounds."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bisect_right([], 42) == 0


def test_bisect_right_matches_stdlib():
    """bisect_right agrees with the standard library implementation."""
    collection = sorted(random.Random(1).sample(range(1000), 100))
    for item in range(-5, 1005, 7):
        assert bisect_right(collection, item) == std_bisect.bisect_right(
            collection, item
        )


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_inserts_in_middle():
    """insort_left inserts a new value in the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the item before existing equal values (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Build the item at runtime: a literal (5, 5) would be the *same* object as the
    # constant already inside `collection`, and identity checks would be meaningless.
    item = tuple([5, 5])
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]
    assert item is not collection[2]


@pytest.mark.parametrize(
    "item, expected",
    [
        (-10, [-10, 0, 5, 7, 10, 15]),
        (20, [0, 5, 7, 10, 15, 20]),
        (0, [0, 0, 5, 7, 10, 15]),
    ],
)
def test_insort_left_boundaries(item, expected):
    """insort_left handles insertion at the very start, end and on a duplicate."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, item)
    assert collection == expected


def test_insort_left_with_bounds():
    """insort_left respects lo/hi bounds, which may leave the list unsorted."""
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


def test_insort_right_inserts_in_middle():
    """insort_right inserts a new value in the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_right(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_inserts_after_duplicates():
    """insort_right places the item after existing equal values (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Build the item at runtime: a literal (5, 5) would be the *same* object as the
    # constant already inside `collection`, and identity checks would be meaningless.
    item = tuple([5, 5])
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is not collection[1]
    assert item is collection[2]


@pytest.mark.parametrize(
    "item, expected",
    [
        (-10, [-10, 0, 5, 7, 10, 15]),
        (20, [0, 5, 7, 10, 15, 20]),
        (0, [0, 0, 5, 7, 10, 15]),
    ],
)
def test_insort_right_boundaries(item, expected):
    """insort_right handles insertion at the very start, end and on a duplicate."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, item)
    assert collection == expected


def test_insort_right_with_bounds():
    """insort_right respects lo/hi bounds, which may leave the list unsorted."""
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
    "item, expected",
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
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_index():
    """binary_search with duplicates returns an index holding the item."""
    collection = [1, 2, 2, 2, 3]
    result = binary_search(collection, 2)
    assert collection[result] == 2


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, expected",
    [(0, 0), (15, 4), (5, 1), (6, -1), (-1, -1), (100, -1)],
)
def test_binary_search_std_lib_happy_path(item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert binary_search_std_lib(SORTED, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib on an empty collection returns -1."""
    assert binary_search_std_lib([], 1) == -1


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib raises ValueError for a non-ascending collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_std_lib([3, 1, 2], 1)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib accepts any sequence, e.g. a range object."""
    assert binary_search_std_lib(range(1000), 500) == 500
    assert binary_search_std_lib(range(1000), 1000) == -1


# ---------------------------------------------------------------------------
# binary_search_with_duplicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, [0]),
        ([0, 5, 7, 10, 15], 15, [4]),
        ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
        ([1, 2, 2, 2, 3], 4, []),
        ([2, 2, 2], 2, [0, 1, 2]),
        ([1, 2, 3], 2, [1]),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns all indexes of the item."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates on an empty collection returns []."""
    assert binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_with_duplicates([3, 1, 2], 1)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value is always a list, even for a single match."""
    result = binary_search_with_duplicates([1, 2, 3], 2)
    assert isinstance(result, list)
    assert result == [1]


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, expected",
    [(0, 0), (15, 4), (5, 1), (6, -1), (100, -1)],
)
def test_binary_search_by_recursion_happy_path(item, expected):
    """binary_search_by_recursion returns the index of the item or -1."""
    assert binary_search_by_recursion(SORTED, item, 0, 4) == expected


@pytest.mark.xfail(
    reason="binary_search_by_recursion treats a negative `right` as 'use len-1', so a "
    "recursive call with midpoint-1 == -1 restarts the whole range and never "
    "terminates when the item is smaller than every element",
    raises=RecursionError,
    strict=False,
)
def test_binary_search_by_recursion_below_minimum():
    """Known limitation: searching below the smallest element overflows the stack."""
    assert binary_search_by_recursion(SORTED, -1, 0, 4) == -1


def test_binary_search_by_recursion_default_right():
    """binary_search_by_recursion computes right=len-1 when right is negative."""
    assert binary_search_by_recursion(SORTED, 15) == 4
    assert binary_search_by_recursion(SORTED, 6) == -1


def test_binary_search_by_recursion_empty_collection():
    """binary_search_by_recursion on an empty collection returns -1."""
    assert binary_search_by_recursion([], 1) == -1


def test_binary_search_by_recursion_right_less_than_left():
    """binary_search_by_recursion returns -1 when right < left."""
    assert binary_search_by_recursion(SORTED, 5, 3, 1) == -1


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_by_recursion([3, 1, 2], 1, 0, 2)


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, expected",
    [(0, 0), (15, 4), (5, 1), (7, 2), (10, 3), (6, -1), (100, -1)],
)
def test_exponential_search_happy_path(item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert exponential_search(SORTED, item) == expected


@pytest.mark.xfail(
    reason="exponential_search delegates to binary_search_by_recursion, which treats "
    "a negative `right` as 'use len-1'; an item below the minimum therefore "
    "recurses forever instead of returning -1",
    raises=RecursionError,
    strict=False,
)
def test_exponential_search_below_minimum():
    """Known limitation: searching below the smallest element overflows the stack."""
    assert exponential_search(SORTED, -1) == -1


def test_exponential_search_empty_collection():
    """exponential_search on an empty collection returns -1."""
    assert exponential_search([], 1) == -1


def test_exponential_search_single_element():
    """exponential_search works on a one-element collection."""
    assert exponential_search([42], 42) == 0
    assert exponential_search([42], 43) == -1


@pytest.mark.xfail(
    reason="same binary_search_by_recursion `right < 0` sentinel issue for items "
    "below the smallest element",
    raises=RecursionError,
    strict=False,
)
def test_exponential_search_single_element_below():
    """Known limitation: a too-small item on a one-element list recurses forever."""
    assert exponential_search([42], 41) == -1


def test_exponential_search_large_collection():
    """exponential_search finds items across a large collection."""
    collection = list(range(0, 2000, 2))
    assert exponential_search(collection, 0) == 0
    assert exponential_search(collection, 1998) == 999
    assert exponential_search(collection, 1000) == 500
    assert exponential_search(collection, 1001) == -1


def test_exponential_search_unsorted_raises():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        exponential_search([3, 1, 2], 1)


# ---------------------------------------------------------------------------
# Cross-implementation consistency & module attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", searches)
def test_all_searches_agree(search):
    """Every search implementation in `searches` agrees on the same inputs."""
    collection = [0, 5, 7, 10, 15]
    for item, expected in [(0, 0), (5, 1), (7, 2), (10, 3), (15, 4), (6, -1)]:
        assert search(collection, item) == expected


def test_searches_tuple_contents():
    """The module-level `searches` tuple contains the expected callables."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )
    assert all(callable(fn) for fn in searches)
