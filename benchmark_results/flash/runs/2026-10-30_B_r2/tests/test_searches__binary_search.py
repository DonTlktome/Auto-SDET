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
    - module-level `searches` tuple
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
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi window, including hi=-1 meaning len()."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bisect_left([], 42) == 0


def test_bisect_left_matches_stdlib_on_random_data():
    """bisect_left agrees with the stdlib implementation on random sorted data."""
    rng = random.Random(1234)
    for _ in range(50):
        collection = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
        item = rng.randint(-60, 60)
        assert bisect_left(collection, item) == std_bisect.bisect_left(collection, item)


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -5, 0),
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
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi window."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bisect_right([], 42) == 0


def test_bisect_right_duplicates_boundary():
    """bisect_right skips past all duplicates of the item."""
    assert bisect_right([1, 2, 2, 2, 3], 2) == 4
    assert bisect_left([1, 2, 2, 2, 3], 2) == 1


def test_bisect_right_matches_stdlib_on_random_data():
    """bisect_right agrees with the stdlib implementation on random sorted data."""
    rng = random.Random(4321)
    for _ in range(50):
        collection = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
        item = rng.randint(-60, 60)
        assert bisect_right(collection, item) == std_bisect.bisect_right(
            collection, item
        )


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_happy_path():
    """insort_left inserts the item keeping the collection sorted."""
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
    assert item is not collection[0]


def test_insort_left_appends_at_end():
    """insort_left appends an item larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_left_with_bounds():
    """insort_left respects the lo/hi window when choosing the insert index."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_empty_collection():
    """insort_left on an empty collection simply appends the item."""
    collection = []
    insort_left(collection, 3)
    assert collection == [3]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------


def test_insort_right_happy_path():
    """insort_right inserts the item keeping the collection sorted."""
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
    assert item is not collection[0]


def test_insort_right_appends_at_end():
    """insort_right appends an item larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_right_with_bounds():
    """insort_right respects the lo/hi window when choosing the insert index."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_empty_collection():
    """insort_right on an empty collection simply appends the item."""
    collection = []
    insort_right(collection, 3)
    assert collection == [3]


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -100, -1),
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert binary_search(collection, item) == expected


def test_binary_search_empty_collection():
    """binary_search on an empty collection returns -1."""
    assert binary_search([], 1) == -1


def test_binary_search_single_element():
    """binary_search handles a one-element collection in both directions."""
    assert binary_search([7], 7) == 0
    assert binary_search([7], 8) == -1


def test_binary_search_unsorted_raises():
    """binary_search raises ValueError for a non-ascending collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_index():
    """binary_search returns an index pointing at the item when duplicates exist."""
    collection = [1, 2, 2, 2, 3]
    index = binary_search(collection, 2)
    assert collection[index] == 2


def test_binary_search_large_collection():
    """binary_search finds every element of a large sorted collection."""
    collection = list(range(0, 2000, 2))
    for expected, value in enumerate(collection):
        assert binary_search(collection, value) == expected


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
        ([0, 5, 7, 10, 15], -1, -1),
        ([0, 5, 7, 10, 15], 16, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib on an empty collection returns -1."""
    assert binary_search_std_lib([], 5) == -1


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib raises ValueError for a non-ascending collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sequence, e.g. a range object."""
    assert binary_search_std_lib(range(10), 4) == 4
    assert binary_search_std_lib(range(10), 10) == -1


def test_binary_search_std_lib_duplicates_returns_first_index():
    """binary_search_std_lib returns the first index of a duplicated item."""
    assert binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1


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
        ([5, 5, 5], 5, [0, 1, 2]),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns all indexes of the item."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates returns an empty list for an empty collection."""
    assert binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """binary_search_with_duplicates always returns a list, even when empty."""
    result = binary_search_with_duplicates([1, 3, 5], 2)
    assert isinstance(result, list)
    assert result == []


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
        ([0, 5, 7, 10, 15], 7, 0, -1, 2),
        ([0, 5, 7, 10, 15], 10, 2, 4, 3),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the item within the given bounds."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


def test_binary_search_by_recursion_empty_collection():
    """binary_search_by_recursion returns -1 for an empty collection."""
    assert binary_search_by_recursion([], 1) == -1


def test_binary_search_by_recursion_right_less_than_left():
    """binary_search_by_recursion returns -1 when right < left."""
    assert binary_search_by_recursion([1, 2, 3], 2, 2, 1) == -1


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_by_recursion([3, 2, 1], 2, 0, 2)


def test_binary_search_by_recursion_matches_iterative():
    """Recursive and iterative binary searches agree on random sorted data."""
    rng = random.Random(99)
    for _ in range(30):
        collection = sorted(rng.sample(range(200), rng.randint(1, 40)))
        for item in (collection[0], collection[-1], collection[len(collection) // 2]):
            assert binary_search_by_recursion(collection, item) == binary_search(
                collection, item
            )


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
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_empty_collection():
    """exponential_search returns -1 for an empty collection."""
    assert exponential_search([], 1) == -1


def test_exponential_search_single_element():
    """exponential_search handles a one-element collection."""
    assert exponential_search([9], 9) == 0


def test_exponential_search_unsorted_raises():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        exponential_search([9, 8, 7], 8)


def test_exponential_search_large_collection():
    """exponential_search finds elements across a large sorted collection."""
    collection = list(range(0, 5000, 5))
    for expected in (0, 1, 10, 500, len(collection) - 1):
        assert exponential_search(collection, collection[expected]) == expected


# ---------------------------------------------------------------------------
# module-level `searches` tuple
# ---------------------------------------------------------------------------


def test_searches_tuple_contents():
    """The `searches` tuple exposes the four search callables in order."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )


def test_searches_tuple_all_callables_find_item():
    """Every callable in `searches` locates an existing item correctly."""
    for search in searches:
        assert search(SORTED, 10) == 3


def test_searches_tuple_all_callables_report_missing_item():
    """Every callable in `searches` returns -1 for a missing item."""
    for search in searches:
        assert search(SORTED, 6) == -1
