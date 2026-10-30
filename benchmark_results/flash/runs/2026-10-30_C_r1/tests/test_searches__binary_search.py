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
        ([], 3, 0),
        ([1], 0, 0),
        ([1], 1, 0),
        ([1], 2, 1),
        ([1, 1, 1], 1, 0),
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
        ([0, 5, 7, 10, 15], 0, 2, -1, 2),
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
        ([0, 5, 7, 10, 15], 100, 5, -1, 5),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi window, including hi == -1 meaning len()."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_matches_stdlib_on_random_data():
    """bisect_left agrees with the standard library bisect.bisect_left."""
    rng = random.Random(1234)
    for _ in range(50):
        collection = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
        item = rng.randint(-60, 60)
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
        ([0, 5, 7, 10, 15], 20, 5),
        ([0, 5, 7, 10, 15], -5, 0),
        ([], 3, 0),
        ([1], 1, 1),
        ([1, 1, 1], 1, 3),
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
        ([0, 5, 7, 10, 15], 0, 0, 0, 0),
        ([0, 5, 7, 10, 15], 100, 5, -1, 5),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi window, including hi == -1 meaning len()."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_matches_stdlib_on_random_data():
    """bisect_right agrees with the standard library bisect.bisect_right."""
    rng = random.Random(4321)
    for _ in range(50):
        collection = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
        item = rng.randint(-60, 60)
        assert bisect_right(collection, item) == std_bisect.bisect_right(
            collection, item
        )


def test_bisect_left_and_right_differ_on_duplicates():
    """For a duplicated value, bisect_left < bisect_right."""
    collection = [1, 2, 2, 2, 3]
    assert bisect_left(collection, 2) == 1
    assert bisect_right(collection, 2) == 4


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 4, [4]),
        ([1, 1, 1], 1, [1, 1, 1, 1]),
    ],
)
def test_insort_left_happy_path(collection, item, expected):
    """insort_left inserts the item in place, keeping the list sorted."""
    insort_left(collection, item)
    assert collection == expected


def test_insort_left_inserts_before_equal_values():
    """insort_left places the new item before existing equal values."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = tuple([5, 5])
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]
    assert item is not collection[2]


def test_insort_left_with_bounds():
    """insort_left respects the lo/hi window when choosing the insertion index."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_returns_none():
    """insort_left mutates in place and returns None."""
    collection = [1, 3]
    assert insort_left(collection, 2) is None


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 4, [4]),
        ([1, 1, 1], 1, [1, 1, 1, 1]),
    ],
)
def test_insort_right_happy_path(collection, item, expected):
    """insort_right inserts the item in place, keeping the list sorted."""
    insort_right(collection, item)
    assert collection == expected


def test_insort_right_inserts_after_equal_values():
    """insort_right places the new item after existing equal values."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = tuple([5, 5])
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is not collection[1]
    assert item is collection[2]


def test_insort_right_with_bounds():
    """insort_right respects the lo/hi window when choosing the insertion index."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_returns_none():
    """insort_right mutates in place and returns None."""
    collection = [1, 3]
    assert insort_right(collection, 2) is None


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 7, 2),
        ([0, 5, 7, 10, 15], 10, 3),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -100, -1),
        ([0, 5, 7, 10, 15], 100, -1),
        ([42], 42, 0),
        ([42], 41, -1),
        ([], 1, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item, or -1 when absent."""
    assert binary_search(collection, item) == expected


def test_binary_search_raises_on_unsorted_collection():
    """binary_search raises ValueError when the collection is not ascending."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_accepts_duplicates():
    """binary_search returns some index of the item when duplicates exist."""
    collection = [1, 2, 2, 2, 3]
    result = binary_search(collection, 2)
    assert collection[result] == 2


def test_binary_search_does_not_mutate_collection():
    """binary_search is read-only with respect to the input collection."""
    collection = [0, 5, 7, 10, 15]
    binary_search(collection, 7)
    assert collection == [0, 5, 7, 10, 15]


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -1, -1),
        ([0, 5, 7, 10, 15], 16, -1),
        ([7], 7, 0),
        ([7], 8, -1),
        ([], 0, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item, or -1 when absent."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_raises_on_unsorted_collection():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sequence, e.g. a range object."""
    assert binary_search_std_lib(range(1000), 500) == 500
    assert binary_search_std_lib(range(1000), 1000) == -1


def test_binary_search_std_lib_returns_first_duplicate_index():
    """binary_search_std_lib returns the leftmost index of a duplicated value."""
    assert binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1


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
        ([1, 2, 2, 2, 3], 0, []),
        ([5, 5, 5, 5], 5, [0, 1, 2, 3]),
        ([], 1, []),
        ([1], 1, [0]),
        ([1], 2, []),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index where item occurs."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_raises_on_unsorted_collection():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value is always a list, even when nothing is found."""
    result = binary_search_with_duplicates([1, 2, 3], 99)
    assert isinstance(result, list)
    assert result == []


def test_binary_search_with_duplicates_indexes_are_valid():
    """Every returned index actually points at the searched item."""
    collection = [0, 1, 1, 1, 2, 3, 3, 4]
    for item in set(collection):
        indexes = binary_search_with_duplicates(collection, item)
        assert indexes == [i for i, value in enumerate(collection) if value == item]


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, left, right, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0, 4, 0),
        ([0, 5, 7, 10, 15], 15, 0, 4, 4),
        ([0, 5, 7, 10, 15], 5, 0, 4, 1),
        ([0, 5, 7, 10, 15], 6, 0, 4, -1),
        ([0, 5, 7, 10, 15], 0, 0, -1, 0),
        ([0, 5, 7, 10, 15], 15, 0, -1, 4),
        ([0, 5, 7, 10, 15], 6, 0, -1, -1),
        ([0, 5, 7, 10, 15], 7, 2, 2, 2),
        ([0, 5, 7, 10, 15], 10, 0, 1, -1),
        ([], 1, 0, -1, -1),
        ([9], 9, 0, -1, 0),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the item within the given window."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


def test_binary_search_by_recursion_raises_on_unsorted_collection():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_by_recursion([3, 2, 1], 2, 0, 2)


def test_binary_search_by_recursion_empty_window_returns_minus_one():
    """A window where right < left yields -1 without recursing further."""
    assert binary_search_by_recursion([1, 2, 3], 1, 2, 1) == -1


def test_binary_search_by_recursion_matches_iterative():
    """Recursive and iterative binary search agree on random sorted data."""
    rng = random.Random(99)
    for _ in range(30):
        collection = sorted(rng.sample(range(200), rng.randint(1, 40)))
        for item in (collection[0], collection[-1], 500):
            assert binary_search_by_recursion(
                collection, item, 0, len(collection) - 1
            ) == binary_search(collection, item)


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], 1, -1),
        ([0, 5, 7, 10, 15], 100, -1),
        ([1], 1, 0),
        ([1], 2, -1),
        ([], 1, -1),
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item, or -1 when absent."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_raises_on_unsorted_collection():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        exponential_search([10, 9, 8], 9)


def test_exponential_search_large_collection():
    """exponential_search finds items across a large sorted collection."""
    collection = list(range(0, 2000, 2))
    assert exponential_search(collection, 0) == 0
    assert exponential_search(collection, 1998) == 999
    assert exponential_search(collection, 1000) == 500
    assert exponential_search(collection, 1001) == -1


def test_exponential_search_matches_binary_search():
    """exponential_search agrees with binary_search on random sorted data."""
    rng = random.Random(7)
    for _ in range(30):
        collection = sorted(rng.sample(range(500), rng.randint(1, 50)))
        for item in (collection[0], collection[-1], collection[len(collection) // 2]):
            assert exponential_search(collection, item) == binary_search(
                collection, item
            )


# ---------------------------------------------------------------------------
# module-level `searches` tuple
# ---------------------------------------------------------------------------


def test_searches_tuple_contains_all_search_functions():
    """The `searches` tuple exposes the four search callables in order."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )


@pytest.mark.parametrize("search", searches)
def test_every_search_function_finds_and_misses(search):
    """Each registered search function finds a present item and misses an absent one."""
    assert search([0, 5, 7, 10, 15], 10) == 3
    assert search([0, 5, 7, 10, 15], 11) == -1
