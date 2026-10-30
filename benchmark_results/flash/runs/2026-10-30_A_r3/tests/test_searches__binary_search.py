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
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], 20, 5),
        ([0, 5, 7, 10, 15], -1, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([], 3, 0),
        ([1, 1, 1, 1], 1, 0),
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
        ([0, 5, 7, 10, 15], 100, 5, 5, 5),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left respects the lo/hi slice bounds, including empty ranges."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_matches_stdlib_on_random_data():
    """bisect_left agrees with the standard library implementation."""
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
        ([], 3, 0),
        ([1, 1, 1, 1], 1, 4),
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
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right respects the lo/hi slice bounds."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_matches_stdlib_on_random_data():
    """bisect_right agrees with the standard library implementation."""
    rng = random.Random(4321)
    for _ in range(50):
        collection = sorted(rng.randint(-50, 50) for _ in range(rng.randint(0, 30)))
        item = rng.randint(-60, 60)
        assert bisect_right(collection, item) == std_bisect.bisect_right(
            collection, item
        )


def test_bisect_left_and_right_differ_on_duplicates():
    """For a duplicated value, left < right and both bracket the run."""
    collection = [1, 2, 2, 2, 3]
    assert bisect_left(collection, 2) == 1
    assert bisect_right(collection, 2) == 4


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_inserts_in_middle():
    """insort_left inserts a new value at the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the item before existing equal values."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is item
    assert collection[2] == (5, 5)


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 4, [4]),
        ([3], 3, [3, 3]),
    ],
)
def test_insort_left_boundaries(collection, item, expected):
    """insort_left handles empty lists, single elements and out-of-range values."""
    insort_left(collection, item)
    assert collection == expected


def test_insort_left_with_bounds():
    """insort_left honours lo/hi, which may leave the list unsorted."""
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
    """insort_right places the item after existing equal values."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] == (5, 5)
    assert collection[2] is item


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 4, [4]),
        ([3], 3, [3, 3]),
    ],
)
def test_insort_right_boundaries(collection, item, expected):
    """insort_right handles empty lists, single elements and out-of-range values."""
    insort_right(collection, item)
    assert collection == expected


def test_insort_right_with_bounds():
    """insort_right honours lo/hi, which may leave the list unsorted."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_and_right_agree_on_unique_values():
    """For values not already present, both insort variants produce the same list."""
    left = [0, 5, 7, 10, 15]
    right = [0, 5, 7, 10, 15]
    insort_left(left, 8)
    insort_right(right, 8)
    assert left == right == [0, 5, 7, 8, 10, 15]


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
        ([0, 5, 7, 10, 15], -100, -1),
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert binary_search(collection, item) == expected


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([], 1, -1),
        ([42], 42, 0),
        ([42], 41, -1),
        ([42], 43, -1),
        ([1, 2], 2, 1),
        ([1, 2], 1, 0),
    ],
)
def test_binary_search_edge_cases(collection, item, expected):
    """binary_search handles empty, single-element and two-element collections."""
    assert binary_search(collection, item) == expected


def test_binary_search_raises_on_unsorted_collection():
    """binary_search raises ValueError when the collection is not ascending."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search([3, 1, 2], 1)


def test_binary_search_accepts_duplicates():
    """binary_search returns some valid index when duplicates are present."""
    collection = [1, 2, 2, 2, 3]
    result = binary_search(collection, 2)
    assert collection[result] == 2


def test_binary_search_does_not_mutate_input():
    """binary_search must not modify the collection it searches."""
    collection = [0, 5, 7, 10, 15]
    snapshot = list(collection)
    binary_search(collection, 7)
    assert collection == snapshot


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


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([], 1, -1),
        ([42], 42, 0),
        ([42], 41, -1),
        ([1, 2, 2, 2, 3], 2, 1),
    ],
)
def test_binary_search_std_lib_edge_cases(collection, item, expected):
    """binary_search_std_lib handles empty lists, singletons and duplicates."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_raises_on_unsorted_collection():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sequence convertible to a list."""
    assert binary_search_std_lib(range(10), 4) == 4
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
        ([7, 7, 7], 7, [0, 1, 2]),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index where the item occurs."""
    assert binary_search_with_duplicates(collection, item) == expected


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([], 1, []),
        ([5], 5, [0]),
        ([5], 4, []),
        ([5], 6, []),
    ],
)
def test_binary_search_with_duplicates_edge_cases(collection, item, expected):
    """binary_search_with_duplicates handles empty and single-element lists."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_raises_on_unsorted_collection():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value is always a list, even when nothing is found."""
    assert isinstance(binary_search_with_duplicates([1, 2, 3], 99), list)


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
        ([0, 5, 7, 10, 15], 7, 0, 4, 2),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds items within an explicit index range."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 6, -1),
        ([], 1, -1),
        ([42], 42, 0),
    ],
)
def test_binary_search_by_recursion_default_bounds(collection, item, expected):
    """binary_search_by_recursion defaults right to len(collection) - 1."""
    assert binary_search_by_recursion(collection, item) == expected


def test_binary_search_by_recursion_empty_range():
    """binary_search_by_recursion returns -1 when right < left."""
    assert binary_search_by_recursion([0, 5, 7, 10, 15], 5, 3, 2) == -1


def test_binary_search_by_recursion_raises_on_unsorted_collection():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_by_recursion([9, 8, 7], 8)


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


@pytest.mark.parametrize(
    ("collection", "item", "expected"),
    [
        ([], 1, -1),
        ([42], 42, 0),
        ([42], 43, -1),
        (list(range(100)), 99, 99),
        (list(range(100)), 0, 0),
        (list(range(100)), 100, -1),
    ],
)
def test_exponential_search_edge_cases(collection, item, expected):
    """exponential_search handles empty, singleton and large collections."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_raises_on_unsorted_collection():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        exponential_search([3, 2, 1], 2)


def test_exponential_search_agrees_with_binary_search():
    """exponential_search and binary_search agree on every element of a list."""
    collection = list(range(0, 200, 3))
    for index, value in enumerate(collection):
        assert exponential_search(collection, value) == index
        assert binary_search(collection, value) == index


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
def test_all_searches_find_every_element(search):
    """Every search implementation locates each element of a sorted list."""
    collection = [0, 5, 7, 10, 15, 20, 25, 30]
    for index, value in enumerate(collection):
        assert search(collection, value) == index


@pytest.mark.parametrize(
    "search",
    [
        binary_search,
        binary_search_std_lib,
        binary_search_by_recursion,
        exponential_search,
    ],
)
def test_all_searches_return_minus_one_for_missing(search):
    """Every search implementation returns -1 for a value not in the list."""
    collection = [0, 5, 7, 10, 15]
    for missing in (1, 6, 14, 16, 1000):
        assert search(collection, missing) == -1


def test_searches_tuple_contains_all_implementations():
    """The module-level `searches` tuple lists the four search functions."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )
    assert all(callable(search) for search in searches)
