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
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], -1, 0),
        ([1, 2, 2, 2, 3], 2, 1),
        ([], 5, 0),
        ([5], 5, 0),
        ([5], 4, 0),
        ([5], 6, 1),
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
        ([0, 5, 7, 10, 15], 100, 0, 2, 2),
        ([0, 5, 7, 10, 15], 6, 3, 3, 3),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi slice bounds, with hi=-1 meaning len()."""
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
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -5, 0),
        ([1, 2, 2, 2, 3], 2, 4),
        ([], 5, 0),
        ([5], 5, 1),
        ([5], 4, 0),
        ([5], 6, 1),
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
        ([0, 5, 7, 10, 15], 100, 0, 2, 2),
        ([0, 5, 7, 10, 15], 6, 3, 3, 3),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi slice bounds, with hi=-1 meaning len()."""
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
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 3, [3]),
        ([1, 2, 3], 2, [1, 2, 2, 3]),
    ],
)
def test_insort_left_happy_path(collection, item, expected):
    """insort_left inserts the item in place, keeping the list sorted."""
    insort_left(collection, item)
    assert collection == expected


def test_insort_left_inserts_before_equal_values():
    """insort_left places the new item before existing equal values (identity)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Build the item at runtime: a literal ``(5, 5)`` would be folded into the
    # very same constant object stored in the collection literal above, which
    # would make the ``is``/``is not`` identity checks meaningless.
    item = tuple([5, 5])
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is item
    assert collection[2] is not item


def test_insort_left_with_bounds():
    """insort_left respects lo/hi bounds, inserting at the bounded position."""
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
    ("collection", "item", "expected"),
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 3, [3]),
        ([1, 2, 3], 2, [1, 2, 2, 3]),
    ],
)
def test_insort_right_happy_path(collection, item, expected):
    """insort_right inserts the item in place, keeping the list sorted."""
    insort_right(collection, item)
    assert collection == expected


def test_insort_right_inserts_after_equal_values():
    """insort_right places the new item after existing equal values (identity)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # See the note in ``test_insort_left_inserts_before_equal_values``: the
    # item must be created at runtime to be a genuinely distinct object.
    item = tuple([5, 5])
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is not item
    assert collection[2] is item


def test_insort_right_with_bounds():
    """insort_right respects lo/hi bounds, inserting at the bounded position."""
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
        ([42], 42, 0),
        ([42], 41, -1),
        ([], 1, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
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


def test_binary_search_does_not_mutate_collection():
    """binary_search must not modify the input collection."""
    collection = [0, 5, 7, 10, 15]
    binary_search(collection, 7)
    assert collection == [0, 5, 7, 10, 15]


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
        ([0, 5, 7, 10, 15], -10, -1),
        ([0, 5, 7, 10, 15], 1000, -1),
        ([7], 7, 0),
        ([7], 8, -1),
        ([], 0, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_raises_on_unsorted_collection():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sequence, e.g. a range object."""
    assert binary_search_std_lib(range(1000), 500) == 500
    assert binary_search_std_lib(range(1000), 1000) == -1


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
        ([], 1, []),
        ([9], 9, [0]),
        ([9], 8, []),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns all indexes of the item."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_raises_on_unsorted_collection():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value is always a list, even when nothing is found."""
    result = binary_search_with_duplicates([1, 3, 5], 4)
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
        ([0, 5, 7, 10, 15], 0, 0, -1, 0),
        ([0, 5, 7, 10, 15], 15, 0, -1, 4),
        ([0, 5, 7, 10, 15], 6, 0, -1, -1),
        ([0, 5, 7, 10, 15], 0, 2, 4, -1),
        ([], 1, 0, -1, -1),
        ([42], 42, 0, -1, 0),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the index or returns -1."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


def test_binary_search_by_recursion_raises_on_unsorted_collection():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        binary_search_by_recursion([3, 2, 1], 2, 0, 2)


def test_binary_search_by_recursion_empty_range_returns_minus_one():
    """When right < left the search space is empty and -1 is returned."""
    assert binary_search_by_recursion([1, 2, 3], 2, 2, 1) == -1


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
        ([7], 7, 0),
        ([], 1, -1),
        (list(range(100)), 99, 99),
        (list(range(100)), 0, 0),
        (list(range(100)), 100, -1),
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_target_below_first_element_recurses_forever():
    """Targets smaller than the first element overflow the recursion limit.

    ``exponential_search`` calls ``binary_search_by_recursion`` with
    ``right = min(bound, len(collection) - 1)``.  When the probe at index 0 is
    larger than the target, the helper recurses with ``right = -1``, which it
    re-interprets as ``len(collection) - 1``, so the very same call repeats
    until ``RecursionError`` is raised instead of returning -1.  This test
    documents the current behaviour of the implementation under test.
    """
    with pytest.raises(RecursionError):
        exponential_search([0, 5, 7, 10, 15], -1)
    with pytest.raises(RecursionError):
        exponential_search([7], 6)


def test_exponential_search_raises_on_unsorted_collection():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        exponential_search([10, 9, 8], 9)


def test_exponential_search_agrees_with_binary_search():
    """exponential_search and binary_search agree on every element of a list."""
    collection = list(range(0, 200, 3))
    for index, value in enumerate(collection):
        assert exponential_search(collection, value) == index
        assert binary_search(collection, value) == index


# ---------------------------------------------------------------------------
# module-level `searches` tuple
# ---------------------------------------------------------------------------


def test_searches_tuple_contains_all_search_functions():
    """The `searches` tuple exposes the four search implementations."""
    assert searches == (
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    )


@pytest.mark.parametrize("search", searches)
def test_all_searches_find_existing_item(search):
    """Every registered search function locates an existing item."""
    assert search([0, 5, 7, 10, 15], 10) == 3


@pytest.mark.parametrize("search", searches)
def test_all_searches_return_minus_one_for_missing_item(search):
    """Every registered search function returns -1 for a missing item."""
    assert search([0, 5, 7, 10, 15], 6) == -1


@pytest.mark.parametrize("search", searches)
def test_all_searches_raise_on_unsorted_input(search):
    """Every registered search function rejects unsorted input."""
    with pytest.raises(ValueError):
        search([5, 4, 3, 2, 1], 3)
