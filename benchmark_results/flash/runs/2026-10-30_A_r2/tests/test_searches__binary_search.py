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
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], -1, 0),
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
        ([0, 5, 7, 10, 15], 100, 0, 2, 2),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left respects the lo/hi window, with hi=-1 meaning len(collection)."""
    assert bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bisect_left([], 42) == 0


def test_bisect_left_duplicates_returns_first_index():
    """bisect_left returns the leftmost position among equal elements."""
    assert bisect_left([1, 2, 2, 2, 3], 2) == 1


def test_bisect_left_matches_stdlib():
    """bisect_left agrees with the standard library implementation."""
    collection = list(range(0, 100, 3))
    for item in range(-5, 105):
        assert bisect_left(collection, item) == stdlib_bisect.bisect_left(
            collection, item
        )


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -10, 0),
        ([0, 5, 7, 10, 15], 100, 5),
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
        ([0, 5, 7, 10, 15], 0, 0, 1, 1),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right respects the lo/hi window."""
    assert bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bisect_right([], 1) == 0


def test_bisect_right_duplicates_returns_after_last():
    """bisect_right returns the index just past the last equal element."""
    assert bisect_right([1, 2, 2, 2, 3], 2) == 4


def test_bisect_right_matches_stdlib():
    """bisect_right agrees with the standard library implementation."""
    collection = list(range(0, 100, 3))
    for item in range(-5, 105):
        assert bisect_right(collection, item) == stdlib_bisect.bisect_right(
            collection, item
        )


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------


def test_insort_left_inserts_in_middle():
    """insort_left inserts an item into the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the new item before existing equal items."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Built at runtime so it is a *distinct* object from the (5, 5) already in
    # the collection (identical tuple literals are folded into one constant).
    item = tuple([5, 5])
    insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is item
    assert collection[2] is not item


def test_insort_left_appends_at_end():
    """insort_left appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_left_into_empty_collection():
    """insort_left on an empty list produces a single-element list."""
    collection = []
    insort_left(collection, 3)
    assert collection == [3]


def test_insort_left_with_bounds():
    """insort_left honours the lo/hi window when locating the insertion point."""
    collection = [0, 5, 7, 10, 15]
    insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------


def test_insort_right_inserts_in_middle():
    """insort_right inserts an item into the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert insort_right(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_inserts_after_duplicates():
    """insort_right places the new item after existing equal items."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    # Built at runtime so it is a *distinct* object from the (5, 5) already in
    # the collection (identical tuple literals are folded into one constant).
    item = tuple([5, 5])
    insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is not item
    assert collection[2] is item


def test_insort_right_appends_at_end():
    """insort_right appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_right_into_empty_collection():
    """insort_right on an empty list produces a single-element list."""
    collection = []
    insort_right(collection, 3)
    assert collection == [3]


def test_insort_right_with_bounds():
    """insort_right honours the lo/hi window when locating the insertion point."""
    collection = [0, 5, 7, 10, 15]
    insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


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
    ],
)
def test_binary_search_finds_present_items(collection, item, expected):
    """binary_search returns the index of an item that is present."""
    assert binary_search(collection, item) == expected


@pytest.mark.parametrize(
    "collection, item",
    [
        ([0, 5, 7, 10, 15], 6),
        ([0, 5, 7, 10, 15], -1),
        ([0, 5, 7, 10, 15], 100),
        ([], 1),
        ([1], 2),
    ],
)
def test_binary_search_returns_minus_one_when_absent(collection, item):
    """binary_search returns -1 when the item is not present."""
    assert binary_search(collection, item) == -1


def test_binary_search_single_element_found():
    """binary_search finds the only element of a one-element collection."""
    assert binary_search([42], 42) == 0


def test_binary_search_raises_on_unsorted_collection():
    """binary_search raises ValueError for a descending collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search([5, 4, 3, 2, 1], 3)


def test_binary_search_raises_on_partially_unsorted_collection():
    """binary_search raises ValueError when only one adjacent pair is out of order."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search([1, 2, 3, 5, 4], 4)


def test_binary_search_accepts_duplicates():
    """binary_search returns some valid index when duplicates exist."""
    collection = [1, 2, 2, 2, 3]
    result = binary_search(collection, 2)
    assert collection[result] == 2


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
        ([0, 5, 7, 10, 15], -3, -1),
        ([0, 5, 7, 10, 15], 99, -1),
    ],
)
def test_binary_search_std_lib(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1."""
    assert binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib returns -1 for an empty collection."""
    assert binary_search_std_lib([], 0) == -1


def test_binary_search_std_lib_raises_on_unsorted_collection():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_std_lib([3, 1, 2], 1)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sequence, e.g. a range object."""
    assert binary_search_std_lib(range(10), 7) == 7


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
        ([2, 2, 2], 2, [0, 1, 2]),
    ],
)
def test_binary_search_with_duplicates(collection, item, expected):
    """binary_search_with_duplicates returns every index of the target."""
    assert binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates returns an empty list for an empty input."""
    assert binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_raises_on_unsorted_collection():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_with_duplicates([2, 1, 2], 2)


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
        ([0, 5, 7, 10, 15], 10, 0, 2, -1),
    ],
)
def test_binary_search_by_recursion_with_explicit_bounds(
    collection, item, left, right, expected
):
    """binary_search_by_recursion searches within the given [left, right] window."""
    assert binary_search_by_recursion(collection, item, left, right) == expected


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], 20, -1),
        ([], 1, -1),
        ([9], 9, 0),
    ],
)
def test_binary_search_by_recursion_default_bounds(collection, item, expected):
    """binary_search_by_recursion defaults right to len(collection) - 1."""
    assert binary_search_by_recursion(collection, item) == expected


def test_binary_search_by_recursion_item_below_first_element():
    """Characterises a limitation of the current implementation.

    When the midpoint is 0 and the item is smaller than ``collection[0]``, the
    recursive call receives ``right=-1``; the function then re-interprets that
    sentinel as "use the default bound", so the search window never shrinks and
    ``RecursionError`` is raised instead of returning -1.
    """
    with pytest.raises(RecursionError):
        binary_search_by_recursion([9], 8)


def test_binary_search_by_recursion_raises_on_unsorted_collection():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        binary_search_by_recursion([3, 2, 1], 2)


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
    ],
)
def test_exponential_search(collection, item, expected):
    """exponential_search returns the index of the item or -1."""
    assert exponential_search(collection, item) == expected


def test_exponential_search_empty_collection():
    """exponential_search returns -1 for an empty collection."""
    assert exponential_search([], 5) == -1


def test_exponential_search_single_element():
    """exponential_search handles a one-element collection."""
    assert exponential_search([7], 7) == 0
    assert exponential_search([7], 8) == -1


def test_exponential_search_large_collection():
    """exponential_search finds items in a large collection."""
    collection = list(range(0, 2000, 2))
    assert exponential_search(collection, 1000) == 500
    assert exponential_search(collection, 1001) == -1


def test_exponential_search_item_below_first_element():
    """Characterises a limitation inherited from binary_search_by_recursion.

    Searching for a value smaller than the first element of the collection
    recurses until ``RecursionError`` is raised instead of returning -1.
    """
    with pytest.raises(RecursionError):
        exponential_search([0, 5, 7, 10, 15], -1)


def test_exponential_search_raises_on_unsorted_collection():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="ascending order"):
        exponential_search([4, 3, 2, 1], 3)


# ---------------------------------------------------------------------------
# Cross-implementation consistency & module attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("search", searches)
def test_all_search_functions_agree(search):
    """Every registered search function returns the same index for a sorted list."""
    collection = [0, 5, 7, 10, 15]
    for index, value in enumerate(collection):
        assert search(collection, value) == index
    assert search(collection, 6) == -1


def test_searches_tuple_contents():
    """The module-level `searches` tuple contains the expected callables."""
    assert isinstance(searches, tuple)
    assert set(searches) == {
        binary_search_std_lib,
        binary_search,
        exponential_search,
        binary_search_by_recursion,
    }
