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

import bisect
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by absolute path (no package assumptions).
# The path is resolved relative to this test file / the working directory so
# the suite works regardless of the operating system it runs on.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

_POSSIBLE_PATHS = (
    _HERE / "searches" / "binary_search.py",
    _HERE / "binary_search.py",
    Path.cwd() / "searches" / "binary_search.py",
    Path.cwd() / "binary_search.py",
    Path("/workspace/searches/binary_search.py"),
)


def _load_module():
    """Load the target module from a discovered path (fall back to a normal import)."""
    for candidate in _POSSIBLE_PATHS:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "binary_search_under_test", candidate
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module

    # Last resort: import it as a module from the project root.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import searches.binary_search as module  # type: ignore[import-not-found]

    return module


bs = _load_module()

SORTED = [0, 5, 7, 10, 15]
DUPLICATES = [1, 2, 2, 2, 3]


# ---------------------------------------------------------------------------
# bisect_left
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], 20, 5),
        ([0, 5, 7, 10, 15], -1, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([1, 2, 2, 2, 3], 2, 1),
        ([], 5, 0),
        ([7], 7, 0),
        ([7], 8, 1),
    ],
)
def test_bisect_left_matches_stdlib(collection, item, expected):
    """bisect_left must agree with the stdlib implementation and expected values."""
    assert bs.bisect_left(collection, item) == expected
    assert bs.bisect_left(collection, item) == bisect.bisect_left(collection, item)


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
        ([0, 5, 7, 10, 15], 100, 0, 2, 2),
        ([0, 5, 7, 10, 15], 5, 3, 3, 3),
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left must honour the lo/hi slice bounds like the stdlib."""
    assert bs.bisect_left(collection, item, lo, hi) == expected
    assert bs.bisect_left(collection, item, lo, hi) == bisect.bisect_left(
        collection, item, lo, hi
    )


def test_bisect_left_empty_range_returns_lo():
    """When lo == hi the search range is empty and lo is returned unchanged."""
    assert bs.bisect_left(SORTED, 7, 3, 3) == 3


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 1),
        ([0, 5, 7, 10, 15], 15, 5),
        ([0, 5, 7, 10, 15], 6, 2),
        ([0, 5, 7, 10, 15], -5, 0),
        ([0, 5, 7, 10, 15], 100, 5),
        ([1, 2, 2, 2, 3], 2, 4),
        ([], 5, 0),
        ([7], 7, 1),
    ],
)
def test_bisect_right_matches_stdlib(collection, item, expected):
    """bisect_right must agree with the stdlib implementation and expected values."""
    assert bs.bisect_right(collection, item) == expected
    assert bs.bisect_right(collection, item) == bisect.bisect_right(collection, item)


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
        ([0, 5, 7, 10, 15], 5, 3, 3, 3),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right must honour the lo/hi slice bounds like the stdlib."""
    assert bs.bisect_right(collection, item, lo, hi) == expected
    assert bs.bisect_right(collection, item, lo, hi) == bisect.bisect_right(
        collection, item, lo, hi
    )


def test_bisect_right_empty_range_returns_lo():
    """When lo == hi the search range is empty and lo is returned unchanged."""
    assert bs.bisect_right(SORTED, 7, 2, 2) == 2


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 3, [3]),
        ([1, 2, 2, 3], 2, [1, 2, 2, 2, 3]),
    ],
)
def test_insort_left_inserts_in_place(collection, item, expected):
    """insort_left must insert the item in place keeping the list sorted."""
    bs.insort_left(collection, item)
    assert collection == expected


def test_insort_left_inserts_before_equal_elements():
    """insort_left must place the new item before existing equal elements."""
    # Build the collection at runtime so its tuples are not shared with the
    # (constant-folded) tuple literal used for `item`.
    collection = [(value, value) for value in (0, 5, 7, 10, 15)]
    item = (5, 5)
    bs.insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is item
    assert collection[2] is not item


def test_insort_left_with_bounds():
    """insort_left must respect lo/hi bounds (inserting at the bound index)."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_returns_none():
    """insort_left mutates in place and returns None."""
    collection = [1, 3]
    assert bs.insort_left(collection, 2) is None


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 6, [0, 5, 6, 7, 10, 15]),
        ([0, 5, 7, 10, 15], 20, [0, 5, 7, 10, 15, 20]),
        ([0, 5, 7, 10, 15], -1, [-1, 0, 5, 7, 10, 15]),
        ([], 3, [3]),
        ([1, 2, 2, 3], 2, [1, 2, 2, 2, 3]),
    ],
)
def test_insort_right_inserts_in_place(collection, item, expected):
    """insort_right must insert the item in place keeping the list sorted."""
    bs.insort_right(collection, item)
    assert collection == expected


def test_insort_right_inserts_after_equal_elements():
    """insort_right must place the new item after existing equal elements."""
    # Build the collection at runtime so its tuples are not shared with the
    # (constant-folded) tuple literal used for `item`.
    collection = [(value, value) for value in (0, 5, 7, 10, 15)]
    item = (5, 5)
    bs.insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert collection[1] is not item
    assert collection[2] is item


def test_insort_right_with_bounds():
    """insort_right must respect lo/hi bounds (inserting at the bound index)."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_returns_none():
    """insort_right mutates in place and returns None."""
    collection = [1, 3]
    assert bs.insort_right(collection, 2) is None


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
def test_binary_search(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert bs.binary_search(collection, item) == expected


def test_binary_search_unsorted_raises():
    """binary_search must raise ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="ascending order"):
        bs.binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_valid_index():
    """With duplicates, binary_search returns an index holding the item."""
    collection = [1, 2, 2, 2, 3]
    result = bs.binary_search(collection, 2)
    assert collection[result] == 2


def test_binary_search_does_not_mutate_input():
    """binary_search must not modify the input collection."""
    collection = [0, 5, 7, 10, 15]
    snapshot = list(collection)
    bs.binary_search(collection, 7)
    assert collection == snapshot


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
        ([42], 42, 0),
        ([42], 43, -1),
        ([], 1, -1),
    ],
)
def test_binary_search_std_lib(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert bs.binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib must raise ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="ascending order"):
        bs.binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib must accept any sequence, e.g. a range object."""
    assert bs.binary_search_std_lib(range(1000), 500) == 500
    assert bs.binary_search_std_lib(range(1000), 1000) == -1


def test_binary_search_std_lib_duplicates_returns_first_index():
    """With duplicates, binary_search_std_lib returns the first matching index."""
    assert bs.binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1


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
        ([], 1, []),
        ([7], 7, [0]),
        ([7], 8, []),
    ],
)
def test_binary_search_with_duplicates(collection, item, expected):
    """binary_search_with_duplicates returns all indexes of the item."""
    assert bs.binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates must raise ValueError when unsorted."""
    with pytest.raises(ValueError, match="ascending order"):
        bs.binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value must always be a list, even when nothing is found."""
    result = bs.binary_search_with_duplicates(SORTED, 999)
    assert isinstance(result, list)
    assert result == []


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
        ([0, 5, 7, 10, 15], 7, 2, 4, 2),
        ([0, 5, 7, 10, 15], 0, 1, 4, -1),
        ([42], 42, 0, 0, 0),
        ([], 1, 0, -1, -1),
    ],
)
def test_binary_search_by_recursion(collection, item, left, right, expected):
    """binary_search_by_recursion returns the index of the item or -1."""
    assert bs.binary_search_by_recursion(collection, item, left, right) == expected


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 6, -1),
        ([], 1, -1),
    ],
)
def test_binary_search_by_recursion_default_bounds(collection, item, expected):
    """With right=-1 the default bounds must cover the whole collection."""
    assert bs.binary_search_by_recursion(collection, item) == expected


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion must raise ValueError when unsorted."""
    with pytest.raises(ValueError, match="ascending order"):
        bs.binary_search_by_recursion([9, 8, 7], 8, 0, 2)


def test_binary_search_by_recursion_empty_range_returns_minus_one():
    """When right < left the search range is empty and -1 is returned."""
    assert bs.binary_search_by_recursion(SORTED, 5, 3, 2) == -1


def test_binary_search_by_recursion_single_element_below_minimum_recurses():
    """Documented limitation for a single-element collection.

    When the only element is greater than the target, the recursive call is
    made with right == midpoint - 1 == -1, which the helper interprets as
    "use the default bounds" (len(collection) - 1 == 0), so it recurses until
    the interpreter's recursion limit is reached instead of returning -1.
    """
    with pytest.raises(RecursionError):
        bs.binary_search_by_recursion([42], 41, 0, 0)


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
        ([0, 5, 7, 10, 15], 100, -1),
        ([42], 42, 0),
        ([], 1, -1),
        (list(range(100)), 99, 99),
        (list(range(100)), 0, 0),
        (list(range(100)), 50, 50),
        (list(range(100)), 100, -1),
    ],
)
def test_exponential_search(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert bs.exponential_search(collection, item) == expected


def test_exponential_search_unsorted_raises():
    """exponential_search must raise ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="ascending order"):
        bs.exponential_search([3, 2, 1], 2)


def test_exponential_search_agrees_with_binary_search():
    """exponential_search must agree with binary_search on a large sorted list."""
    collection = list(range(0, 2000, 3))
    for item in (0, 3, 999, 1998, 1999):
        assert bs.exponential_search(collection, item) == bs.binary_search(
            collection, item
        )


def test_exponential_search_below_minimum_hits_recursion_limit():
    """Documented limitation: searching for an item smaller than every element

    makes `binary_search_by_recursion` see right == -1, which it interprets as
    "use the default bounds", so the helper recurses until the interpreter's
    recursion limit is reached instead of returning -1.
    """
    with pytest.raises(RecursionError):
        bs.exponential_search([0, 5, 7, 10, 15], -1)
    with pytest.raises(RecursionError):
        bs.exponential_search([42], 41)


# ---------------------------------------------------------------------------
# Module-level `searches` tuple
# ---------------------------------------------------------------------------
def test_searches_tuple_contents():
    """The module exposes a `searches` tuple of the four search callables."""
    assert isinstance(bs.searches, tuple)
    assert bs.searches == (
        bs.binary_search_std_lib,
        bs.binary_search,
        bs.exponential_search,
        bs.binary_search_by_recursion,
    )


@pytest.mark.parametrize("search", list(bs.searches))
def test_all_search_functions_find_existing_item(search):
    """Every registered search function must locate an existing element."""
    assert search([0, 5, 7, 10, 15], 10) == 3


@pytest.mark.parametrize("search", list(bs.searches))
def test_all_search_functions_report_missing_item(search):
    """Every registered search function must return -1 for a missing element."""
    assert search([0, 5, 7, 10, 15], 6) == -1
