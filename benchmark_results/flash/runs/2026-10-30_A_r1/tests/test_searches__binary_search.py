# AUTO-SDET: Test generation FAILED after 3 retries
# Last error: 

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
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by file path so the test file is self-contained
# and does not depend on the package layout / sys.path configuration.
# The source may live next to the test, one level down in `searches/`, or be
# reached from the current working directory, so several candidates are tried.
# ---------------------------------------------------------------------------
_TEST_DIR = Path(__file__).resolve().parent

_CANDIDATES = [
    _TEST_DIR / "binary_search.py",
    _TEST_DIR / "searches" / "binary_search.py",
    Path.cwd() / "binary_search.py",
    Path.cwd() / "searches" / "binary_search.py",
    Path(r"D:\Documents\Study\Projects\algorithms-bench\searches\binary_search.py"),
]


def _find_source():
    """Return the first existing candidate path for the module under test."""
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _load_module():
    """Load the target module from its resolved absolute path."""
    source = _find_source()
    if source is None:  # pragma: no cover - defensive
        pytest.skip(
            "binary_search.py not found; looked in: "
            + ", ".join(str(c) for c in _CANDIDATES),
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(
        "binary_search_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bs = _load_module()


# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------
SORTED = [0, 5, 7, 10, 15]
DUPLICATES = [1, 2, 2, 2, 3]


@pytest.fixture
def sorted_collection():
    """A fresh ascending-sorted list for each test that mutates it."""
    return [0, 5, 7, 10, 15]


# ===========================================================================
# bisect_left
# ===========================================================================
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),      # first element
        ([0, 5, 7, 10, 15], 6, 2),      # between elements
        ([0, 5, 7, 10, 15], 20, 5),     # greater than all
        ([0, 5, 7, 10, 15], -1, 0),     # smaller than all
        ([0, 5, 7, 10, 15], 15, 4),     # last element
        ([], 3, 0),                     # empty collection
        ([1, 2, 2, 2, 3], 2, 1),        # duplicates -> first occurrence
    ],
)
def test_bisect_left_happy_path(collection, item, expected):
    """bisect_left returns the leftmost insertion index for various inputs."""
    assert bs.bisect_left(collection, item) == expected


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),   # hi restricts the search window
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),   # lo restricts the search window
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),    # item below window -> returns lo
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),  # empty window
    ],
)
def test_bisect_left_with_bounds(collection, item, lo, hi, expected):
    """bisect_left honours the lo/hi window arguments."""
    assert bs.bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_matches_stdlib():
    """bisect_left must agree with the standard library implementation."""
    collection = list(range(0, 100, 3))
    for item in range(-5, 105):
        assert bs.bisect_left(collection, item) == bisect.bisect_left(collection, item)


# ===========================================================================
# bisect_right
# ===========================================================================
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 1),      # first element -> after it
        ([0, 5, 7, 10, 15], 15, 5),     # last element -> end
        ([0, 5, 7, 10, 15], 6, 2),      # between elements
        ([0, 5, 7, 10, 15], 20, 5),     # greater than all
        ([0, 5, 7, 10, 15], -1, 0),     # smaller than all
        ([], 3, 0),                     # empty collection
        ([1, 2, 2, 2, 3], 2, 4),        # duplicates -> after last occurrence
    ],
)
def test_bisect_right_happy_path(collection, item, expected):
    """bisect_right returns the rightmost insertion index for various inputs."""
    assert bs.bisect_right(collection, item) == expected


@pytest.mark.parametrize(
    "collection, item, lo, hi, expected",
    [
        ([0, 5, 7, 10, 15], 15, 1, 3, 3),
        ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ([0, 5, 7, 10, 15], 0, 2, 4, 2),
        ([0, 5, 7, 10, 15], 100, 0, 0, 0),
    ],
)
def test_bisect_right_with_bounds(collection, item, lo, hi, expected):
    """bisect_right honours the lo/hi window arguments."""
    assert bs.bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_matches_stdlib():
    """bisect_right must agree with the standard library implementation."""
    collection = list(range(0, 100, 3))
    for item in range(-5, 105):
        assert bs.bisect_right(collection, item) == bisect.bisect_right(
            collection, item
        )


def test_bisect_left_and_right_differ_on_duplicates():
    """For a duplicated value, left index < right index."""
    assert bs.bisect_left(DUPLICATES, 2) < bs.bisect_right(DUPLICATES, 2)


# ===========================================================================
# insort_left
# ===========================================================================

def test_insort_left_inserts_in_middle(sorted_collection):
    """insort_left inserts a new value at the correct sorted position."""
    bs.insort_left(sorted_collection, 6)
    assert sorted_collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_returns_none(sorted_collection):
    """insort_left mutates in place and returns None."""
    assert bs.insort_left(sorted_collection, 6) is None


@pytest.mark.parametrize(
    "item, expected",
    [
        (-5, [-5, 0, 5, 7, 10, 15]),   # before all
        (20, [0, 5, 7, 10, 15, 20]),   # after all
        (0, [0, 0, 5, 7, 10, 15]),     # duplicate of first
        (15, [0, 5, 7, 10, 15, 15]),   # duplicate of last
    ],
)
def test_insort_left_boundaries(item, expected):
    """insort_left handles insertion at both ends and duplicate values."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_left(collection, item)
    assert collection == expected


def test_insort_left_places_before_equal_elements():
    """insort_left inserts *before* existing equal elements (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    bs.insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]


def test_insort_left_with_bounds():
    """insort_left respects the lo/hi window (may break global ordering)."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_empty_collection():
    """insort_left on an empty list simply appends the item."""
    collection = []
    bs.insort_left(collection, 42)
    assert collection == [42]


# ===========================================================================
# insort_right
# ===========================================================================

def test_insort_right_inserts_in_middle(sorted_collection):
    """insort_right inserts a new value at the correct sorted position."""
    bs.insort_right(sorted_collection, 6)
    assert sorted_collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_returns_none(sorted_collection):
    """insort_right mutates in place and returns None."""
    assert bs.insort_right(sorted_collection, 6) is None


@pytest.mark.parametrize(
    "item, expected",
    [
        (-5, [-5, 0, 5, 7, 10, 15]),
        (20, [0, 5, 7, 10, 15, 20]),
        (0, [0, 0, 5, 7, 10, 15]),
        (15, [0, 5, 7, 10, 15, 15]),
    ],
)
def test_insort_right_boundaries(item, expected):
    """insort_right handles insertion at both ends and duplicate values."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_right(collection, item)
    assert collection == expected


def test_insort_right_places_after_equal_elements():
    """insort_right inserts *after* existing equal elements (identity check)."""
    collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
    item = (5, 5)
    bs.insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[2]


def test_insort_right_with_bounds():
    """insort_right respects the lo/hi window (may break global ordering)."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_empty_collection():
    """insort_right on an empty list simply appends the item."""
    collection = []
    bs.insort_right(collection, 42)
    assert collection == [42]


def test_insort_left_and_right_agree_on_unique_values():
    """For values not already present, both insort variants produce the same list."""
    left = [0, 5, 7, 10, 15]
    right = [0, 5, 7, 10, 15]
    bs.insort_left(left, 6)
    bs.insort_right(right, 6)
    assert left == right


# ===========================================================================
# binary_search
# ===========================================================================
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
        ([42], 42, 0),          # single element, found
        ([42], 7, -1),          # single element, not found
        ([], 1, -1),            # empty collection
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert bs.binary_search(collection, item) == expected


def test_binary_search_raises_on_unsorted_collection():
    """binary_search raises ValueError when the collection is not ascending."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search([5, 3, 1], 3)


def test_binary_search_raises_on_descending_collection():
    """A strictly descending collection must also raise ValueError."""
    with pytest.raises(ValueError):
        bs.binary_search([15, 10, 7, 5, 0], 10)


def test_binary_search_accepts_duplicates():
    """binary_search returns *some* valid index when duplicates exist."""
    collection = [1, 2, 2, 2, 3]
    result = bs.binary_search(collection, 2)
    assert collection[result] == 2


def test_binary_search_does_not_mutate_input():
    """binary_search must not modify the collection it searches."""
    collection = [0, 5, 7, 10, 15]
    snapshot = list(collection)
    bs.binary_search(collection, 7)
    assert collection == snapshot


# ===========================================================================
# binary_search_std_lib
# ===========================================================================
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
        ([42], 1, -1),
        ([], 1, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert bs.binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_raises_on_unsorted_collection():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_std_lib([3, 1, 2], 1)


def test_binary_search_std_lib_returns_first_duplicate_index():
    """With duplicates, the stdlib variant returns the leftmost matching index."""
    assert bs.binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1


def test_binary_search_std_lib_agrees_with_binary_search():
    """Both search implementations agree on a sorted collection of unique values."""
    collection = list(range(0, 200, 7))
    for item in range(-10, 210):
        assert bs.binary_search_std_lib(collection, item) == bs.binary_search(
            collection, item
        )


# ===========================================================================
# binary_search_with_duplicates
# ===========================================================================
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, [0]),
        ([0, 5, 7, 10, 15], 15, [4]),
        ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
        ([1, 2, 2, 2, 3], 4, []),
        ([1, 2, 2, 2, 3], 0, []),
        ([7, 7, 7], 7, [0, 1, 2]),   # all elements identical
        ([], 1, []),                 # empty collection
        ([42], 42, [0]),             # single element, found
        ([42], 1, []),               # single element, not found
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index where the item occurs."""
    assert bs.binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_raises_on_unsorted_collection():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """The return value is always a list, even when nothing is found."""
    assert isinstance(bs.binary_search_with_duplicates([1, 2, 3], 99), list)


def test_binary_search_with_duplicates_indexes_are_valid():
    """Every returned index must actually point at the searched item."""
    collection = [0, 0, 1, 1, 1, 2, 3, 3]
    for item in set(collection):
        indexes = bs.binary_search_with_duplicates(collection, item)
        assert indexes == sorted(indexes)
        assert all(collection[i] == item for i in indexes)
        assert len(indexes) == collection.count(item)


# ===========================================================================
# binary_search_by_recursion
# ===========================================================================
@pytest.mark.parametrize(
    "collection, item, left, right, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0, 4, 0),
        ([0, 5, 7, 10, 15], 15, 0, 4, 4),
        ([0, 5, 7, 10, 15], 5, 0, 4, 1),
        ([0, 5, 7, 10, 15], 6, 0, 4, -1),
        ([0, 5, 7, 10, 15], 7, 2, 4, 2),    # restricted window
        ([0, 5, 7, 10, 15], 0, 2, 4, -1),   # item outside window
        ([42], 42, 0, 0, 0),                # single element
        ([], 1, 0, -1, -1),                 # empty collection
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the index within the given window."""
    assert bs.binary_search_by_recursion(collection, item, left, right) == expected


def test_binary_search_by_recursion_default_bounds():
    """Omitting left/right uses the whole collection (right defaults to -1)."""
    assert bs.binary_search_by_recursion([0, 5, 7, 10, 15], 15) == 4
    assert bs.binary_search_by_recursion([0, 5, 7, 10, 15], 6) == -1


def test_binary_search_by_recursion_raises_on_unsorted_collection():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_by_recursion([9, 4, 2], 4)


def test_binary_search_by_recursion_empty_window_returns_minus_one():
    """When right < left the search space is empty and -1 is returned."""
    assert bs.binary_search_by_recursion([0, 5, 7], 5, 2, 1) == -1


def test_binary_search_by_recursion_agrees_with_binary_search():
    """Recursive and iterative implementations agree on unique sorted values."""
    collection = list(range(0, 100, 4))
    for item in range(-5, 105):
        assert bs.binary_search_by_recursion(
            collection, item, 0, len(collection) - 1
        ) == bs.binary_search(collection, item)


# ===========================================================================
# exponential_search
# ===========================================================================
@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([0, 5, 7, 10, 15], -1, -1),
        ([0, 5, 7, 10, 15], 100, -1),
        ([42], 42, 0),      # single element, found
        ([42], 1, -1),      # single element, not found
        ([], 1, -1),        # empty collection
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert bs.exponential_search(collection, item) == expected


def test_exponential_search_raises_on_unsorted_collection():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.exponential_search([10, 5, 1], 5)


def test_exponential_search_large_collection():
    """exponential_search works on a large collection (exercises bound doubling)."""
    collection = list(range(0, 10_000))
    assert bs.exponential_search(collection, 0) == 0
    assert bs.exponential_search(collection, 9_999) == 9_999
    assert bs.exponential_search(collection, 5_000) == 5_000
    assert bs.exponential_search(collection, 10_000) == -1


def test_exponential_search_agrees_with_binary_search():
    """Exponential and plain binary search agree on unique sorted values."""
    collection = list(range(0, 128))
    for item in range(-2, 130):
        assert bs.exponential_search(collection, item) == bs.binary_search(
            collection, item
        )


# ===========================================================================
# Module-level `searches` tuple
# ===========================================================================
def test_searches_tuple_contains_expected_functions():
    """The module exposes a `searches` tuple of the four search callables."""
    assert isinstance(bs.searches, tuple)
    assert len(bs.searches) == 4
    assert bs.searches == (
        bs.binary_search_std_lib,
        bs.binary_search,
        bs.exponential_search,
        bs.binary_search_by_recursion,
    )


@pytest.mark.parametrize("search", list(bs.searches), ids=lambda f: f.__name__)
def test_every_search_in_tuple_finds_element(search):
    """Every function in `searches` locates an existing element correctly."""
    assert search([0, 5, 7, 10, 15], 10) == 3


@pytest.mark.parametrize("search", list(bs.searches), ids=lambda f: f.__name__)
def test_every_search_in_tuple_returns_minus_one_when_missing(search):
    """Every function in `searches` returns -1 for a missing element."""
    assert search([0, 5, 7, 10, 15], 6) == -1


# ===========================================================================
# Cross-implementation consistency
# ===========================================================================
def test_all_search_implementations_agree():
    """All four search implementations produce identical results on sorted data."""
    collection = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    for item in range(-1, 92):
        results = {search(collection, item) for search in bs.searches}
        assert len(results) == 1, f"Disagreement for item {item}: {results}"


def test_bisect_helpers_are_consistent_with_insort():
    """insort_left/insort_right insert exactly at bisect_left/bisect_right."""
    for item in [-1, 0, 6, 15, 20]:
        left = [0, 5, 7, 10, 15]
        right = [0, 5, 7, 10, 15]
        expected_left = bs.bisect_left(left, item)
        expected_right = bs.bisect_right(right, item)
        bs.insort_left(left, item)
        bs.insort_right(right, item)
        assert left[expected_left] == item
        assert right[expected_right] == item
