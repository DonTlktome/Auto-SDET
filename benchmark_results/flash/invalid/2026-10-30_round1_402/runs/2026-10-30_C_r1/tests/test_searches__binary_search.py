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
# Locate the module under test regardless of cwd / platform.
# ---------------------------------------------------------------------------
def _find_source() -> Path:
    """Return the path to binary_search.py, searching common locations."""
    here = Path(__file__).resolve()
    names = ("binary_search.py", Path("searches") / "binary_search.py")

    candidates: list[Path] = []
    for base in (here.parent, *here.parent.parents):
        for name in names:
            candidates.append(base / name)

    cwd = Path.cwd()
    for name in names:
        candidates.append(cwd / name)

    candidates.append(Path("/workspace/searches/binary_search.py"))
    candidates.append(Path("/workspace/binary_search.py"))

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - unreadable path on this platform
            continue

    raise ModuleNotFoundError(
        "Could not locate binary_search.py. Searched:\n"
        + "\n".join(str(c) for c in candidates)
    )


_SOURCE_PATH = _find_source()


def _load_module():
    """Load the target module from its resolved path."""
    spec = importlib.util.spec_from_file_location("binary_search_under_test", _SOURCE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ModuleNotFoundError(f"Unable to load module from {_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bs = _load_module()

SORTED = [0, 5, 7, 10, 15]
DUPLICATES = [1, 2, 2, 2, 3]


def _point(x: int, y: int) -> tuple[int, int]:
    """Return a *fresh* ``(x, y)`` tuple object.

    The compiler merges equal tuple literals inside one code object into a
    single constant, so two ``(5, 5)`` literals written in the same test would
    be the *same* object and any ``is`` / ``is not`` assertion would be
    vacuous.  Building the tuples at runtime keeps object identity meaningful.
    """
    return (x, y)


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
    assert bs.bisect_left(collection, item) == expected


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
    """bisect_left honours the lo/hi slice bounds."""
    assert bs.bisect_left(collection, item, lo, hi) == expected


def test_bisect_left_empty_collection():
    """bisect_left on an empty collection returns 0."""
    assert bs.bisect_left([], 42) == 0


def test_bisect_left_duplicates_returns_first_occurrence():
    """bisect_left returns the leftmost index of a duplicated value."""
    assert bs.bisect_left(DUPLICATES, 2) == 1


@pytest.mark.parametrize(
    "collection, item",
    [
        ([0, 5, 7, 10, 15], 0),
        ([0, 5, 7, 10, 15], 6),
        ([0, 5, 7, 10, 15], 20),
        ([1, 2, 2, 2, 3], 2),
        ([], 3),
    ],
)
def test_bisect_left_matches_stdlib(collection, item):
    """bisect_left agrees with the standard library implementation."""
    assert bs.bisect_left(collection, item) == bisect.bisect_left(collection, item)


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
    ],
)
def test_bisect_right_happy_path(collection, item, expected):
    """bisect_right returns the first index whose value is > item."""
    assert bs.bisect_right(collection, item) == expected


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
    assert bs.bisect_right(collection, item, lo, hi) == expected


def test_bisect_right_empty_collection():
    """bisect_right on an empty collection returns 0."""
    assert bs.bisect_right([], 42) == 0


def test_bisect_right_duplicates_returns_after_last_occurrence():
    """bisect_right returns the index just past the last duplicate."""
    assert bs.bisect_right(DUPLICATES, 2) == 4


@pytest.mark.parametrize(
    "collection, item",
    [
        ([0, 5, 7, 10, 15], 0),
        ([0, 5, 7, 10, 15], 6),
        ([0, 5, 7, 10, 15], 20),
        ([1, 2, 2, 2, 3], 2),
        ([], 3),
    ],
)
def test_bisect_right_matches_stdlib(collection, item):
    """bisect_right agrees with the standard library implementation."""
    assert bs.bisect_right(collection, item) == bisect.bisect_right(collection, item)


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------
def test_insort_left_inserts_in_middle():
    """insort_left inserts a value into the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert bs.insort_left(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_left_inserts_before_duplicates():
    """insort_left places the new item before existing equal items."""
    # Fresh tuple objects are used on purpose: tuple literals of equal value
    # are folded into one constant by the compiler, which would make the
    # identity assertions below compare an object with itself.
    collection = [
        _point(0, 0),
        _point(5, 5),
        _point(7, 7),
        _point(10, 10),
        _point(15, 15),
    ]
    item = _point(5, 5)
    bs.insort_left(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is collection[1]
    assert item is not collection[2]


def test_insort_left_appends_at_end():
    """insort_left appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_left(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_left_with_bounds():
    """insort_left respects the lo/hi bounds when locating the insertion point."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_left(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_left_empty_collection():
    """insort_left on an empty list inserts the single element."""
    collection = []
    bs.insort_left(collection, 3)
    assert collection == [3]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------
def test_insort_right_inserts_in_middle():
    """insort_right inserts a value into the correct sorted position."""
    collection = [0, 5, 7, 10, 15]
    assert bs.insort_right(collection, 6) is None
    assert collection == [0, 5, 6, 7, 10, 15]


def test_insort_right_inserts_after_duplicates():
    """insort_right places the new item after existing equal items."""
    # See test_insort_left_inserts_before_duplicates for why _point() is used.
    collection = [
        _point(0, 0),
        _point(5, 5),
        _point(7, 7),
        _point(10, 10),
        _point(15, 15),
    ]
    item = _point(5, 5)
    bs.insort_right(collection, item)
    assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
    assert item is not collection[1]
    assert item is collection[2]


def test_insort_right_appends_at_end():
    """insort_right appends when the item is larger than every element."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_right(collection, 20)
    assert collection == [0, 5, 7, 10, 15, 20]


def test_insort_right_with_bounds():
    """insort_right respects the lo/hi bounds when locating the insertion point."""
    collection = [0, 5, 7, 10, 15]
    bs.insort_right(collection, 15, 1, 3)
    assert collection == [0, 5, 7, 15, 10, 15]


def test_insort_right_empty_collection():
    """insort_right on an empty list inserts the single element."""
    collection = []
    bs.insort_right(collection, 3)
    assert collection == [3]


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
        ([0, 5, 7, 10, 15], -1, -1),
        ([0, 5, 7, 10, 15], 100, -1),
    ],
)
def test_binary_search_happy_path(collection, item, expected):
    """binary_search returns the index of the item or -1 when absent."""
    assert bs.binary_search(collection, item) == expected


def test_binary_search_empty_collection():
    """binary_search on an empty collection returns -1."""
    assert bs.binary_search([], 1) == -1


def test_binary_search_single_element_found():
    """binary_search finds the only element of a one-element collection."""
    assert bs.binary_search([42], 42) == 0


def test_binary_search_single_element_not_found():
    """binary_search returns -1 for a one-element collection without the item."""
    assert bs.binary_search([42], 7) == -1


def test_binary_search_unsorted_raises():
    """binary_search raises ValueError when the collection is not sorted."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search([3, 1, 2], 1)


def test_binary_search_duplicates_returns_some_valid_index():
    """binary_search returns an index that actually holds the searched value."""
    collection = [1, 2, 2, 2, 3]
    index = bs.binary_search(collection, 2)
    assert collection[index] == 2


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
        ([0, 5, 7, 10, 15], -10, -1),
        ([0, 5, 7, 10, 15], 1000, -1),
    ],
)
def test_binary_search_std_lib_happy_path(collection, item, expected):
    """binary_search_std_lib returns the index of the item or -1 when absent."""
    assert bs.binary_search_std_lib(collection, item) == expected


def test_binary_search_std_lib_empty_collection():
    """binary_search_std_lib on an empty collection returns -1."""
    assert bs.binary_search_std_lib([], 5) == -1


def test_binary_search_std_lib_unsorted_raises():
    """binary_search_std_lib raises ValueError for an unsorted collection."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_std_lib([5, 4, 3], 4)


def test_binary_search_std_lib_duplicates_returns_first_index():
    """binary_search_std_lib returns the leftmost index of a duplicated value."""
    assert bs.binary_search_std_lib(DUPLICATES, 2) == 1


def test_binary_search_std_lib_accepts_range():
    """binary_search_std_lib works with any sorted sequence, e.g. a range."""
    assert bs.binary_search_std_lib(range(1000), 500) == 500


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
        ([1, 2, 2, 2, 3], 1, [0]),
        ([1, 2, 2, 2, 3], 3, [4]),
        ([2, 2, 2], 2, [0, 1, 2]),
    ],
)
def test_binary_search_with_duplicates_happy_path(collection, item, expected):
    """binary_search_with_duplicates returns every index of the target."""
    assert bs.binary_search_with_duplicates(collection, item) == expected


def test_binary_search_with_duplicates_empty_collection():
    """binary_search_with_duplicates returns an empty list for an empty input."""
    assert bs.binary_search_with_duplicates([], 1) == []


def test_binary_search_with_duplicates_unsorted_raises():
    """binary_search_with_duplicates raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_with_duplicates([2, 1, 2], 2)


def test_binary_search_with_duplicates_returns_list_type():
    """binary_search_with_duplicates always returns a list instance."""
    assert isinstance(bs.binary_search_with_duplicates([1, 2, 3], 2), list)
    assert isinstance(bs.binary_search_with_duplicates([1, 2, 3], 9), list)


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
        ([0, 5, 7, 10, 15], 10, 0, 4, 3),
    ],
)
def test_binary_search_by_recursion_happy_path(collection, item, left, right, expected):
    """binary_search_by_recursion finds the index or returns -1."""
    assert bs.binary_search_by_recursion(collection, item, left, right) == expected


@pytest.mark.parametrize(
    "collection, item, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 6, -1),
    ],
)
def test_binary_search_by_recursion_default_bounds(collection, item, expected):
    """binary_search_by_recursion computes default bounds when right < 0."""
    assert bs.binary_search_by_recursion(collection, item) == expected


def test_binary_search_by_recursion_empty_collection():
    """binary_search_by_recursion returns -1 for an empty collection."""
    assert bs.binary_search_by_recursion([], 1) == -1


def test_binary_search_by_recursion_right_less_than_left():
    """binary_search_by_recursion returns -1 when right < left."""
    assert bs.binary_search_by_recursion([0, 5, 7], 5, 2, 1) == -1


def test_binary_search_by_recursion_unsorted_raises():
    """binary_search_by_recursion raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.binary_search_by_recursion([9, 8, 7], 8)


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
        ([0, 5, 7, 10, 15], 7, 2),
        ([0, 5, 7, 10, 15], 10, 3),
        # NOTE: items below the smallest element are covered separately by
        # test_search_below_minimum_raises_recursion_error (known limitation).
        ([0, 5, 7, 10, 15], 99, -1),
    ],
)
def test_exponential_search_happy_path(collection, item, expected):
    """exponential_search returns the index of the item or -1 when absent."""
    assert bs.exponential_search(collection, item) == expected


def test_exponential_search_empty_collection():
    """exponential_search returns -1 for an empty collection."""
    assert bs.exponential_search([], 1) == -1


def test_exponential_search_single_element():
    """exponential_search handles a one-element collection."""
    assert bs.exponential_search([7], 7) == 0
    assert bs.exponential_search([7], 8) == -1


def test_exponential_search_unsorted_raises():
    """exponential_search raises ValueError for unsorted input."""
    with pytest.raises(ValueError, match="sorted in ascending order"):
        bs.exponential_search([3, 2, 1], 2)


def test_exponential_search_large_collection():
    """exponential_search finds items in a large sorted collection."""
    collection = list(range(0, 2000, 2))
    assert bs.exponential_search(collection, 1000) == 500
    assert bs.exponential_search(collection, 1001) == -1


# ---------------------------------------------------------------------------
# Cross-implementation consistency & module attributes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("item", [0, 5, 7, 10, 15, 6, 100])
def test_all_search_implementations_agree(item):
    """All search implementations return the same index for the same input.

    Items below the smallest element are excluded: they hit the limitation
    documented in test_search_below_minimum_raises_recursion_error.
    """
    results = {
        bs.binary_search(SORTED, item),
        bs.binary_search_std_lib(SORTED, item),
        bs.binary_search_by_recursion(SORTED, item),
        bs.exponential_search(SORTED, item),
    }
    assert len(results) == 1


@pytest.mark.parametrize(
    "search, item",
    [
        (bs.binary_search_by_recursion, -1),
        (bs.exponential_search, -1),
        (bs.exponential_search, -3),
    ],
)
def test_search_below_minimum_raises_recursion_error(search, item):
    """Known limitation of the reference implementation.

    ``binary_search_by_recursion`` overloads ``right == -1`` as the "use the
    default bounds" sentinel, so whenever the search narrows down past index 0
    (which always happens for items smaller than every element) the bounds are
    reset to the full range and the function recurses until Python raises
    ``RecursionError``.  ``exponential_search`` delegates to it whenever the
    exponential probe never moves past the first element, so it inherits the
    same behaviour.

    This test pins the current behaviour so that a future fix of the module is
    noticed immediately instead of silently changing semantics.
    """
    with pytest.raises(RecursionError):
        search(SORTED, item)


def test_searches_tuple_contains_all_implementations():
    """The module-level `searches` tuple lists the four search functions."""
    assert bs.searches == (
        bs.binary_search_std_lib,
        bs.binary_search,
        bs.exponential_search,
        bs.binary_search_by_recursion,
    )


def test_searches_are_callable():
    """Every entry in `searches` is callable and returns an int."""
    for search in bs.searches:
        result = search(SORTED, 10)
        assert isinstance(result, int)
        assert result == 3
