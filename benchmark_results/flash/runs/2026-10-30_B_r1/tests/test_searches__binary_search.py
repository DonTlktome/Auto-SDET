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
import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The source file lives at .../searches/binary_search.py and is not guaranteed
# to be importable as a package member, so we load it directly by path.
# ---------------------------------------------------------------------------
_SOURCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "binary_search.py"
)

if not os.path.exists(_SOURCE_PATH):  # pragma: no cover - fallback for odd layouts
    _SOURCE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "searches",
        "binary_search.py",
    )

_spec = importlib.util.spec_from_file_location("binary_search_under_test", _SOURCE_PATH)
binary_search_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = binary_search_module
_spec.loader.exec_module(binary_search_module)

bisect_left = binary_search_module.bisect_left
bisect_right = binary_search_module.bisect_right
insort_left = binary_search_module.insort_left
insort_right = binary_search_module.insort_right
binary_search = binary_search_module.binary_search
binary_search_std_lib = binary_search_module.binary_search_std_lib
binary_search_with_duplicates = binary_search_module.binary_search_with_duplicates
binary_search_by_recursion = binary_search_module.binary_search_by_recursion
exponential_search = binary_search_module.exponential_search
searches = binary_search_module.searches


SORTED = [0, 5, 7, 10, 15]


# ---------------------------------------------------------------------------
# bisect_left
# ---------------------------------------------------------------------------
class TestBisectLeft:
    """Tests for the custom `bisect_left` implementation."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 6, 2),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 20, 5),
            ([0, 5, 7, 10, 15], -1, 0),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """bisect_left returns the first index whose value is >= item."""
        assert bisect_left(collection, item) == expected

    def test_empty_collection(self):
        """bisect_left on an empty list returns 0."""
        assert bisect_left([], 42) == 0

    def test_single_element_equal(self):
        """bisect_left on a one-element list equal to the item returns 0."""
        assert bisect_left([7], 7) == 0

    def test_single_element_greater(self):
        """bisect_left on a one-element list greater than the item returns 0."""
        assert bisect_left([7], 3) == 0

    def test_single_element_smaller(self):
        """bisect_left on a one-element list smaller than the item returns 1."""
        assert bisect_left([7], 9) == 1

    def test_with_duplicates_returns_first_occurrence(self):
        """bisect_left must return the index of the FIRST duplicate."""
        assert bisect_left([1, 2, 2, 2, 3], 2) == 1

    def test_explicit_lo_hi_bounds(self):
        """bisect_left honours explicit lo/hi bounds."""
        assert bisect_left(SORTED, 15, 1, 3) == 3
        assert bisect_left(SORTED, 6, 2) == 2

    def test_lo_equals_hi(self):
        """When lo == hi the search range is empty and lo is returned."""
        assert bisect_left(SORTED, 7, 2, 2) == 2

    def test_negative_hi_means_end_of_list(self):
        """A negative hi is interpreted as len(sorted_collection)."""
        assert bisect_left(SORTED, 20, 0, -1) == len(SORTED)

    def test_matches_stdlib(self):
        """The custom bisect_left agrees with the stdlib implementation."""
        collection = list(range(0, 100, 3))
        for item in range(-2, 105):
            assert bisect_left(collection, item) == stdlib_bisect.bisect_left(
                collection, item
            )


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------
class TestBisectRight:
    """Tests for the custom `bisect_right` implementation."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 1),
            ([0, 5, 7, 10, 15], 5, 2),
            ([0, 5, 7, 10, 15], 6, 2),
            ([0, 5, 7, 10, 15], 15, 5),
            ([0, 5, 7, 10, 15], 20, 5),
            ([0, 5, 7, 10, 15], -1, 0),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """bisect_right returns the first index whose value is > item."""
        assert bisect_right(collection, item) == expected

    def test_empty_collection(self):
        """bisect_right on an empty list returns 0."""
        assert bisect_right([], 42) == 0

    def test_single_element_equal(self):
        """bisect_right on a one-element list equal to the item returns 1."""
        assert bisect_right([7], 7) == 1

    def test_with_duplicates_returns_after_last_occurrence(self):
        """bisect_right must return the index just past the LAST duplicate."""
        assert bisect_right([1, 2, 2, 2, 3], 2) == 4

    def test_explicit_lo_hi_bounds(self):
        """bisect_right honours explicit lo/hi bounds."""
        assert bisect_right(SORTED, 15, 1, 3) == 3
        assert bisect_right(SORTED, 6, 2) == 2

    def test_lo_equals_hi(self):
        """When lo == hi the search range is empty and lo is returned."""
        assert bisect_right(SORTED, 7, 2, 2) == 2

    def test_matches_stdlib(self):
        """The custom bisect_right agrees with the stdlib implementation."""
        collection = list(range(0, 100, 3))
        for item in range(-2, 105):
            assert bisect_right(collection, item) == stdlib_bisect.bisect_right(
                collection, item
            )


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------
class TestInsortLeft:
    """Tests for `insort_left`."""

    def test_inserts_in_middle(self):
        """insort_left inserts a value into the correct sorted position."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 6)
        assert collection == [0, 5, 6, 7, 10, 15]

    def test_inserts_at_beginning(self):
        """insort_left inserts a value smaller than all elements at index 0."""
        collection = [0, 5, 7]
        insort_left(collection, -3)
        assert collection == [-3, 0, 5, 7]

    def test_inserts_at_end(self):
        """insort_left appends a value larger than all elements."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 20)
        assert collection == [0, 5, 7, 10, 15, 20]

    def test_inserts_into_empty_list(self):
        """insort_left works on an empty list."""
        collection = []
        insort_left(collection, 1)
        assert collection == [1]

    def test_duplicate_goes_before_existing(self):
        """insort_left places an equal item BEFORE existing equal items."""
        collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
        item = (5, 5)
        insort_left(collection, item)
        assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
        assert collection[1] == item
        assert collection[2] == item

    def test_returns_none(self):
        """insort_left mutates in place and returns None."""
        collection = [1, 2, 3]
        assert insort_left(collection, 2) is None

    def test_with_lo_hi_bounds(self):
        """insort_left respects lo/hi bounds (may leave list unsorted)."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 15, 1, 3)
        assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------
class TestInsortRight:
    """Tests for `insort_right`."""

    def test_inserts_in_middle(self):
        """insort_right inserts a value into the correct sorted position."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 6)
        assert collection == [0, 5, 6, 7, 10, 15]

    def test_inserts_at_beginning(self):
        """insort_right inserts a value smaller than all elements at index 0."""
        collection = [0, 5, 7]
        insort_right(collection, -3)
        assert collection == [-3, 0, 5, 7]

    def test_inserts_at_end(self):
        """insort_right appends a value larger than all elements."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 20)
        assert collection == [0, 5, 7, 10, 15, 20]

    def test_inserts_into_empty_list(self):
        """insort_right works on an empty list."""
        collection = []
        insort_right(collection, 1)
        assert collection == [1]

    def test_duplicate_goes_after_existing(self):
        """insort_right places an equal item AFTER existing equal items."""
        collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
        item = (5, 5)
        insort_right(collection, item)
        assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
        assert collection[1] == item
        assert collection[2] == item

    def test_returns_none(self):
        """insort_right mutates in place and returns None."""
        collection = [1, 2, 3]
        assert insort_right(collection, 2) is None

    def test_with_lo_hi_bounds(self):
        """insort_right respects lo/hi bounds (may leave list unsorted)."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 15, 1, 3)
        assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------
class TestBinarySearch:
    """Tests for the iterative `binary_search`."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 7, 2),
            ([0, 5, 7, 10, 15], 10, 3),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 6, -1),
            ([0, 5, 7, 10, 15], -1, -1),
            ([0, 5, 7, 10, 15], 100, -1),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """binary_search returns the index of the item or -1 when absent."""
        assert binary_search(collection, item) == expected

    def test_empty_collection(self):
        """binary_search on an empty list returns -1."""
        assert binary_search([], 1) == -1

    def test_single_element_found(self):
        """binary_search finds the only element of a one-element list."""
        assert binary_search([42], 42) == 0

    def test_single_element_not_found(self):
        """binary_search returns -1 for a one-element list without the item."""
        assert binary_search([42], 7) == -1

    def test_unsorted_collection_raises(self):
        """binary_search raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError, match="ascending order"):
            binary_search([3, 1, 2], 1)

    def test_duplicates_returns_some_valid_index(self):
        """With duplicates, any index pointing at the item is acceptable."""
        collection = [1, 2, 2, 2, 3]
        result = binary_search(collection, 2)
        assert collection[result] == 2

    def test_large_collection(self):
        """binary_search works on a large sorted collection."""
        collection = list(range(10_000))
        assert binary_search(collection, 9_999) == 9_999
        assert binary_search(collection, 10_000) == -1


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------
class TestBinarySearchStdLib:
    """Tests for `binary_search_std_lib`."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 6, -1),
            ([0, 5, 7, 10, 15], -5, -1),
            ([0, 5, 7, 10, 15], 99, -1),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """binary_search_std_lib returns the index of the item or -1."""
        assert binary_search_std_lib(collection, item) == expected

    def test_empty_collection(self):
        """binary_search_std_lib on an empty list returns -1."""
        assert binary_search_std_lib([], 1) == -1

    def test_single_element_found(self):
        """binary_search_std_lib finds the only element of a one-element list."""
        assert binary_search_std_lib([42], 42) == 0

    def test_single_element_not_found(self):
        """binary_search_std_lib returns -1 when the item is absent."""
        assert binary_search_std_lib([42], 7) == -1

    def test_unsorted_collection_raises(self):
        """binary_search_std_lib raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError, match="ascending order"):
            binary_search_std_lib([3, 1, 2], 1)

    def test_duplicates_returns_first_index(self):
        """binary_search_std_lib returns the first index of a duplicated item."""
        assert binary_search_std_lib([1, 2, 2, 2, 3], 2) == 1

    def test_agrees_with_binary_search(self):
        """binary_search_std_lib agrees with binary_search on unique values."""
        collection = list(range(0, 200, 7))
        for item in range(-5, 205):
            assert binary_search_std_lib(collection, item) == binary_search(
                collection, item
            )


# ---------------------------------------------------------------------------
# binary_search_with_duplicates
# ---------------------------------------------------------------------------
class TestBinarySearchWithDuplicates:
    """Tests for `binary_search_with_duplicates`."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, [0]),
            ([0, 5, 7, 10, 15], 15, [4]),
            ([0, 5, 7, 10, 15], 7, [2]),
            ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
            ([1, 2, 2, 2, 3], 4, []),
            ([1, 2, 2, 2, 3], 0, []),
            ([2, 2, 2], 2, [0, 1, 2]),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """Returns every index where the item occurs, or [] when absent."""
        assert binary_search_with_duplicates(collection, item) == expected

    def test_empty_collection(self):
        """An empty collection yields an empty list."""
        assert binary_search_with_duplicates([], 1) == []

    def test_single_element_found(self):
        """A one-element collection returns [0] when the item matches."""
        assert binary_search_with_duplicates([5], 5) == [0]

    def test_single_element_not_found(self):
        """A one-element collection returns [] when the item does not match."""
        assert binary_search_with_duplicates([5], 6) == []

    def test_unsorted_collection_raises(self):
        """Raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError, match="ascending order"):
            binary_search_with_duplicates([3, 1, 2], 1)

    def test_returns_list_type(self):
        """The return value is always a list, even when empty."""
        assert isinstance(binary_search_with_duplicates([1, 2, 3], 9), list)

    def test_all_duplicates_large(self):
        """All indexes are returned for a large list of identical values."""
        collection = [7] * 50
        assert binary_search_with_duplicates(collection, 7) == list(range(50))


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------
class TestBinarySearchByRecursion:
    """Tests for the recursive `binary_search_by_recursion`."""

    @pytest.mark.parametrize(
        "collection, item, left, right, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0, 4, 0),
            ([0, 5, 7, 10, 15], 15, 0, 4, 4),
            ([0, 5, 7, 10, 15], 5, 0, 4, 1),
            ([0, 5, 7, 10, 15], 6, 0, 4, -1),
        ],
    )
    def test_happy_path_with_explicit_bounds(self, collection, item, left, right, expected):
        """Recursive search with explicit left/right bounds."""
        assert binary_search_by_recursion(collection, item, left, right) == expected

    @pytest.mark.parametrize(
        "item, expected",
        [(0, 0), (5, 1), (7, 2), (10, 3), (15, 4), (6, -1), (20, -1)],
    )
    def test_default_bounds(self, item, expected):
        """A negative `right` defaults to len(collection) - 1."""
        assert binary_search_by_recursion(SORTED, item) == expected

    def test_empty_collection(self):
        """An empty collection returns -1."""
        assert binary_search_by_recursion([], 1) == -1

    def test_single_element_found(self):
        """A one-element collection finds its only element."""
        assert binary_search_by_recursion([42], 42) == 0

    def test_single_element_not_found(self):
        """A one-element collection returns -1 when the item is absent."""
        assert binary_search_by_recursion([42], 100, 0, 0) == -1

    def test_right_less_than_left_returns_minus_one(self):
        """An empty search window (right < left) returns -1."""
        assert binary_search_by_recursion(SORTED, 7, 3, 2) == -1

    def test_unsorted_collection_raises(self):
        """Raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError, match="ascending order"):
            binary_search_by_recursion([3, 1, 2], 1)

    def test_agrees_with_iterative(self):
        """Recursive and iterative searches agree on unique values."""
        collection = list(range(0, 100, 4))
        for item in range(0, 105):
            assert binary_search_by_recursion(collection, item) == binary_search(
                collection, item
            )


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------
class TestExponentialSearch:
    """Tests for `exponential_search`."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 7, 2),
            ([0, 5, 7, 10, 15], 10, 3),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 6, -1),
            ([0, 5, 7, 10, 15], 100, -1),
        ],
    )
    def test_happy_path(self, collection, item, expected):
        """exponential_search returns the index of the item or -1."""
        assert exponential_search(collection, item) == expected

    def test_empty_collection(self):
        """An empty collection returns -1."""
        assert exponential_search([], 1) == -1

    def test_single_element_found(self):
        """A one-element collection finds its only element."""
        assert exponential_search([42], 42) == 0

    def test_single_element_not_found(self):
        """A one-element collection returns -1 when the item is absent."""
        assert exponential_search([42], 100) == -1

    def test_unsorted_collection_raises(self):
        """Raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError, match="ascending order"):
            exponential_search([3, 1, 2], 1)

    def test_large_collection(self):
        """exponential_search works on a large sorted collection."""
        collection = list(range(0, 5000, 2))
        assert exponential_search(collection, 4998) == 2499
        assert exponential_search(collection, 4999) == -1

    def test_agrees_with_binary_search(self):
        """exponential_search agrees with binary_search on unique values."""
        collection = list(range(0, 300, 3))
        for item in range(0, 305):
            assert exponential_search(collection, item) == binary_search(
                collection, item
            )


# ---------------------------------------------------------------------------
# Module-level `searches` tuple
# ---------------------------------------------------------------------------
class TestSearchesTuple:
    """Tests for the module-level `searches` registry."""

    def test_is_tuple(self):
        """`searches` is a tuple of callables."""
        assert isinstance(searches, tuple)
        assert all(callable(fn) for fn in searches)

    def test_contains_expected_functions(self):
        """`searches` contains the four documented search functions."""
        assert set(searches) == {
            binary_search_std_lib,
            binary_search,
            exponential_search,
            binary_search_by_recursion,
        }

    @pytest.mark.parametrize("search", searches, ids=lambda fn: fn.__name__)
    def test_every_search_finds_target(self, search):
        """Every registered search finds 10 at index 3 in the sample list."""
        assert search([0, 5, 7, 10, 15], 10) == 3

    @pytest.mark.parametrize("search", searches, ids=lambda fn: fn.__name__)
    def test_every_search_reports_missing(self, search):
        """Every registered search returns -1 for a missing item."""
        assert search([0, 5, 7, 10, 15], 6) == -1


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------
class TestCrossImplementationConsistency:
    """All search implementations must agree on the same inputs."""

    @pytest.mark.parametrize("item", [0, 1, 5, 6, 7, 10, 14, 15, 16, 100])
    def test_all_agree_on_unique_collection(self, item):
        """All implementations return the same index for unique values."""
        results = {search(SORTED, item) for search in searches}
        assert len(results) == 1

    def test_insort_left_and_right_keep_sorted_order(self):
        """Inserting with insort_left/insort_right keeps the list sorted."""
        left_list = [1, 3, 5, 7]
        right_list = [1, 3, 5, 7]
        for value in (0, 2, 4, 6, 8, 3):
            insort_left(left_list, value)
            insort_right(right_list, value)
        assert left_list == sorted(left_list)
        assert right_list == sorted(right_list)
        assert left_list == right_list
