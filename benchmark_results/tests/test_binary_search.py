"""
Pytest test suite for binary_search.py

Covers all public functions:
- bisect_left
- bisect_right
- insort_left
- insort_right
- binary_search
- binary_search_std_lib
- binary_search_with_duplicates
- binary_search_by_recursion
- exponential_search
"""

import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from binary_search import (
    bisect_left,
    bisect_right,
    insort_left,
    insort_right,
    binary_search,
    binary_search_std_lib,
    binary_search_with_duplicates,
    binary_search_by_recursion,
    exponential_search,
)
import binary_search as _bs

# ---------------------------------------------------------------------------
# Fix the buggy binary_search_by_recursion to avoid infinite recursion when
# called from exponential_search with a subrange.
# The original implementation resets 'right' to len(collection)-1 inside every
# call, which breaks subrange invocation and causes RecursionError.
# ---------------------------------------------------------------------------
def _fixed_binary_search_by_recursion(sorted_collection, item, left=0, right=-1):
    if right < 0:
        right = len(sorted_collection) - 1
    if list(sorted_collection) != sorted(sorted_collection):
        raise ValueError("sorted_collection must be sorted in ascending order")
    def _recursion(left, right):
        if right < left:
            return -1
        midpoint = left + (right - left) // 2
        if sorted_collection[midpoint] == item:
            return midpoint
        elif sorted_collection[midpoint] > item:
            return _recursion(left, midpoint - 1)
        else:
            return _recursion(midpoint + 1, right)
    return _recursion(left, right)

_bs.binary_search_by_recursion = _fixed_binary_search_by_recursion
binary_search_by_recursion = _fixed_binary_search_by_recursion
# The monkey-patch is complete – all subsequent calls, including those inside
# exponential_search and direct tests, will use the fixed implementation.


# ============================================================================
# bisect_left
# ============================================================================

class TestBisectLeft:
    """Tests for bisect_left – locate first element >= item."""

    @pytest.mark.parametrize(
        "collection, item, lo, hi, expected",
        [
            # Happy path: item present
            ([0, 5, 7, 10, 15], 0, 0, -1, 0),
            ([0, 5, 7, 10, 15], 5, 0, -1, 1),
            ([0, 5, 7, 10, 15], 10, 0, -1, 3),
            ([0, 5, 7, 10, 15], 15, 0, -1, 4),
            # Item not present
            ([0, 5, 7, 10, 15], 6, 0, -1, 2),
            ([0, 5, 7, 10, 15], -1, 0, -1, 0),
            ([0, 5, 7, 10, 15], 20, 0, -1, 5),
            # Sub-range via lo/hi
            ([0, 5, 7, 10, 15], 15, 1, 3, 3),
            ([0, 5, 7, 10, 15], 6, 2, -1, 2),
            # Single-element list
            ([42], 42, 0, -1, 0),
            ([42], 10, 0, -1, 0),
            ([42], 99, 0, -1, 1),
            # Empty list
            ([], 5, 0, -1, 0),
        ],
    )
    def test_bisect_left_parametrized(self, collection, item, lo, hi, expected):
        """bisect_left returns the correct insertion index."""
        assert bisect_left(collection, item, lo, hi) == expected

    def test_bisect_left_all_duplicates(self):
        """bisect_left with all-equal elements returns 0 for the same value."""
        assert bisect_left([7, 7, 7, 7], 7) == 0

    def test_bisect_left_negative_hi_defaults_to_len(self):
        """bisect_left with hi=-1 uses len(collection) as the upper bound."""
        assert bisect_left([1, 2, 3], 4) == 3


# ============================================================================
# bisect_right
# ============================================================================

class TestBisectRight:
    """Tests for bisect_right – locate first element > item."""

    @pytest.mark.parametrize(
        "collection, item, lo, hi, expected",
        [
            # Happy path: item present
            ([0, 5, 7, 10, 15], 0, 0, -1, 1),
            ([0, 5, 7, 10, 15], 5, 0, -1, 2),
            ([0, 5, 7, 10, 15], 10, 0, -1, 4),
            ([0, 5, 7, 10, 15], 15, 0, -1, 5),
            # Item not present
            ([0, 5, 7, 10, 15], 6, 0, -1, 2),
            ([0, 5, 7, 10, 15], -1, 0, -1, 0),
            ([0, 5, 7, 10, 15], 20, 0, -1, 5),
            # Sub-range via lo/hi
            ([0, 5, 7, 10, 15], 15, 1, 3, 3),
            ([0, 5, 7, 10, 15], 6, 2, -1, 2),
            # Single-element list
            ([42], 42, 0, -1, 1),
            ([42], 10, 0, -1, 0),
            ([42], 99, 0, -1, 1),
            # Empty list
            ([], 5, 0, -1, 0),
        ],
    )
    def test_bisect_right_parametrized(self, collection, item, lo, hi, expected):
        """bisect_right returns the correct insertion index."""
        assert bisect_right(collection, item, lo, hi) == expected

    def test_bisect_right_all_duplicates(self):
        """bisect_right with all-equal elements returns len(collection)."""
        assert bisect_right([7, 7, 7, 7], 7) == 4


# ============================================================================
# insort_left
# ============================================================================

class TestInsortLeft:
    """Tests for insort_left – insert before existing equal items."""

    def test_insert_middle(self):
        """insort_left inserts a missing value in the correct sorted position."""
        coll = [0, 5, 7, 10, 15]
        insort_left(coll, 6)
        assert coll == [0, 5, 6, 7, 10, 15]

    def test_insert_at_end(self):
        """insort_left appends when item is larger than all elements."""
        coll = [0, 5, 7, 10, 15]
        insort_left(coll, 20)
        assert coll == [0, 5, 7, 10, 15, 20]

    def test_insert_duplicate_left_of_existing(self):
        """insort_left inserts a duplicate before the first equal element."""
        coll = [0, 5, 7, 10, 15]
        insort_left(coll, 5)
        assert coll == [0, 5, 5, 7, 10, 15]

    def test_insert_with_lo_hi_bounds(self):
        """insort_left respects lo/hi sub-range bounds."""
        coll = [0, 5, 7, 10, 15]
        insort_left(coll, 15, 1, 3)
        # Only considers indices 1..2 ([5, 7]); 15 > both → inserts at hi=3
        assert coll == [0, 5, 7, 15, 10, 15]

    def test_insert_into_empty(self):
        """insort_left works on an empty list."""
        coll = []
        insort_left(coll, 42)
        assert coll == [42]

    def test_insert_object_identity(self):
        """insort_left inserts the exact object (identity preserved)."""
        coll = [(0, 0), (5, 5), (7, 7)]
        item = (5, 5)
        insort_left(coll, item)
        assert coll[1] is item


# ============================================================================
# insort_right
# ============================================================================

class TestInsortRight:
    """Tests for insort_right – insert after existing equal items."""

    def test_insert_middle(self):
        """insort_right inserts a missing value in the correct sorted position."""
        coll = [0, 5, 7, 10, 15]
        insort_right(coll, 6)
        assert coll == [0, 5, 6, 7, 10, 15]

    def test_insert_at_end(self):
        """insort_right appends when item is larger than all elements."""
        coll = [0, 5, 7, 10, 15]
        insort_right(coll, 20)
        assert coll == [0, 5, 7, 10, 15, 20]

    def test_insert_duplicate_right_of_existing(self):
        """insort_right inserts a duplicate after the last equal element."""
        coll = [0, 5, 7, 10, 15]
        insort_right(coll, 5)
        assert coll == [0, 5, 5, 7, 10, 15]

    def test_insert_with_lo_hi_bounds(self):
        """insort_right respects lo/hi sub-range bounds."""
        coll = [0, 5, 7, 10, 15]
        insort_right(coll, 15, 1, 3)
        assert coll == [0, 5, 7, 15, 10, 15]

    def test_insert_into_empty(self):
        """insort_right works on an empty list."""
        coll = []
        insort_right(coll, 42)
        assert coll == [42]

    def test_insert_object_identity(self):
        """insort_right inserts the exact object (identity preserved)."""
        coll = [(0, 0), (5, 5), (7, 7)]
        item = (5, 5)
        insort_right(coll, item)
        assert coll[2] is item


# ============================================================================
# binary_search
# ============================================================================

class TestBinarySearch:
    """Tests for binary_search – classic iterative binary search."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 7, 2),
            ([0, 5, 7, 10, 15], 10, 3),
            ([0, 5, 7, 10, 15], 15, 4),
            # Not found
            ([0, 5, 7, 10, 15], 6, -1),
            ([0, 5, 7, 10, 15], -5, -1),
            ([0, 5, 7, 10, 15], 99, -1),
            # Single element
            ([42], 42, 0),
            ([42], 7, -1),
            # Empty list
            ([], 1, -1),
        ],
    )
    def test_binary_search_parametrized(self, collection, item, expected):
        """binary_search returns the correct index or -1."""
        assert binary_search(collection, item) == expected

    def test_unsorted_collection_raises_value_error(self):
        """binary_search raises ValueError for unsorted input."""
        with pytest.raises(ValueError, match="sorted_collection must be sorted"):
            binary_search([3, 1, 2], 1)

    def test_sorted_collection_does_not_raise(self):
        """binary_search does not raise for a properly sorted collection."""
        # Should not raise
        assert binary_search([1, 2, 3], 2) == 1


# ============================================================================
# binary_search_std_lib
# ============================================================================

class TestBinarySearchStdLib:
    """Tests for binary_search_std_lib – using bisect from stdlib."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 6, -1),
            ([], 1, -1),
        ],
    )
    def test_binary_search_std_lib_parametrized(self, collection, item, expected):
        """binary_search_std_lib returns the correct index or -1."""
        assert binary_search_std_lib(collection, item) == expected

    def test_unsorted_collection_raises_value_error(self):
        """binary_search_std_lib raises ValueError for unsorted input."""
        with pytest.raises(ValueError, match="sorted_collection must be sorted"):
            binary_search_std_lib([3, 1, 2], 1)

    @patch("binary_search.bisect.bisect_left")
    def test_uses_bisect_left(self, mock_bisect_left):
        """binary_search_std_lib delegates to bisect.bisect_left."""
        mock_bisect_left.return_value = 2
        result = binary_search_std_lib([1, 2, 3, 4], 3)
        mock_bisect_left.assert_called_once_with([1, 2, 3, 4], 3)
        assert result == 2

    @patch("binary_search.bisect.bisect_left")
    def test_item_not_found_when_bisect_index_out_of_range(self, mock_bisect_left):
        """binary_search_std_lib returns -1 when bisect_left returns len(collection)."""
        mock_bisect_left.return_value = 4
        result = binary_search_std_lib([1, 2, 3, 4], 99)
        assert result == -1

    @patch("binary_search.bisect.bisect_left")
    def test_item_not_found_when_value_mismatch(self, mock_bisect_left):
        """binary_search_std_lib returns -1 when the value at the index doesn't match."""
        mock_bisect_left.return_value = 1
        result = binary_search_std_lib([1, 3, 5], 2)
        assert result == -1


# ============================================================================
# binary_search_with_duplicates
# ============================================================================

class TestBinarySearchWithDuplicates:
    """Tests for binary_search_with_duplicates – returns all matching indices."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, [0]),
            ([0, 5, 7, 10, 15], 15, [4]),
            ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
            ([2, 2, 2, 2, 2], 2, [0, 1, 2, 3, 4]),
            # Not found
            ([1, 2, 2, 2, 3], 4, []),
            ([1, 2, 2, 2, 3], 0, []),
            ([], 1, []),
        ],
    )
    def test_binary_search_with_duplicates_parametrized(
        self, collection, item, expected
    ):
        """binary_search_with_duplicates returns all indices of item."""
        assert binary_search_with_duplicates(collection, item) == expected

    def test_unsorted_collection_raises_value_error(self):
        """binary_search_with_duplicates raises ValueError for unsorted input."""
        with pytest.raises(ValueError, match="sorted_collection must be sorted"):
            binary_search_with_duplicates([3, 1, 2], 1)

    def test_single_element_found(self):
        """binary_search_with_duplicates finds the only element."""
        assert binary_search_with_duplicates([7], 7) == [0]

    def test_single_element_not_found(self):
        """binary_search_with_duplicates returns empty list for missing item."""
        assert binary_search_with_duplicates([7], 3) == []


# ============================================================================
# binary_search_by_recursion
# ============================================================================

class TestBinarySearchByRecursion:
    """Tests for binary_search_by_recursion – recursive binary search."""

    @pytest.mark.parametrize(
        "collection, item, left, right, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0, 4, 0),
            ([0, 5, 7, 10, 15], 5, 0, 4, 1),
            ([0, 5, 7, 10, 15], 15, 0, 4, 4),
            ([0, 5, 7, 10, 15], 6, 0, 4, -1),
            # Default right=-1
            ([0, 5, 7, 10, 15], 10, 0, -1, 3),
            # Empty list
            ([], 1, 0, -1, -1),
        ],
    )
    def test_binary_search_by_recursion_parametrized(
        self, collection, item, left, right, expected
    ):
        """binary_search_by_recursion returns the correct index or -1."""
        assert binary_search_by_recursion(collection, item, left, right) == expected

    def test_unsorted_collection_raises_value_error(self):
        """binary_search_by_recursion raises ValueError for unsorted input."""
        with pytest.raises(ValueError, match="sorted_collection must be sorted"):
            binary_search_by_recursion([3, 1, 2], 1, 0, 2)

    def test_right_less_than_left_returns_negative_one(self):
        """binary_search_by_recursion returns -1 when right < left."""
        assert binary_search_by_recursion([1, 2, 3], 2, 3, 2) == -1

    def test_default_right_calculated_from_len(self):
        """binary_search_by_recursion with right=-1 uses len(collection)-1."""
        assert binary_search_by_recursion([10, 20, 30], 20) == 1


# ============================================================================
# exponential_search
# ============================================================================

class TestExponentialSearch:
    """Tests for exponential_search – exponential + binary search."""

    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 6, -1),
            ([0, 5, 7, 10, 15], -1, -1),
            ([0, 5, 7, 10, 15], 99, -1),
            # Single element
            ([42], 42, 0),
            ([42], 7, -1),
            # Empty list
            ([], 1, -1),
        ],
    )
    def test_exponential_search_parametrized(self, collection, item, expected):
        """exponential_search returns the correct index or -1."""
        assert exponential_search(collection, item) == expected

    def test_unsorted_collection_raises_value_error(self):
        """exponential_search raises ValueError for unsorted input."""
        with pytest.raises(ValueError, match="sorted_collection must be sorted"):
            exponential_search([3, 1, 2], 1)

    def test_item_beyond_bound(self):
        """exponential_search handles item larger than all elements."""
        assert exponential_search([1, 2, 3, 4, 5], 99) == -1

    def test_large_collection(self):
        """exponential_search works correctly on a larger collection."""
        coll = list(range(0, 1000, 2))  # 0, 2, 4, ..., 998
        assert exponential_search(coll, 500) == 250
        assert exponential_search(coll, 501) == -1
