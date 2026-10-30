# AUTO-SDET: Test generation FAILED after 3 retries
# Last error: 

import pytest

from binary_search import (
    binary_search,
    binary_search_by_recursion,
    binary_search_std_lib,
    binary_search_with_duplicates,
    bisect_left,
    bisect_right,
    exponential_search,
    insort_left,
    insort_right,
)


# ---------------------------------------------------------------------------
# bisect_left
# ---------------------------------------------------------------------------
class TestBisectLeft:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 6, 2),
            ([0, 5, 7, 10, 15], 20, 5),
            ([0, 5, 7, 10, 15], 15, 4),
            ([1, 2, 2, 2, 3], 2, 1),
        ],
    )
    def test_bisect_left_happy_path(self, collection, item, expected):
        """bisect_left returns the first index where item can be inserted."""
        assert bisect_left(collection, item) == expected

    @pytest.mark.parametrize(
        "collection, item, lo, hi, expected",
        [
            ([0, 5, 7, 10, 15], 15, 1, 3, 3),
            ([0, 5, 7, 10, 15], 6, 2, -1, 2),
            ([0, 5, 7, 10, 15], 0, 0, 0, 0),
        ],
    )
    def test_bisect_left_with_bounds(self, collection, item, lo, hi, expected):
        """bisect_left respects explicit lo/hi bounds."""
        assert bisect_left(collection, item, lo, hi) == expected

    def test_bisect_left_empty_collection(self):
        """bisect_left on an empty collection returns 0."""
        assert bisect_left([], 5) == 0


# ---------------------------------------------------------------------------
# bisect_right
# ---------------------------------------------------------------------------
class TestBisectRight:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 1),
            ([0, 5, 7, 10, 15], 15, 5),
            ([0, 5, 7, 10, 15], 6, 2),
            ([1, 2, 2, 2, 3], 2, 4),
        ],
    )
    def test_bisect_right_happy_path(self, collection, item, expected):
        """bisect_right returns the index after all equal items."""
        assert bisect_right(collection, item) == expected

    @pytest.mark.parametrize(
        "collection, item, lo, hi, expected",
        [
            ([0, 5, 7, 10, 15], 15, 1, 3, 3),
            ([0, 5, 7, 10, 15], 6, 2, -1, 2),
        ],
    )
    def test_bisect_right_with_bounds(self, collection, item, lo, hi, expected):
        """bisect_right respects explicit lo/hi bounds."""
        assert bisect_right(collection, item, lo, hi) == expected

    def test_bisect_right_empty_collection(self):
        """bisect_right on an empty collection returns 0."""
        assert bisect_right([], 5) == 0


# ---------------------------------------------------------------------------
# insort_left
# ---------------------------------------------------------------------------
class TestInsortLeft:
    def test_insort_left_inserts_before_equal(self):
        """insort_left inserts the item before existing equal values."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 6)
        assert collection == [0, 5, 6, 7, 10, 15]

    def test_insort_left_duplicate_identity(self):
        """insort_left inserts a duplicate before the existing equal item."""
        collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
        item = (5, 5)
        insort_left(collection, item)
        assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
        assert item is collection[1]
        assert item is not collection[2]

    def test_insort_left_appends_at_end(self):
        """insort_left appends when item is larger than all elements."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 20)
        assert collection == [0, 5, 7, 10, 15, 20]

    def test_insort_left_with_bounds(self):
        """insort_left respects lo/hi bounds when inserting."""
        collection = [0, 5, 7, 10, 15]
        insort_left(collection, 15, 1, 3)
        assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# insort_right
# ---------------------------------------------------------------------------
class TestInsortRight:
    def test_insort_right_inserts_after_equal(self):
        """insort_right inserts the item after existing equal values."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 6)
        assert collection == [0, 5, 6, 7, 10, 15]

    def test_insort_right_duplicate_identity(self):
        """insort_right inserts a duplicate after the existing equal item."""
        collection = [(0, 0), (5, 5), (7, 7), (10, 10), (15, 15)]
        item = (5, 5)
        insort_right(collection, item)
        assert collection == [(0, 0), (5, 5), (5, 5), (7, 7), (10, 10), (15, 15)]
        assert item is not collection[1]
        assert item is collection[2]

    def test_insort_right_appends_at_end(self):
        """insort_right appends when item is larger than all elements."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 20)
        assert collection == [0, 5, 7, 10, 15, 20]

    def test_insort_right_with_bounds(self):
        """insort_right respects lo/hi bounds when inserting."""
        collection = [0, 5, 7, 10, 15]
        insort_right(collection, 15, 1, 3)
        assert collection == [0, 5, 7, 15, 10, 15]


# ---------------------------------------------------------------------------
# binary_search
# ---------------------------------------------------------------------------
class TestBinarySearch:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 6, -1),
        ],
    )
    def test_binary_search_happy_path(self, collection, item, expected):
        """binary_search finds existing items and returns -1 for missing ones."""
        assert binary_search(collection, item) == expected

    def test_binary_search_empty_collection(self):
        """binary_search on an empty collection returns -1."""
        assert binary_search([], 5) == -1

    def test_binary_search_single_element_found(self):
        """binary_search finds the only element in a single-element list."""
        assert binary_search([42], 42) == 0

    def test_binary_search_single_element_not_found(self):
        """binary_search returns -1 for a missing item in a single-element list."""
        assert binary_search([42], 7) == -1

    def test_binary_search_unsorted_raises(self):
        """binary_search raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError):
            binary_search([3, 1, 2], 2)


# ---------------------------------------------------------------------------
# binary_search_std_lib
# ---------------------------------------------------------------------------
class TestBinarySearchStdLib:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 6, -1),
        ],
    )
    def test_binary_search_std_lib_happy_path(self, collection, item, expected):
        """binary_search_std_lib finds items and returns -1 for missing ones."""
        assert binary_search_std_lib(collection, item) == expected

    def test_binary_search_std_lib_empty_collection(self):
        """binary_search_std_lib on an empty collection returns -1."""
        assert binary_search_std_lib([], 5) == -1

    def test_binary_search_std_lib_unsorted_raises(self):
        """binary_search_std_lib raises ValueError for an unsorted collection."""
        with pytest.raises(ValueError):
            binary_search_std_lib([3, 1, 2], 2)


# ---------------------------------------------------------------------------
# binary_search_with_duplicates
# ---------------------------------------------------------------------------
class TestBinarySearchWithDuplicates:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, [0]),
            ([0, 5, 7, 10, 15], 15, [4]),
            ([1, 2, 2, 2, 3], 2, [1, 2, 3]),
            ([1, 2, 2, 2, 3], 4, []),
        ],
    )
    def test_binary_search_with_duplicates_happy_path(
        self, collection, item, expected
    ):
        """binary_search_with_duplicates returns all matching indexes."""
        assert binary_search_with_duplicates(collection, item) == expected

    def test_binary_search_with_duplicates_all_same(self):
        """binary_search_with_duplicates returns every index when all equal."""
        assert binary_search_with_duplicates([2, 2, 2, 2], 2) == [0, 1, 2, 3]

    def test_binary_search_with_duplicates_empty_collection(self):
        """binary_search_with_duplicates on an empty collection returns []."""
        assert binary_search_with_duplicates([], 5) == []

    def test_binary_search_with_duplicates_unsorted_raises(self):
        """binary_search_with_duplicates raises ValueError for unsorted input."""
        with pytest.raises(ValueError):
            binary_search_with_duplicates([3, 1, 2], 2)


# ---------------------------------------------------------------------------
# binary_search_by_recursion
# ---------------------------------------------------------------------------
class TestBinarySearchByRecursion:
    @pytest.mark.parametrize(
        "collection, item, left, right, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0, 4, 0),
            ([0, 5, 7, 10, 15], 15, 0, 4, 4),
            ([0, 5, 7, 10, 15], 5, 0, 4, 1),
            ([0, 5, 7, 10, 15], 6, 0, 4, -1),
        ],
    )
    def test_binary_search_by_recursion_happy_path(
        self, collection, item, left, right, expected
    ):
        """binary_search_by_recursion finds items and returns -1 for missing."""
        assert binary_search_by_recursion(collection, item, left, right) == expected

    def test_binary_search_by_recursion_default_right(self):
        """binary_search_by_recursion computes right when it defaults to -1."""
        assert binary_search_by_recursion([0, 5, 7, 10, 15], 10) == 3

    def test_binary_search_by_recursion_empty_collection(self):
        """binary_search_by_recursion on an empty collection returns -1."""
        assert binary_search_by_recursion([], 5) == -1

    def test_binary_search_by_recursion_unsorted_raises(self):
        """binary_search_by_recursion raises ValueError for unsorted input."""
        with pytest.raises(ValueError):
            binary_search_by_recursion([3, 1, 2], 2)


# ---------------------------------------------------------------------------
# exponential_search
# ---------------------------------------------------------------------------
class TestExponentialSearch:
    @pytest.mark.parametrize(
        "collection, item, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),
            ([0, 5, 7, 10, 15], 15, 4),
            ([0, 5, 7, 10, 15], 5, 1),
            ([0, 5, 7, 10, 15], 6, -1),
        ],
    )
    def test_exponential_search_happy_path(self, collection, item, expected):
        """exponential_search finds items and returns -1 for missing ones."""
        assert exponential_search(collection, item) == expected

    def test_exponential_search_empty_collection(self):
        """exponential_search on an empty collection returns -1."""
        assert exponential_search([], 5) == -1

    def test_exponential_search_large_collection(self):
        """exponential_search works on a large collection."""
        collection = list(range(1000))
        assert exponential_search(collection, 500) == 500
        assert exponential_search(collection, 9999) == -1

    def test_exponential_search_unsorted_raises(self):
        """exponential_search raises ValueError for unsorted input."""
        with pytest.raises(ValueError):
            exponential_search([3, 1, 2], 2)