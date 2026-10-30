"""Tests for the selection_sort module."""

import pytest

from sorts.selection_sort import selection_sort


class TestSelectionSortPositive:
    """Happy-path tests for selection_sort."""

    def test_sorts_unsorted_list(self):
        """A typical unsorted list is returned in ascending order."""
        assert selection_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]

    def test_sorts_negative_numbers(self):
        """Negative integers are sorted correctly."""
        assert selection_sort([-2, -5, -45]) == [-45, -5, -2]

    def test_sorts_mixed_sign_numbers(self):
        """A mix of negative, zero and positive values is sorted."""
        assert selection_sort([3, -1, 0, -7, 8]) == [-7, -1, 0, 3, 8]

    def test_returns_same_list_object(self):
        """The function sorts in place and returns the same list object."""
        data = [4, 1, 3]
        result = selection_sort(data)
        assert result is data
        assert data == [1, 3, 4]


class TestSelectionSortEdgeCases:
    """Boundary and edge-case tests for selection_sort."""

    def test_empty_list(self):
        """An empty list is returned unchanged."""
        assert selection_sort([]) == []

    def test_single_element(self):
        """A single-element list is returned unchanged."""
        assert selection_sort([42]) == [42]

    def test_two_elements_already_sorted(self):
        """A two-element sorted list stays sorted."""
        assert selection_sort([1, 2]) == [1, 2]

    def test_two_elements_reversed(self):
        """A two-element reversed list gets swapped."""
        assert selection_sort([2, 1]) == [1, 2]

    def test_already_sorted_list(self):
        """An already sorted list is returned unchanged."""
        data = [1, 2, 3, 4, 5]
        assert selection_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted_list(self):
        """A reverse sorted list is fully reversed."""
        assert selection_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_all_equal_elements(self):
        """A list of identical elements is returned unchanged."""
        assert selection_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_duplicates_preserved(self):
        """Duplicate values are all preserved after sorting."""
        assert selection_sort([3, 1, 3, 1, 2]) == [1, 1, 2, 3, 3]

    def test_large_values(self):
        """Very large and very small integers are handled."""
        data = [10**18, -(10**18), 0]
        assert selection_sort(data) == [-(10**18), 0, 10**18]


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([1], [1]),
        ([2, 1], [1, 2]),
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([-2, -5, -45], [-45, -5, -2]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([9, 9, 9], [9, 9, 9]),
        ([100, -100, 0, 50, -50], [-100, -50, 0, 50, 100]),
    ],
)
def test_selection_sort_parametrized(collection, expected):
    """selection_sort produces the expected output across many inputs."""
    assert selection_sort(collection) == expected


@pytest.mark.parametrize(
    "collection",
    [
        [],
        [1],
        [3, 1, 2],
        [5, 5, 5, 1, 1],
        list(range(20, 0, -1)),
        [0, -1, 1, -2, 2],
    ],
)
def test_selection_sort_matches_builtin_sorted(collection):
    """selection_sort output matches Python's built-in sorted()."""
    assert selection_sort(list(collection)) == sorted(collection)


def test_selection_sort_does_not_mutate_length():
    """Sorting never changes the number of elements in the list."""
    data = [4, 2, 9, 1, 5, 3]
    original_length = len(data)
    selection_sort(data)
    assert len(data) == original_length