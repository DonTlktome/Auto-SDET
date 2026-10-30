"""Tests for the merge_sort module."""

import pytest

from sorts.merge_sort import merge_sort


class TestMergeSortHappyPath:
    """Positive test cases for merge_sort."""

    def test_sorts_unsorted_list(self):
        """A typical unsorted list is returned in ascending order."""
        assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]

    def test_sorts_negative_numbers(self):
        """Negative numbers are sorted correctly."""
        assert merge_sort([-2, -5, -45]) == [-45, -5, -2]

    def test_sorts_mixed_sign_numbers(self):
        """A mix of negative, zero and positive values is sorted."""
        assert merge_sort([3, -1, 0, -10, 7, 2]) == [-10, -1, 0, 2, 3, 7]

    def test_sorts_floats(self):
        """Floating point values are sorted correctly."""
        assert merge_sort([2.5, 1.1, 3.3, 0.0]) == [0.0, 1.1, 2.5, 3.3]

    def test_sorts_strings(self):
        """Strings are sorted lexicographically."""
        assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]

    def test_sorts_tuples(self):
        """Tuples are sorted by their natural ordering."""
        assert merge_sort([(2, "b"), (1, "a"), (2, "a")]) == [(1, "a"), (2, "a"), (2, "b")]


class TestMergeSortEdgeCases:
    """Boundary and edge cases for merge_sort."""

    def test_empty_list(self):
        """An empty list is returned unchanged."""
        assert merge_sort([]) == []

    def test_single_element(self):
        """A single element list is returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_two_elements_sorted(self):
        """A two element list that is already sorted stays sorted."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_two_elements_reversed(self):
        """A two element list in reverse order is swapped."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_already_sorted(self):
        """An already sorted list is returned unchanged."""
        data = [1, 2, 3, 4, 5]
        assert merge_sort(data) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        """A reverse sorted list is fully reversed."""
        assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_all_equal_elements(self):
        """A list of identical elements is returned unchanged."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_duplicates_preserved(self):
        """Duplicate values are all preserved in the output."""
        result = merge_sort([3, 1, 3, 2, 1, 3])
        assert result == [1, 1, 2, 3, 3, 3]

    def test_odd_length_list(self):
        """An odd-length list is sorted correctly (exercises uneven split)."""
        assert merge_sort([9, 1, 5, 3, 7]) == [1, 3, 5, 7, 9]

    def test_even_length_list(self):
        """An even-length list is sorted correctly."""
        assert merge_sort([8, 4, 6, 2]) == [2, 4, 6, 8]

    def test_large_list(self):
        """A large list is sorted correctly and matches Python's sorted()."""
        data = list(range(1000, 0, -1))
        assert merge_sort(data) == sorted(data)

    def test_does_not_mutate_input(self):
        """The original input list is not modified by merge_sort."""
        data = [3, 1, 2]
        merge_sort(data)
        assert data == [3, 1, 2]

    def test_returns_new_list_for_multi_element_input(self):
        """For inputs longer than one element a new list object is returned."""
        data = [2, 1]
        result = merge_sort(data)
        assert result == [1, 2]
        assert result is not data

    def test_returns_same_object_for_single_element(self):
        """For a single element the same list object is returned (early return)."""
        data = [1]
        assert merge_sort(data) is data

    def test_returns_same_object_for_empty_list(self):
        """For an empty list the same list object is returned (early return)."""
        data = []
        assert merge_sort(data) is data


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
        ([4, 4, 4], [4, 4, 4]),
        ([10, -10, 0], [-10, 0, 10]),
        (["b", "a", "c"], ["a", "b", "c"]),
    ],
)
def test_merge_sort_parametrized(collection, expected):
    """Parametrized check that merge_sort matches the expected sorted output."""
    assert merge_sort(collection) == expected


@pytest.mark.parametrize(
    "collection",
    [
        [],
        [1],
        [2, 1],
        [5, 3, 8, 1, 9, 2, 7],
        list(range(50, 0, -1)),
        [0, 0, 0, 1, 1, 0],
    ],
)
def test_merge_sort_matches_builtin_sorted(collection):
    """merge_sort output always matches Python's built-in sorted()."""
    assert merge_sort(collection) == sorted(collection)