"""Tests for the merge_sort module."""

import pytest

from sorts.merge_sort import merge_sort


class TestMergeSortHappyPath:
    """Positive test cases for merge_sort."""

    def test_sorts_unsorted_list(self):
        """A typical unsorted list is returned in ascending order."""
        assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]

    def test_sorts_already_sorted_list(self):
        """An already sorted list remains unchanged."""
        assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_sorts_reverse_sorted_list(self):
        """A reverse sorted list is fully reversed into ascending order."""
        assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_sorts_negative_numbers(self):
        """Negative numbers are ordered correctly."""
        assert merge_sort([-2, -5, -45]) == [-45, -5, -2]

    def test_sorts_mixed_sign_numbers(self):
        """A mix of negative, zero and positive values is ordered correctly."""
        assert merge_sort([3, -1, 0, -7, 8, 2]) == [-7, -1, 0, 2, 3, 8]

    def test_sorts_floats(self):
        """Floating point values are ordered correctly."""
        assert merge_sort([2.5, 1.1, 3.3, 0.0]) == [0.0, 1.1, 2.5, 3.3]

    def test_sorts_strings(self):
        """Strings are ordered lexicographically."""
        assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]

    def test_returns_list_type(self):
        """The return value is always a list."""
        assert isinstance(merge_sort([3, 1, 2]), list)


class TestMergeSortEdgeCases:
    """Boundary and edge cases for merge_sort."""

    def test_empty_list(self):
        """An empty list is returned unchanged."""
        assert merge_sort([]) == []

    def test_single_element_list(self):
        """A single element list is returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_two_element_list_unsorted(self):
        """The smallest non-trivial case is sorted correctly."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_two_element_list_sorted(self):
        """A sorted two element list is returned unchanged."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_all_duplicates(self):
        """A list of identical values is returned unchanged."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_duplicates_preserved(self):
        """Duplicate values are all preserved in the output."""
        result = merge_sort([3, 1, 3, 2, 1])
        assert result == [1, 1, 2, 3, 3]
        assert len(result) == 5

    def test_odd_length_list(self):
        """An odd-length list is sorted correctly (exercises uneven split)."""
        assert merge_sort([9, 1, 5, 3, 7]) == [1, 3, 5, 7, 9]

    def test_even_length_list(self):
        """An even-length list is sorted correctly (exercises even split)."""
        assert merge_sort([8, 4, 6, 2]) == [2, 4, 6, 8]

    def test_large_list(self):
        """A large list is sorted correctly and preserves all elements."""
        data = list(range(1000, 0, -1))
        result = merge_sort(data)
        assert result == list(range(1, 1001))

    def test_does_not_mutate_input(self):
        """The original input list is not modified by merge_sort."""
        original = [3, 1, 2]
        snapshot = list(original)
        merge_sort(original)
        assert original == snapshot

    def test_returns_new_list_for_multi_element_input(self):
        """A new list object is returned for inputs with more than one element."""
        original = [2, 1]
        result = merge_sort(original)
        assert result == [1, 2]
        assert result is not original

    def test_single_element_returns_same_object(self):
        """For a single element the same list object is returned (early exit)."""
        original = [5]
        assert merge_sort(original) is original

    def test_empty_list_returns_same_object(self):
        """For an empty list the same list object is returned (early exit)."""
        original = []
        assert merge_sort(original) is original


class TestMergeSortParametrized:
    """Parametrized coverage of many input/output pairs."""

    @pytest.mark.parametrize(
        "collection, expected",
        [
            ([], []),
            ([1], [1]),
            ([2, 1], [1, 2]),
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([-2, -5, -45], [-45, -5, -2]),
            ([10, -10, 0], [-10, 0, 10]),
            ([1, 1, 1], [1, 1, 1]),
            ([5, 3, 8, 1, 9, 2, 7], [1, 2, 3, 5, 7, 8, 9]),
            ([100, 50, 25, 12, 6, 3], [3, 6, 12, 25, 50, 100]),
            (["b", "a", "c"], ["a", "b", "c"]),
        ],
    )
    def test_merge_sort_parametrized(self, collection, expected):
        """merge_sort returns the expected sorted output for each input."""
        assert merge_sort(collection) == expected

    @pytest.mark.parametrize(
        "collection",
        [
            [],
            [1],
            [2, 1],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5],
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
            [-1, -2, -3, 0, 1, 2, 3],
        ],
    )
    def test_output_is_sorted_and_same_length(self, collection):
        """The output is sorted ascending and has the same length as the input."""
        result = merge_sort(list(collection))
        assert result == sorted(collection)
        assert len(result) == len(collection)