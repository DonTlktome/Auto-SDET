"""Tests for the merge_sort module."""

import random

import pytest

from sorts.merge_sort import merge_sort


class TestMergeSortHappyPath:
    """Positive test cases for merge_sort."""

    def test_sorts_unsorted_list(self):
        """A typical unsorted list of integers is sorted in ascending order."""
        assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]

    def test_sorts_already_sorted_list(self):
        """An already sorted list is returned unchanged."""
        assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_sorts_reverse_sorted_list(self):
        """A reverse sorted list is sorted in ascending order."""
        assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_sorts_negative_numbers(self):
        """Negative numbers are sorted correctly."""
        assert merge_sort([-2, -5, -45]) == [-45, -5, -2]

    def test_sorts_floats(self):
        """Floating point values are sorted correctly."""
        assert merge_sort([3.5, 1.2, 2.8, 0.1]) == [0.1, 1.2, 2.8, 3.5]

    def test_sorts_strings(self):
        """Strings are sorted lexicographically."""
        assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]

    def test_sorts_mixed_sign_numbers(self):
        """A mix of negative, zero and positive values is sorted correctly."""
        assert merge_sort([3, -1, 0, -7, 5]) == [-7, -1, 0, 3, 5]


class TestMergeSortEdgeCases:
    """Boundary and edge cases for merge_sort."""

    def test_empty_list(self):
        """An empty list returns an empty list."""
        assert merge_sort([]) == []

    def test_single_element(self):
        """A single element list is returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_two_elements_unsorted(self):
        """A two element unsorted list is sorted."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_two_elements_sorted(self):
        """A two element sorted list is returned unchanged."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_all_duplicates(self):
        """A list of identical values is returned unchanged."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_duplicates_preserved(self):
        """Duplicate values are all preserved in the output."""
        result = merge_sort([3, 1, 3, 2, 1])
        assert result == [1, 1, 2, 3, 3]
        assert len(result) == 5

    def test_does_not_mutate_input(self):
        """The original input list is not modified by merge_sort."""
        original = [4, 2, 9, 1]
        snapshot = list(original)
        merge_sort(original)
        assert original == snapshot

    def test_returns_new_list_for_multi_element(self):
        """A multi-element input yields a distinct list object."""
        original = [3, 1, 2]
        result = merge_sort(original)
        assert result is not original

    def test_returns_same_object_for_single_element(self):
        """A list of length <= 1 is returned as-is (same object)."""
        original = [5]
        assert merge_sort(original) is original

    def test_returns_same_object_for_empty(self):
        """An empty list is returned as-is (same object)."""
        original = []
        assert merge_sort(original) is original


class TestMergeSortParametrized:
    """Parametrized coverage of many input/output pairs."""

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            ([], []),
            ([1], [1]),
            ([2, 1], [1, 2]),
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([-2, -5, -45], [-45, -5, -2]),
            ([10, -10, 0], [-10, 0, 10]),
            ([1, 1, 1], [1, 1, 1]),
            ([9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ([100, 1, 50, 25, 75], [1, 25, 50, 75, 100]),
        ],
    )
    def test_various_inputs(self, input_list, expected):
        """merge_sort produces the expected sorted output for many inputs."""
        assert merge_sort(input_list) == expected

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 50, 100])
    def test_matches_builtin_sorted(self, size):
        """merge_sort output matches Python's built-in sorted for random data."""
        rng = random.Random(size)
        data = [rng.randint(-1000, 1000) for _ in range(size)]
        assert merge_sort(list(data)) == sorted(data)

    @pytest.mark.parametrize("size", [1, 2, 5, 20])
    def test_output_length_preserved(self, size):
        """The number of elements is preserved after sorting."""
        data = list(range(size, 0, -1))
        assert len(merge_sort(data)) == size