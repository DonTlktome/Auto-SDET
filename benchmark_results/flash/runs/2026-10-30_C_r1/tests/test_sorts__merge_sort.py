"""Tests for the merge sort implementation in sorts/merge_sort.py."""

import random

import pytest

from sorts.merge_sort import merge_sort


class TestMergeSortHappyPath:
    """Positive test cases for merge_sort."""

    def test_sorts_unsorted_list(self):
        """A typical unsorted list is returned in ascending order."""
        assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]

    def test_sorts_already_sorted_list(self):
        """An already sorted list stays sorted."""
        assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_sorts_reverse_sorted_list(self):
        """A reverse sorted list is fully reversed into ascending order."""
        assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]

    def test_sorts_negative_numbers(self):
        """Negative numbers are ordered correctly."""
        assert merge_sort([-2, -5, -45]) == [-45, -5, -2]

    def test_sorts_mixed_sign_numbers(self):
        """Mixed positive, negative and zero values are ordered correctly."""
        assert merge_sort([3, -1, 0, -7, 8, 2]) == [-7, -1, 0, 2, 3, 8]

    def test_sorts_floats(self):
        """Floating point values are supported since they are comparable."""
        assert merge_sort([2.5, 1.1, 3.3, 0.0]) == [0.0, 1.1, 2.5, 3.3]

    def test_sorts_strings(self):
        """Strings are comparable and therefore sortable."""
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

    def test_two_element_list_sorted(self):
        """A two element sorted list is returned unchanged."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_two_element_list_unsorted(self):
        """A two element unsorted list is swapped."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_all_duplicates(self):
        """A list of identical values is returned unchanged."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_duplicates_preserved(self):
        """Duplicate values are all preserved in the output."""
        result = merge_sort([4, 1, 4, 2, 1, 4])
        assert result == [1, 1, 2, 4, 4, 4]
        assert len(result) == 6

    def test_large_values(self):
        """Very large integers are handled correctly."""
        assert merge_sort([10**18, -(10**18), 0]) == [-(10**18), 0, 10**18]

    def test_does_not_mutate_input(self):
        """The original input list is not modified by merge_sort."""
        original = [3, 1, 2]
        snapshot = list(original)
        merge_sort(original)
        assert original == snapshot

    def test_returns_new_list_for_multi_element_input(self):
        """For inputs longer than one element a new list object is returned."""
        original = [3, 1, 2]
        result = merge_sort(original)
        assert result is not original

    def test_single_element_returns_same_object(self):
        """For a single element the same list object is returned (early exit)."""
        original = [1]
        assert merge_sort(original) is original

    def test_empty_list_returns_same_object(self):
        """For an empty list the same list object is returned (early exit)."""
        original = []
        assert merge_sort(original) is original


class TestMergeSortParametrized:
    """Parametrized comparisons against Python's built-in sorted()."""

    @pytest.mark.parametrize(
        "collection",
        [
            [],
            [1],
            [2, 1],
            [0, 5, 3, 2, 2],
            [-2, -5, -45],
            [1, 1, 1, 1, 1],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [100, -100, 50, -50, 0],
            [1.5, -2.5, 0.0, 3.25],
            ["b", "a", "c", "aa"],
        ],
    )
    def test_matches_builtin_sorted(self, collection):
        """merge_sort output matches Python's built-in sorted() for many inputs."""
        assert merge_sort(list(collection)) == sorted(collection)

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 100, 257])
    def test_random_lists_match_builtin_sorted(self, size):
        """Randomly generated lists of various sizes match sorted()."""
        rng = random.Random(size)
        data = [rng.randint(-1000, 1000) for _ in range(size)]
        assert merge_sort(list(data)) == sorted(data)

    @pytest.mark.parametrize(
        "collection, expected",
        [
            ([3, 1, 2], [1, 2, 3]),
            ([], []),
            ([5], [5]),
            ([-1, -2, -3], [-3, -2, -1]),
        ],
    )
    def test_explicit_expected_outputs(self, collection, expected):
        """Explicit input/expected pairs produce the expected sorted output."""
        assert merge_sort(collection) == expected


class TestMergeSortStabilityAndOrdering:
    """Tests around ordering semantics of merge_sort."""

    def test_stable_for_equal_keys(self):
        """Equal keys keep their original relative order (stable sort)."""
        data = [(1, "a"), (0, "b"), (1, "c"), (0, "d")]
        result = merge_sort(data)
        assert result == [(0, "b"), (0, "d"), (1, "a"), (1, "c")]

    def test_output_is_sorted_ascending(self):
        """The output is non-decreasing for a moderately sized random list."""
        rng = random.Random(12345)
        data = [rng.randint(-50, 50) for _ in range(50)]
        result = merge_sort(list(data))
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_output_is_permutation_of_input(self):
        """The output contains exactly the same multiset of elements as the input."""
        data = [5, 3, 8, 3, 1, 9, 1]
        result = merge_sort(list(data))
        assert sorted(result) == sorted(data)
        assert len(result) == len(data)