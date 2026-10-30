"""Tests for the merge sort implementation in sorts/merge_sort.py."""

import random

import pytest

from sorts.merge_sort import merge_sort


class TestMergeSortHappyPath:
    """Positive test cases for merge_sort."""

    @pytest.mark.parametrize(
        "collection, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([-2, -5, -45], [-45, -5, -2]),
            ([3, -1, 0, -1, 2], [-1, -1, 0, 2, 3]),
            ([10, 1, 9, 2, 8, 3], [1, 2, 3, 8, 9, 10]),
        ],
    )
    def test_sorts_integers_ascending(self, collection, expected):
        """merge_sort should return the collection in ascending order."""
        assert merge_sort(collection) == expected

    def test_sorts_floats(self):
        """merge_sort should handle float values."""
        assert merge_sort([2.5, 1.1, 3.3, -0.5]) == [-0.5, 1.1, 2.5, 3.3]

    def test_sorts_strings(self):
        """merge_sort should handle comparable string values."""
        assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]

    def test_sorts_tuples(self):
        """merge_sort should handle tuples of comparable items."""
        assert merge_sort([(2, "b"), (1, "a"), (2, "a")]) == [(1, "a"), (2, "a"), (2, "b")]

    def test_returns_list_type(self):
        """merge_sort should always return a list."""
        assert isinstance(merge_sort([3, 1, 2]), list)

    def test_does_not_mutate_input(self):
        """merge_sort should not modify the original collection."""
        original = [4, 2, 7, 1]
        snapshot = list(original)
        merge_sort(original)
        assert original == snapshot

    def test_matches_builtin_sorted_on_random_data(self):
        """merge_sort output should match Python's built-in sorted for random data."""
        rng = random.Random(1234)
        data = [rng.randint(-1000, 1000) for _ in range(200)]
        assert merge_sort(data) == sorted(data)


class TestMergeSortEdgeCases:
    """Boundary and edge cases for merge_sort."""

    def test_empty_list(self):
        """An empty list should be returned unchanged."""
        assert merge_sort([]) == []

    def test_single_element(self):
        """A single-element list should be returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_two_elements_sorted(self):
        """A two-element already-sorted list should stay sorted."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_two_elements_reversed(self):
        """A two-element reversed list should be swapped."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_all_duplicates(self):
        """A list of identical values should be returned unchanged."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_negative_and_positive_mix(self):
        """Mixed-sign values should be sorted correctly."""
        assert merge_sort([0, -1, 1, -2, 2]) == [-2, -1, 0, 1, 2]

    def test_large_list(self):
        """A large list should be sorted correctly."""
        data = list(range(1000, 0, -1))
        assert merge_sort(data) == list(range(1, 1001))

    def test_odd_length_list(self):
        """An odd-length list should be sorted correctly."""
        assert merge_sort([9, 3, 7, 1, 5]) == [1, 3, 5, 7, 9]

    def test_even_length_list(self):
        """An even-length list should be sorted correctly."""
        assert merge_sort([8, 4, 6, 2]) == [2, 4, 6, 8]

    def test_stability_of_equal_keys(self):
        """Equal keys should preserve their relative order (stable sort)."""
        data = [(1, "a"), (1, "b"), (0, "c"), (1, "d")]
        assert merge_sort(data) == [(0, "c"), (1, "a"), (1, "b"), (1, "d")]

    def test_returns_same_object_for_trivial_input(self):
        """For length <= 1 the input object itself is returned (documented behavior)."""
        data = [5]
        assert merge_sort(data) is data

    def test_raises_on_incomparable_types(self):
        """Sorting a list of incomparable types should raise TypeError."""
        with pytest.raises(TypeError):
            merge_sort([1, "a", 2])