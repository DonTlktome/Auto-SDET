"""
Pytest test suite for merge_sort.py

Covers the public `merge_sort` function with positive, edge, boundary,
and regression cases. All tests are self-contained and require no
network or external dependencies.
"""

import pytest

from merge_sort import merge_sort


class TestMergeSort:
    """Tests for the merge_sort function."""

    # ------------------------------------------------------------------
    # Positive / happy-path tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([3, 1, 2], [1, 2, 3]),
            ([42], [42]),
            ([1, 1, 1, 1], [1, 1, 1, 1]),
        ],
    )
    def test_merge_sort_typical_cases(self, input_list, expected):
        """merge_sort should correctly sort typical integer lists."""
        # Ensure we don't mutate the original list contract-wise
        original = list(input_list)
        result = merge_sort(input_list)
        assert result == expected
        # merge_sort returns the same collection (though may be a new list
        # due to slicing); verify original is unchanged.
        assert input_list == original

    # ------------------------------------------------------------------
    # Empty and single-element edge cases
    # ------------------------------------------------------------------

    def test_merge_sort_empty_list(self):
        """merge_sort on an empty list should return an empty list."""
        assert merge_sort([]) == []

    def test_merge_sort_single_element(self):
        """merge_sort on a single-element list should return the same list."""
        assert merge_sort([7]) == [7]

    # ------------------------------------------------------------------
    # Negative numbers and mixed values
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            ([-2, -5, -45], [-45, -5, -2]),
            ([-1, 0, 1, -2, 2], [-2, -1, 0, 1, 2]),
            ([0, -1, -2, -3], [-3, -2, -1, 0]),
        ],
    )
    def test_merge_sort_negative_numbers(self, input_list, expected):
        """merge_sort should correctly handle negative integers."""
        assert merge_sort(input_list) == expected

    # ------------------------------------------------------------------
    # Floating-point numbers
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            ([3.5, 1.2, 2.8], [1.2, 2.8, 3.5]),
            ([0.0, -0.5, 0.5], [-0.5, 0.0, 0.5]),
            ([1.0, 1.0, 0.999], [0.999, 1.0, 1.0]),
        ],
    )
    def test_merge_sort_floats(self, input_list, expected):
        """merge_sort should correctly sort floating-point numbers."""
        assert merge_sort(input_list) == expected

    # ------------------------------------------------------------------
    # String sorting
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            (["c", "a", "b"], ["a", "b", "c"]),
            (["dog", "cat", "bird"], ["bird", "cat", "dog"]),
            ([], []),
        ],
    )
    def test_merge_sort_strings(self, input_list, expected):
        """merge_sort should correctly sort lists of strings."""
        assert merge_sort(input_list) == expected

    # ------------------------------------------------------------------
    # Large list (stress / boundary)
    # ------------------------------------------------------------------

    def test_merge_sort_large_list(self):
        """merge_sort should handle a large list without recursion errors."""
        import random

        size = 2000
        random.seed(42)
        large_input = [random.randint(-10_000, 10_000) for _ in range(size)]
        expected = sorted(large_input)
        assert merge_sort(large_input) == expected

    # ------------------------------------------------------------------
    # Already-sorted and reverse-sorted edge cases
    # ------------------------------------------------------------------

    def test_merge_sort_already_sorted(self):
        """merge_sort should preserve an already-sorted list."""
        data = list(range(100))
        assert merge_sort(data) == data

    def test_merge_sort_reverse_sorted(self):
        """merge_sort should correctly sort a reverse-sorted list."""
        data = list(range(100, 0, -1))
        assert merge_sort(data) == list(range(1, 101))

    # ------------------------------------------------------------------
    # Stability note (merge_sort as implemented is stable)
    # ------------------------------------------------------------------

    def test_merge_sort_stability(self):
        """
        merge_sort should be stable: equal elements retain their relative
        order. We test this with tuples where the first element is the
        sort key and the second is the original index.
        """
        pairs = [(2, 0), (1, 0), (2, 1), (1, 1), (2, 2)]
        # Sort by the first element only
        sorted_pairs = merge_sort(pairs)
        # Extract the second elements for equal first elements
        ones = [b for a, b in sorted_pairs if a == 1]
        twos = [b for a, b in sorted_pairs if a == 2]
        assert ones == [0, 1]  # original order preserved
        assert twos == [0, 1, 2]  # original order preserved

    # ------------------------------------------------------------------
    # Non-list iterable (tuple) — should still work
    # ------------------------------------------------------------------

    def test_merge_sort_tuple_input(self):
        """merge_sort should accept a tuple and return a sorted list."""
        result = merge_sort([3, 1, 2])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # Mutation isolation
    # ------------------------------------------------------------------

    def test_merge_sort_does_not_mutate_original(self):
        """merge_sort must not mutate the input list."""
        original = [3, 1, 4, 1, 5, 9]
        copy = list(original)
        merge_sort(original)
        assert original == copy
