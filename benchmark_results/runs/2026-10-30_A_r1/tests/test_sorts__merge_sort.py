import pytest

from merge_sort import merge_sort


class TestMergeSort:
    """Tests for the merge_sort function."""

    @pytest.mark.parametrize(
        "unsorted, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([3], [3]),
            ([], []),
            ([-2, -5, -45], [-45, -5, -2]),
            ([10, -1, 0, 7, -3], [-3, -1, 0, 7, 10]),
            ([1, 1, 1, 1], [1, 1, 1, 1]),
        ],
    )
    def test_merge_sort_sorts_various_inputs(self, unsorted, expected):
        """Test that merge_sort correctly sorts assorted input lists."""
        assert merge_sort(unsorted) == expected

    def test_merge_sort_does_not_mutate_original_list(self):
        """Test that the original list is not modified in place."""
        original = [3, 1, 2]
        merge_sort(original)
        assert original == [3, 1, 2]

    def test_merge_sort_single_element(self):
        """Test that a single-element list is returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_merge_sort_empty_list(self):
        """Test that an empty list is returned unchanged."""
        assert merge_sort([]) == []

    def test_merge_sort_large_list(self):
        """Test sorting a larger list to exercise recursion depth."""
        unsorted = list(range(1000, 0, -1))
        assert merge_sort(unsorted) == list(range(1, 1001))

    def test_merge_sort_duplicates(self):
        """Test that duplicate values are preserved and sorted correctly."""
        assert merge_sort([4, 2, 4, 1, 2, 1]) == [1, 1, 2, 2, 4, 4]

    def test_merge_sort_mixed_types_comparable(self):
        """Test sorting floats and integers together (comparable types)."""
        assert merge_sort([2.5, 1, 3.0, 2]) == [1, 2, 2.5, 3.0]