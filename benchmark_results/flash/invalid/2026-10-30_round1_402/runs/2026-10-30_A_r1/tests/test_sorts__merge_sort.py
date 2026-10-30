"""Tests for the merge sort implementation in sorts/merge_sort.py."""

import random
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

import merge_sort as merge_sort_module
from merge_sort import merge_sort

# Resolve the on-disk location of the module so that its __main__ block can be
# executed without depending on how the package is mounted on sys.path.
MODULE_FILE = Path(merge_sort_module.__file__)


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
            ([1.5, -0.5, 2.25, 0.0], [-0.5, 0.0, 1.5, 2.25]),
            (["banana", "apple", "cherry"], ["apple", "banana", "cherry"]),
        ],
    )
    def test_sorts_various_collections(self, collection, expected):
        """merge_sort returns the collection in ascending order."""
        assert merge_sort(collection) == expected

    def test_returns_list_type(self):
        """merge_sort always returns a list instance."""
        result = merge_sort([3, 1, 2])
        assert isinstance(result, list)

    def test_does_not_mutate_input(self):
        """merge_sort must not modify the caller's original list."""
        original = [4, 2, 9, 1]
        snapshot = list(original)
        merge_sort(original)
        assert original == snapshot

    def test_matches_builtin_sorted_on_random_data(self):
        """merge_sort output matches Python's built-in sorted for random data."""
        rng = random.Random(1234)
        data = [rng.randint(-1000, 1000) for _ in range(200)]
        assert merge_sort(data) == sorted(data)


class TestMergeSortEdgeCases:
    """Boundary and edge cases for merge_sort."""

    def test_empty_list(self):
        """An empty list is returned unchanged."""
        assert merge_sort([]) == []

    def test_single_element(self):
        """A single-element list is returned unchanged."""
        assert merge_sort([42]) == [42]

    def test_two_elements_unsorted(self):
        """A two-element unsorted list is swapped correctly."""
        assert merge_sort([2, 1]) == [1, 2]

    def test_two_elements_sorted(self):
        """A two-element already-sorted list stays sorted."""
        assert merge_sort([1, 2]) == [1, 2]

    def test_all_duplicates(self):
        """A list of identical values is preserved."""
        assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_negative_and_zero(self):
        """Negative numbers and zero are ordered correctly."""
        assert merge_sort([0, -1, -100, 5]) == [-100, -1, 0, 5]

    def test_large_list(self):
        """A large list is sorted correctly (exercises recursion depth)."""
        data = list(range(1000, 0, -1))
        assert merge_sort(data) == list(range(1, 1001))

    def test_tuple_input_is_not_supported(self):
        """Sequences without .pop() (e.g. tuples) are rejected by merge()."""
        with pytest.raises(AttributeError):
            merge_sort((3, 1, 2))

    def test_converted_tuple_is_sorted(self):
        """Converting a tuple to a list yields a correctly sorted list."""
        result = merge_sort(list((3, 1, 2)))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_returns_same_object_for_single_element(self):
        """For length <= 1 the original object is returned as-is."""
        data = [99]
        assert merge_sort(data) is data

    def test_returns_same_object_for_empty(self):
        """For an empty list the original object is returned as-is."""
        data = []
        assert merge_sort(data) is data


class TestMergeSortMainBlock:
    """Tests for the interactive __main__ behaviour via module execution."""

    def test_main_block_sorts_valid_input(self, capsys):
        """Running the module as a script sorts comma-separated integers."""
        with patch("builtins.input", return_value="3,1,2"):
            runpy.run_path(str(MODULE_FILE), run_name="__main__")
        captured = capsys.readouterr()
        assert "1,2,3" in captured.out

    def test_main_block_handles_invalid_input(self, capsys):
        """Invalid input prints an error message instead of raising."""
        with patch("builtins.input", return_value="a,b,c"):
            runpy.run_path(str(MODULE_FILE), run_name="__main__")
        captured = capsys.readouterr()
        assert "Invalid input" in captured.out
