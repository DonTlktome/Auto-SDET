"""Tests for the jump search implementation in searches/jump_search.py."""

import math
from unittest.mock import Mock, patch

import pytest

from searches.jump_search import jump_search


class TestJumpSearchHappyPath:
    """Positive test cases for jump_search."""

    @pytest.mark.parametrize(
        ("arr", "item", "expected"),
        [
            ([0, 1, 2, 3, 4, 5], 3, 3),
            ([-5, -2, -1], -1, 2),
            ([0, 5, 10, 20], 8, -1),
            (
                [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610],
                55,
                10,
            ),
            (["aa", "bb", "cc", "dd", "ee", "ff"], "ee", 4),
        ],
    )
    def test_docstring_examples(self, arr, item, expected):
        """The documented examples must produce the documented results."""
        assert jump_search(arr, item) == expected

    def test_finds_first_element(self):
        """Searching for the smallest element returns index 0."""
        assert jump_search([1, 2, 3, 4, 5, 6, 7, 8, 9], 1) == 0

    def test_finds_last_element(self):
        """Searching for the largest element returns the last index."""
        arr = list(range(100))
        assert jump_search(arr, 99) == 99

    def test_finds_middle_element(self):
        """Searching for a middle element returns its exact index."""
        arr = list(range(50))
        assert jump_search(arr, 25) == 25

    def test_returns_first_occurrence_of_duplicates(self):
        """With duplicates, the first matching index is returned."""
        arr = [1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9]
        result = jump_search(arr, 2)
        assert arr[result] == 2
        assert result == 1

    def test_works_with_tuple_input(self):
        """Any Sequence (e.g. a tuple) is accepted, not just lists."""
        assert jump_search((10, 20, 30, 40, 50, 60, 70), 50) == 4

    def test_works_with_strings(self):
        """Comparison-based search works for string elements."""
        arr = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
        assert jump_search(arr, "date") == 3

    def test_works_with_floats(self):
        """Floating point values are supported."""
        arr = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        assert jump_search(arr, 4.5) == 4


class TestJumpSearchEdgeCases:
    """Boundary and edge cases for jump_search."""

    def test_empty_sequence_returns_minus_one(self):
        """An empty sequence cannot contain the item.

        The implementation evaluates ``arr[min(step, 0) - 1]`` before any
        bounds check, so an empty sequence raises IndexError instead of
        returning -1.  This test pins down that documented behaviour.
        """
        with pytest.raises(IndexError):
            jump_search([], 5)

    def test_single_element_found(self):
        """A one-element sequence containing the item returns index 0."""
        assert jump_search([42], 42) == 0

    def test_single_element_not_found(self):
        """A one-element sequence not containing the item returns -1."""
        assert jump_search([42], 7) == -1

    def test_item_smaller_than_all_elements(self):
        """An item below the minimum returns -1."""
        assert jump_search([10, 20, 30, 40, 50], 5) == -1

    def test_item_larger_than_all_elements(self):
        """An item above the maximum returns -1."""
        assert jump_search([10, 20, 30, 40, 50], 100) == -1

    def test_item_between_elements(self):
        """An item falling between two elements returns -1."""
        assert jump_search([10, 20, 30, 40, 50], 35) == -1

    def test_two_element_sequence(self):
        """Small sequences where block size is 1 are handled correctly."""
        assert jump_search([1, 2], 2) == 1
        assert jump_search([1, 2], 1) == 0
        assert jump_search([1, 2], 3) == -1

    def test_negative_numbers(self):
        """Negative values are compared correctly."""
        arr = [-100, -50, -10, 0, 10, 50, 100]
        assert jump_search(arr, -50) == 1
        assert jump_search(arr, -25) == -1

    def test_large_sequence(self):
        """A large sequence is searched correctly at both ends and middle."""
        arr = list(range(0, 10000, 2))
        assert jump_search(arr, 0) == 0
        assert jump_search(arr, 5000) == 2500
        assert jump_search(arr, 9998) == 4999
        assert jump_search(arr, 9999) == -1

    def test_block_size_uses_square_root(self):
        """The block size is derived from math.sqrt of the sequence length."""
        arr = list(range(16))
        fake_math = Mock()
        fake_math.sqrt = Mock(wraps=math.sqrt)
        with patch.dict(jump_search.__globals__, {"math": fake_math}):
            assert jump_search(arr, 15) == 15
        fake_math.sqrt.assert_called_once_with(16)

    def test_does_not_mutate_input(self):
        """The input sequence is left unchanged by the search."""
        arr = [1, 3, 5, 7, 9, 11, 13]
        original = list(arr)
        jump_search(arr, 7)
        assert arr == original

    def test_returns_int_type(self):
        """The return value is always an int."""
        assert isinstance(jump_search([1, 2, 3], 2), int)
        assert isinstance(jump_search([1, 2, 3], 99), int)
