"""Tests for the jump search implementation in searches/jump_search.py."""

import math
from unittest.mock import patch

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
        assert result == 1
        assert arr[result] == 2

    def test_works_with_tuple_input(self):
        """Any Sequence (e.g. a tuple) is accepted, not just lists."""
        assert jump_search((1, 3, 5, 7, 9, 11, 13, 15, 17), 11) == 5

    def test_works_with_strings(self):
        """Comparable non-numeric types such as strings are supported."""
        assert jump_search(["apple", "banana", "cherry", "date"], "cherry") == 2

    def test_works_with_floats(self):
        """Floating point values are supported."""
        arr = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        assert jump_search(arr, 6.5) == 6


class TestJumpSearchEdgeCases:
    """Boundary and edge cases for jump_search."""

    @pytest.mark.xfail(
        raises=IndexError,
        reason="jump_search currently raises IndexError for empty sequences",
    )
    def test_empty_sequence_returns_minus_one(self):
        """An empty sequence cannot contain the item, so -1 is returned."""
        assert jump_search([], 5) == -1

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

    def test_all_identical_elements(self):
        """A sequence of identical values returns index 0 when matched."""
        assert jump_search([7, 7, 7, 7, 7, 7, 7, 7, 7], 7) == 0

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

    def test_result_index_is_valid(self):
        """Whenever a non-negative index is returned it points at the item."""
        arr = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        for item in arr:
            idx = jump_search(arr, item)
            assert idx >= 0
            assert arr[idx] == item


class TestJumpSearchInternals:
    """Tests that verify the algorithm's use of math.sqrt via mocking."""

    def test_block_size_uses_square_root(self):
        """The block size is derived from math.sqrt of the sequence length."""
        arr = list(range(16))
        math_module = jump_search.__globals__["math"]
        with patch.object(math_module, "sqrt", wraps=math.sqrt) as mock_sqrt:
            assert jump_search(arr, 15) == 15
        mock_sqrt.assert_called_once_with(16)

    def test_works_when_sqrt_is_mocked(self):
        """Behaviour is unchanged when math.sqrt is replaced by a stub."""
        arr = list(range(9))
        math_module = jump_search.__globals__["math"]
        with patch.object(math_module, "sqrt", return_value=3.0):
            assert jump_search(arr, 4) == 4
            assert jump_search(arr, 100) == -1
