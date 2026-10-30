"""Tests for the jump search implementation in searches/jump_search.py."""

import inspect
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from searches.jump_search import jump_search

# Resolve the path and the module object of the implementation once, so the
# tests below never depend on the dotted module name being importable.
SOURCE_PATH = inspect.getsourcefile(jump_search)
JUMP_SEARCH_MODULE = sys.modules[jump_search.__module__]


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
        """The examples documented in the module docstring must hold."""
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
        arr = list(range(100))
        assert jump_search(arr, 50) == 50

    def test_returns_first_occurrence_of_duplicates(self):
        """With duplicates, the first matching index is returned."""
        arr = [1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9]
        result = jump_search(arr, 2)
        assert arr[result] == 2
        assert result == 1

    def test_works_with_tuple_input(self):
        """Any Sequence (e.g. a tuple) is accepted, not just lists."""
        assert jump_search((10, 20, 30, 40, 50, 60, 70, 80, 90), 70) == 6

    def test_works_with_strings(self):
        """Comparison-based search works for string sequences."""
        words = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
        assert jump_search(words, "date") == 3

    def test_works_with_floats(self):
        """Floating point values are supported."""
        arr = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        assert jump_search(arr, 6.5) == 6


class TestJumpSearchEdgeCases:
    """Boundary and edge cases for jump_search."""

    def test_empty_sequence_raises_index_error(self):
        """An empty sequence has no valid index, so IndexError is raised."""
        with pytest.raises(IndexError):
            jump_search([], 1)

    def test_single_element_found(self):
        """A one-element sequence containing the item returns 0."""
        assert jump_search([42], 42) == 0

    def test_single_element_not_found(self):
        """A one-element sequence without the item returns -1."""
        assert jump_search([42], 7) == -1

    def test_two_elements_found(self):
        """Small sequences below the block size still work."""
        assert jump_search([1, 2], 2) == 1

    def test_two_elements_not_found(self):
        """Small sequences below the block size return -1 when missing."""
        assert jump_search([1, 2], 3) == -1

    def test_item_smaller_than_all(self):
        """An item smaller than every element returns -1."""
        assert jump_search([10, 20, 30, 40, 50, 60, 70, 80, 90], 5) == -1

    def test_item_larger_than_all(self):
        """An item larger than every element returns -1."""
        assert jump_search([10, 20, 30, 40, 50, 60, 70, 80, 90], 100) == -1

    def test_item_between_elements(self):
        """An item falling between two elements returns -1."""
        assert jump_search([10, 20, 30, 40, 50, 60, 70, 80, 90], 65) == -1

    def test_all_equal_elements(self):
        """A sequence of identical values returns index 0 for that value."""
        assert jump_search([5, 5, 5, 5, 5, 5, 5, 5, 5], 5) == 0

    def test_negative_numbers(self):
        """Negative values are handled correctly."""
        arr = [-100, -50, -10, 0, 10, 50, 100, 200, 300]
        assert jump_search(arr, -50) == 1

    def test_large_sequence(self):
        """A large sequence is searched correctly at both ends."""
        arr = list(range(0, 10000, 2))
        assert jump_search(arr, 0) == 0
        assert jump_search(arr, 9998) == 4999
        assert jump_search(arr, 9999) == -1

    def test_block_size_uses_sqrt(self):
        """The block size is derived from math.sqrt of the sequence length."""
        arr = list(range(16))
        fake_math = SimpleNamespace(sqrt=Mock(wraps=math.sqrt))
        with patch.object(JUMP_SEARCH_MODULE, "math", fake_math):
            assert jump_search(arr, 15) == 15
        fake_math.sqrt.assert_called_once_with(16)

    def test_does_not_mutate_input(self):
        """The input sequence is left untouched by the search."""
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        original = list(arr)
        jump_search(arr, 4)
        assert arr == original


class TestJumpSearchMainBlock:
    """Tests for the interactive __main__ behaviour."""

    def test_main_prints_index_when_found(self, capsys):
        """Running the module as a script prints the found index."""
        import runpy

        inputs = iter(["1,2,3,4,5", "3"])
        with patch("builtins.input", lambda *_: next(inputs)):
            runpy.run_path(SOURCE_PATH, run_name="__main__")
        captured = capsys.readouterr()
        assert "Number 3 is at index 2" in captured.out

    def test_main_prints_not_found(self, capsys):
        """Running the module as a script reports a missing number."""
        import runpy

        inputs = iter(["1,2,3,4,5", "99"])
        with patch("builtins.input", lambda *_: next(inputs)):
            runpy.run_path(SOURCE_PATH, run_name="__main__")
        captured = capsys.readouterr()
        assert "Number not found!" in captured.out
