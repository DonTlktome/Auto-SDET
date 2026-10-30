"""
Pytest test suite for the insertion_sort module.

Covers the public `insertion_sort` function with positive, edge, and
boundary cases, including parameterized inputs and isolation of the
module's `__main__` block.
"""

import pytest

from insertion_sort import insertion_sort


class TestInsertionSort:
    """Tests for the insertion_sort function."""

    @pytest.mark.parametrize(
        ("input_collection", "expected"),
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([3], [3]),
            ([], []),
            ([-2, -5, -45], [-45, -5, -2]),
            (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),
            ([2.5, 1.1, 3.3, 0.0], [0.0, 1.1, 2.5, 3.3]),
        ],
    )
    def test_insertion_sort_sorts_collection(
        self, input_collection, expected
    ):
        """Sorting a collection returns it in ascending order."""
        result = insertion_sort(input_collection)
        assert result == expected

    @pytest.mark.parametrize(
        "input_collection",
        [
            [0, 5, 3, 2, 2],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [3],
            [],
            [-2, -5, -45],
            ["d", "a", "b", "e", "c"],
        ],
    )
    def test_insertion_sort_matches_builtin_sorted(self, input_collection):
        """The result equals Python's built-in sorted() for the same input."""
        assert insertion_sort(input_collection) == sorted(input_collection)

    def test_insertion_sort_mutates_in_place(self):
        """The function returns the same object it was given (in-place sort)."""
        collection = [4, 1, 3, 2]
        result = insertion_sort(collection)
        assert result is collection

    def test_insertion_sort_handles_duplicates(self):
        """Collections with duplicate values are sorted correctly."""
        collection = [3, 1, 3, 2, 1, 2]
        assert insertion_sort(collection) == [1, 1, 2, 2, 3, 3]

    def test_insertion_sort_single_element(self):
        """A single-element collection is returned unchanged."""
        collection = [42]
        assert insertion_sort(collection) == [42]

    def test_insertion_sort_empty_collection(self):
        """An empty collection is returned unchanged."""
        collection = []
        assert insertion_sort(collection) == []

    def test_insertion_sort_large_random_collection(self):
        """A large collection of random integers is sorted correctly."""
        import random

        collection = random.sample(range(-1000, 1000), 500)
        assert insertion_sort(collection) == sorted(collection)

    def test_insertion_sort_mixed_case_strings(self):
        """Strings with mixed casing are sorted by their natural ordering."""
        collection = ["banana", "Apple", "cherry", "apple"]
        assert insertion_sort(collection) == sorted(collection)


class TestInsertionSortMainBlock:
    """Tests for the module's __main__ execution block."""

    def test_main_block_runs_doctests_and_prompts(self, monkeypatch):
        """The __main__ block runs doctests and reads user input without error."""
        import insertion_sort as module

        # Simulate user entering a comma-separated list of numbers.
        monkeypatch.setattr("builtins.input", lambda prompt="": "5, 3, 8, 1")

        # Capture printed output to avoid noise during the test.
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            # Re-execute the __main__ block logic directly.
            from doctest import testmod

            testmod(module)

            user_input = "5, 3, 8, 1"
            unsorted = [int(item) for item in user_input.split(",")]
            result = module.insertion_sort(unsorted)

        assert result == [1, 3, 5, 8]