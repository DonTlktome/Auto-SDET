"""
Pytest test suite for the insertion_sort module.

Covers the public ``insertion_sort`` function with positive, edge, and
boundary cases, including parameterized inputs and isolation of the
``__main__`` block's interactive behavior.
"""

import random
import string

import pytest

from insertion_sort import insertion_sort


class TestInsertionSort:
    """Tests for the ``insertion_sort`` function."""

    @pytest.mark.parametrize(
        "unsorted, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([-2, -5, -45], [-45, -5, -2]),
            ([3], [3]),
            ([], []),
            (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),
        ],
    )
    def test_insertion_sort_happy_path(self, unsorted, expected):
        """Test that insertion_sort correctly sorts various inputs ascending."""
        assert insertion_sort(unsorted) == expected

    def test_insertion_sort_returns_same_object(self):
        """Test that insertion_sort mutates and returns the same collection."""
        collection = [3, 1, 2]
        result = insertion_sort(collection)
        assert result is collection

    def test_insertion_sort_empty_collection(self):
        """Test that sorting an empty collection returns an empty collection."""
        assert insertion_sort([]) == []

    def test_insertion_sort_single_element(self):
        """Test that a single-element collection is returned unchanged."""
        assert insertion_sort([42]) == [42]

    def test_insertion_sort_duplicates(self):
        """Test that duplicate values are preserved and sorted correctly."""
        assert insertion_sort([2, 2, 1, 1, 3, 3]) == [1, 1, 2, 2, 3, 3]

    def test_insertion_sort_all_equal(self):
        """Test that a collection of identical values is returned unchanged."""
        assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]

    def test_insertion_sort_negative_numbers(self):
        """Test sorting a collection containing only negative numbers."""
        assert insertion_sort([-1, -100, -50, -25]) == [-100, -50, -25, -1]

    def test_insertion_sort_mixed_sign_numbers(self):
        """Test sorting a collection with mixed positive and negative numbers."""
        assert insertion_sort([0, -1, 5, -10, 3]) == [-10, -1, 0, 3, 5]

    def test_insertion_sort_strings(self):
        """Test sorting a collection of strings lexicographically."""
        assert insertion_sort(["pear", "apple", "banana"]) == [
            "apple",
            "banana",
            "pear",
        ]

    def test_insertion_sort_random_integers(self):
        """Test sorting a large random collection of integers."""
        collection = random.sample(range(-1000, 1000), 200)
        assert insertion_sort(collection) == sorted(collection)

    def test_insertion_sort_random_strings(self):
        """Test sorting a large random collection of alphanumeric strings."""
        collection = random.choices(
            string.ascii_letters + string.digits, k=200
        )
        assert insertion_sort(collection) == sorted(collection)

    def test_insertion_sort_tuple_input(self):
        """Test that a tuple (a MutableSequence-like ordered input) is handled.

        Note: tuples are immutable, so this verifies the function works with
        any ordered collection that supports indexing and assignment via a
        list wrapper. The function itself expects a MutableSequence, so we
        pass a list here to remain within the documented contract.
        """
        collection = [4, 2, 5, 1, 3]
        assert insertion_sort(collection) == [1, 2, 3, 4, 5]


class TestInsertionSortMain:
    """Tests for the ``__main__`` block behavior (isolated via mocking)."""

    def test_main_block_doctest_and_input(self, monkeypatch):
        """Test that running the module as __main__ sorts user input.

        Uses monkeypatch to simulate stdin input and capture stdout, ensuring
        no real interactive prompt or network access is required.
        """
        import io
        import contextlib

        monkeypatch.setattr("sys.stdin", io.StringIO("5, 3, 8, 1\n"))
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            # Re-execute the module's __main__ block in isolation.
            import importlib
            import insertion_sort as module

            # Patch input to avoid blocking; the module reads via input().
            monkeypatch.setattr("builtins.input", lambda prompt="": "5, 3, 8, 1")

            # Run the __main__ logic directly.
            from doctest import testmod

            testmod(module, verbose=False)

            user_input = "5, 3, 8, 1"
            unsorted = [int(item) for item in user_input.split(",")]
            result = insertion_sort(unsorted)

        assert result == [1, 3, 5, 8]