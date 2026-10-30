"""Tests for the insertion sort implementation in sorts/insertion_sort.py."""

import inspect
import random
import runpy
import string
from unittest.mock import patch

import pytest

from sorts.insertion_sort import insertion_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_module_as_main():
    """Execute the insertion_sort source file with __name__ == '__main__'.

    ``runpy.run_module("sorts.insertion_sort")`` would require the ``sorts``
    package to be importable by name, which is not guaranteed in the test
    environment.  Resolving the module file from the already imported function
    avoids that import-by-name resolution entirely.
    """
    path = inspect.getsourcefile(insertion_sort)
    assert path is not None, "could not locate the insertion_sort source file"
    runpy.run_path(path, run_name="__main__")


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_insertion_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_insertion_sort_returns_same_object():
    """The function sorts in place and returns the very same collection object."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_insertion_sort_matches_builtin_sorted_on_random_ints():
    """Sorting a random sample of ints matches Python's built-in sorted()."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_insertion_sort_matches_builtin_sorted_on_random_strings():
    """Sorting a random list of alphanumeric characters matches sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


def test_insertion_sort_works_on_mutable_sequence_subclass():
    """Any MutableSequence (e.g. a list subclass) is supported."""
    class MyList(list):
        pass

    collection = MyList([4, 2, 9, 1])
    result = insertion_sort(collection)
    assert isinstance(result, MyList)
    assert result == [1, 2, 4, 9]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                       # empty collection
        ([42], [42]),                   # single element
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),   # already sorted
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),   # reverse sorted
        ([2, 2, 2, 2], [2, 2, 2, 2]),   # all duplicates
        ([0, 0, -1, 0], [-1, 0, 0, 0]),  # duplicates with negatives
        ([-2, -5, -45], [-45, -5, -2]),  # all negative
        ([3, -1, 0, -1, 3], [-1, -1, 0, 3, 3]),  # mixed signs
        (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),  # strings
        ([1.5, -0.5, 2.25], [-0.5, 1.5, 2.25]),  # floats
    ],
)
def test_insertion_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, sorted, reversed, duplicates) sort correctly."""
    assert insertion_sort(collection) == expected


def test_insertion_sort_empty_list_is_unchanged():
    """An empty list stays empty and is returned as-is."""
    collection = []
    result = insertion_sort(collection)
    assert result == []
    assert result is collection


def test_insertion_sort_single_element_is_unchanged():
    """A one-element list is returned unchanged."""
    collection = [7]
    assert insertion_sort(collection) == [7]


def test_insertion_sort_does_not_mutate_length():
    """Sorting preserves the number of elements in the collection."""
    collection = [5, 3, 8, 1, 9, 2]
    original_len = len(collection)
    insertion_sort(collection)
    assert len(collection) == original_len


def test_insertion_sort_is_idempotent():
    """Sorting an already sorted collection leaves it unchanged."""
    collection = [1, 2, 3, 4, 5]
    insertion_sort(collection)
    assert collection == [1, 2, 3, 4, 5]


def test_insertion_sort_uses_only_less_than_comparisons():
    """The algorithm relies solely on ``__lt__`` for ordering decisions."""
    class Tracked:
        def __init__(self, value):
            self.value = value
            self.lt_calls = 0

        def __lt__(self, other):
            self.lt_calls += 1
            return self.value < other.value

        def __eq__(self, other):
            return self.value == other.value

        def __repr__(self):
            return f"Tracked({self.value})"

    items = [Tracked(v) for v in (3, 1, 2)]
    result = insertion_sort(items)
    assert [item.value for item in result] == [1, 2, 3]
    assert sum(item.lt_calls for item in items) > 0


def test_insertion_sort_raises_on_immutable_sequence():
    """Passing an immutable sequence (tuple) fails because assignment is required."""
    with pytest.raises(TypeError):
        insertion_sort((3, 1, 2))


# ---------------------------------------------------------------------------
# __main__ block behaviour
# ---------------------------------------------------------------------------


def test_main_block_sorts_user_input(capsys):
    """Running the module as a script sorts comma-separated numbers and prints them."""
    with patch("builtins.input", return_value="3,1,2"):
        _run_module_as_main()

    captured = capsys.readouterr()
    assert "insertion_sort(unsorted) = [1, 2, 3]" in captured.out


def test_main_block_strips_whitespace_from_input(capsys):
    """Surrounding whitespace in the user input is stripped before parsing."""
    with patch("builtins.input", return_value="  10, -4 ,7  "):
        _run_module_as_main()

    captured = capsys.readouterr()
    assert "insertion_sort(unsorted) = [-4, 7, 10]" in captured.out
