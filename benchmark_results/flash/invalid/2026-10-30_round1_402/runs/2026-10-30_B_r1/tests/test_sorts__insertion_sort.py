"""Tests for the insertion sort implementation in sorts/insertion_sort.py."""

import random
import string
from unittest.mock import patch

import pytest

from sorts.insertion_sort import insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_insertion_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_insertion_sort_returns_same_object():
    """The function sorts in place and returns the very same collection."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_insertion_sort_sorts_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_insertion_sort_sorts_floats():
    """Floats are sorted in ascending order."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_insertion_sort_matches_builtin_sorted_on_random_data():
    """Random integer data produces the same result as the builtin sorted()."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


def test_insertion_sort_matches_builtin_sorted_on_random_strings():
    """Random alphanumeric strings produce the same result as sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


def test_insertion_sort_empty_collection():
    """An empty collection is returned unchanged."""
    collection = []
    assert insertion_sort(collection) == []
    assert collection == []


def test_insertion_sort_single_element():
    """A single-element collection is returned unchanged."""
    assert insertion_sort([42]) == [42]


def test_insertion_sort_two_elements_already_sorted():
    """An already sorted two-element collection stays sorted."""
    assert insertion_sort([1, 2]) == [1, 2]


def test_insertion_sort_two_elements_reversed():
    """A reversed two-element collection is swapped."""
    assert insertion_sort([2, 1]) == [1, 2]


def test_insertion_sort_already_sorted():
    """An already sorted collection is left untouched."""
    collection = [1, 2, 3, 4, 5]
    assert insertion_sort(collection) == [1, 2, 3, 4, 5]


def test_insertion_sort_reverse_sorted():
    """A reverse sorted collection is fully reversed."""
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_insertion_sort_all_equal_elements():
    """A collection of identical elements is unchanged."""
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_insertion_sort_negative_numbers():
    """Negative numbers are sorted correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_insertion_sort_duplicates_preserved():
    """Duplicate values are all preserved in the output."""
    result = insertion_sort([3, 1, 3, 2, 1])
    assert result == [1, 1, 2, 3, 3]
    assert len(result) == 5


def test_insertion_sort_does_not_change_length():
    """Sorting never adds or removes elements."""
    collection = [9, 0, 5, 5, -3, 12]
    original_length = len(collection)
    insertion_sort(collection)
    assert len(collection) == original_length


def test_insertion_sort_works_with_mutable_sequence_subclass():
    """A list subclass (MutableSequence) is sorted in place."""
    class MyList(list):
        pass

    collection = MyList([3, 1, 2])
    result = insertion_sort(collection)
    assert result is collection
    assert list(collection) == [1, 2, 3]


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([1], [1]),
        ([2, 1], [1, 2]),
        ([1, 2], [1, 2]),
        ([3, 3, 3], [3, 3, 3]),
        ([0, -1, 1], [-1, 0, 1]),
        (["b", "a"], ["a", "b"]),
        ([1.5, 0.5, 2.5], [0.5, 1.5, 2.5]),
        (list(range(10, 0, -1)), list(range(1, 11))),
    ],
)
def test_insertion_sort_parametrized(collection, expected):
    """Parametrized check that insertion_sort matches the expected ordering."""
    assert insertion_sort(collection) == expected


@pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 50])
def test_insertion_sort_random_sizes(size):
    """Random collections of various sizes are sorted like sorted()."""
    rng = random.Random(size)
    collection = [rng.randint(-100, 100) for _ in range(size)]
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


# ---------------------------------------------------------------------------
# Isolation / interaction tests
# ---------------------------------------------------------------------------


def test_insertion_sort_uses_only_lt_comparisons():
    """The algorithm relies solely on the ``<`` operator of the elements."""
    class Tracked:
        def __init__(self, value):
            self.value = value
            self.lt_calls = 0

        def __lt__(self, other):
            self.lt_calls += 1
            return self.value < other.value

        def __repr__(self):
            return f"Tracked({self.value})"

    items = [Tracked(3), Tracked(1), Tracked(2)]
    insertion_sort(items)
    assert [item.value for item in items] == [1, 2, 3]
    assert sum(item.lt_calls for item in items) > 0


def test_insertion_sort_does_not_touch_input_function():
    """The module-level ``input`` builtin is not called by insertion_sort."""
    with patch("builtins.input", side_effect=AssertionError("input called")):
        assert insertion_sort([2, 1]) == [1, 2]


def test_insertion_sort_does_not_print(capsys):
    """insertion_sort produces no stdout output."""
    insertion_sort([3, 2, 1])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""