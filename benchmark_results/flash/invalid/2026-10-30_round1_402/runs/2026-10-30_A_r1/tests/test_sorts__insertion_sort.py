"""Tests for the insertion sort implementation in sorts/insertion_sort.py."""

from __future__ import annotations

import random
import string
from unittest.mock import MagicMock, patch

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
    """Random alphanumeric data produces the same result as the builtin sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


def test_insertion_sort_empty_list():
    """An empty list is returned unchanged."""
    collection: list[int] = []
    assert insertion_sort(collection) == []
    assert collection == []


def test_insertion_sort_single_element():
    """A single-element list is already sorted and untouched."""
    assert insertion_sort([42]) == [42]


def test_insertion_sort_two_elements_swapped():
    """The minimal non-trivial case: two elements out of order get swapped."""
    assert insertion_sort([2, 1]) == [1, 2]


def test_insertion_sort_already_sorted():
    """An already sorted list stays sorted."""
    assert insertion_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_insertion_sort_reverse_sorted():
    """A reverse sorted list is fully reversed."""
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_insertion_sort_all_equal_elements():
    """A list of identical elements is unchanged."""
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_insertion_sort_negative_numbers():
    """Negative numbers are ordered correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_insertion_sort_duplicates_preserved():
    """Duplicate values are all retained after sorting."""
    result = insertion_sort([3, 1, 3, 2, 1])
    assert result == [1, 1, 2, 3, 3]
    assert len(result) == 5


def test_insertion_sort_works_with_mutable_sequence_other_than_list():
    """Any MutableSequence (e.g. a bytearray-like custom sequence) works."""
    from collections import UserList

    collection = UserList([3, 1, 2])
    result = insertion_sort(collection)
    assert result is collection
    assert list(result) == [1, 2, 3]


def test_insertion_sort_does_not_mutate_input_length():
    """Sorting never changes the number of elements."""
    collection = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    insertion_sort(collection)
    assert len(collection) == 10


# ---------------------------------------------------------------------------
# Parametrized coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2]),
        ([2, 1], [1, 2]),
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([-2, -5, -45], [-45, -5, -2]),
        ([1, 1, 1], [1, 1, 1]),
        ([3, -1, 0, -1, 3], [-1, -1, 0, 3, 3]),
        (["b", "a"], ["a", "b"]),
        ([True, False], [False, True]),
    ],
)
def test_insertion_sort_parametrized(collection, expected):
    """insertion_sort returns the ascending ordering for many input shapes."""
    assert insertion_sort(list(collection)) == expected


@pytest.mark.parametrize("size", [0, 1, 2, 3, 10, 50])
def test_insertion_sort_matches_sorted_for_various_sizes(size):
    """For a range of sizes the output equals the builtin sorted() output."""
    rng = random.Random(size)
    collection = [rng.randint(-100, 100) for _ in range(size)]
    assert insertion_sort(collection) == sorted(collection)


# ---------------------------------------------------------------------------
# Isolation / mock-based tests
# ---------------------------------------------------------------------------


def test_insertion_sort_uses_only_indexing_and_assignment():
    """The algorithm interacts with the collection via __getitem__/__setitem__.

    A MagicMock-based MutableSequence stand-in verifies that the function
    performs in-place writes rather than building a new list.
    """
    data = [3, 1, 2]
    sequence = MagicMock()
    sequence.__len__.return_value = len(data)
    sequence.__getitem__.side_effect = lambda i: data[i]
    sequence.__setitem__.side_effect = lambda i, v: data.__setitem__(i, v)

    result = insertion_sort(sequence)

    assert result is sequence
    assert data == [1, 2, 3]
    assert sequence.__setitem__.called


def test_insertion_sort_does_not_call_sorted_or_sort():
    """The implementation must not delegate to Python's builtin sorting."""
    with patch("builtins.sorted", side_effect=AssertionError("sorted() used")):
        assert insertion_sort([3, 1, 2]) == [1, 2, 3]


def test_insertion_sort_does_not_require_network_or_io():
    """Sorting performs no file or network access (open is never called)."""
    with patch("builtins.open", side_effect=AssertionError("open() used")):
        assert insertion_sort([4, 2, 9, 1]) == [1, 2, 4, 9]


def test_insertion_sort_compares_with_lt_only():
    """Only the ``<`` operator is needed from the Comparable protocol."""

    class OnlyLt:
        def __init__(self, value: int) -> None:
            self.value = value

        def __lt__(self, other: "OnlyLt", /) -> bool:
            return self.value < other.value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, OnlyLt) and self.value == other.value

        def __repr__(self) -> str:  # pragma: no cover - debugging aid
            return f"OnlyLt({self.value})"

    collection = [OnlyLt(3), OnlyLt(1), OnlyLt(2)]
    result = insertion_sort(collection)
    assert [item.value for item in result] == [1, 2, 3]