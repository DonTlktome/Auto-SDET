"""Tests for the merge sort implementation in sorts/merge_sort.py."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_sort import merge_sort  # noqa: E402


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_merge_sort_sorts_unsorted_list():
    """A typical unsorted list of integers is returned in ascending order."""
    assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_merge_sort_returns_list_type():
    """The function returns a list instance."""
    result = merge_sort([3, 1, 2])
    assert isinstance(result, list)


def test_merge_sort_does_not_mutate_input():
    """The original collection passed in is left untouched."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    merge_sort(original)
    assert original == snapshot


def test_merge_sort_handles_duplicates():
    """Duplicate values are all preserved in the output."""
    assert merge_sort([5, 1, 5, 1, 5]) == [1, 1, 5, 5, 5]


def test_merge_sort_handles_negative_numbers():
    """Negative numbers are sorted correctly relative to positives."""
    assert merge_sort([-2, -5, -45]) == [-45, -5, -2]
    assert merge_sort([3, -1, 0, -7, 2]) == [-7, -1, 0, 2, 3]


def test_merge_sort_handles_floats():
    """Floating point values are sorted in ascending order."""
    assert merge_sort([2.5, 1.1, 3.3, 0.0]) == [0.0, 1.1, 2.5, 3.3]


def test_merge_sort_handles_strings():
    """Strings are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_handles_tuples_of_comparable_items():
    """Lists of tuples are sorted by their natural ordering."""
    data = [(2, "b"), (1, "a"), (3, "c")]
    assert merge_sort(data) == [(1, "a"), (2, "b"), (3, "c")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        [],
        [42],
        [0],
        [-1],
        ["only"],
    ],
)
def test_merge_sort_trivial_inputs_returned_unchanged(collection):
    """Empty and single-element collections are returned as-is."""
    assert merge_sort(collection) == collection


def test_merge_sort_empty_list_returns_empty_list():
    """An empty list yields an empty list (not None)."""
    result = merge_sort([])
    assert result == []
    assert isinstance(result, list)


def test_merge_sort_already_sorted():
    """An already sorted list stays sorted."""
    assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_merge_sort_reverse_sorted():
    """A reverse sorted list is fully reversed."""
    assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_merge_sort_all_equal_elements():
    """A list of identical elements is returned unchanged."""
    assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_merge_sort_two_elements_unsorted():
    """The smallest non-trivial case (two elements) is swapped correctly."""
    assert merge_sort([2, 1]) == [1, 2]


def test_merge_sort_two_elements_sorted():
    """Two already ordered elements remain in place."""
    assert merge_sort([1, 2]) == [1, 2]


def test_merge_sort_odd_length_list():
    """Odd-length lists split unevenly but still sort correctly."""
    assert merge_sort([9, 1, 8, 2, 7]) == [1, 2, 7, 8, 9]


def test_merge_sort_even_length_list():
    """Even-length lists split evenly and sort correctly."""
    assert merge_sort([9, 1, 8, 2, 7, 3]) == [1, 2, 3, 7, 8, 9]


def test_merge_sort_large_list_matches_builtin_sorted():
    """A large randomized list matches Python's built-in sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(1000)]
    assert merge_sort(data) == sorted(data)


def test_merge_sort_preserves_length():
    """Sorting never adds or drops elements."""
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    assert len(merge_sort(data)) == len(data)


def test_merge_sort_is_stable_for_equal_keys():
    """Equal keys keep their original relative order (stability)."""
    data = [(1, "a"), (0, "x"), (1, "b"), (0, "y")]
    assert merge_sort(data) == [(0, "x"), (0, "y"), (1, "a"), (1, "b")]


def test_merge_sort_raises_on_incomparable_types():
    """Mixing incomparable types raises TypeError."""
    with pytest.raises(TypeError):
        merge_sort([1, "a", 2])


def test_merge_sort_accepts_tuple_input():
    """A tuple is immutable, so it must be converted to a list before sorting."""
    result = merge_sort(list((3, 1, 2)))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_merge_sort_accepts_string_input():
    """A string is immutable, so its characters must be wrapped in a list to sort."""
    assert merge_sort(list("cba")) == ["a", "b", "c"]


def test_merge_sort_does_not_require_mutable_input():
    """Callers may supply immutable sequences by materialising them as lists first."""
    assert merge_sort(list((5, 3, 1))) == [1, 3, 5]
