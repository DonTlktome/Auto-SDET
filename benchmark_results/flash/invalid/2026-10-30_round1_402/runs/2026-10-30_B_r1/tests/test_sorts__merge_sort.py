"""Tests for the merge sort implementation in sorts/merge_sort.py."""

import random
import sys
from pathlib import Path

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_sort import merge_sort  # noqa: E402


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_merge_sort_sorts_unsorted_list():
    """A typical unsorted list of integers is returned in ascending order."""
    assert merge_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_merge_sort_returns_list_type():
    """The function returns a list instance for list input."""
    result = merge_sort([3, 1, 2])
    assert isinstance(result, list)


def test_merge_sort_does_not_mutate_input():
    """The original collection is left untouched (slicing-based recursion)."""
    original = [4, 2, 9, 1]
    snapshot = list(original)
    merge_sort(original)
    assert original == snapshot


def test_merge_sort_sorts_strings():
    """Comparable non-numeric items (strings) are sorted lexicographically."""
    assert merge_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_merge_sort_sorts_floats():
    """Floating point values are sorted correctly."""
    assert merge_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_merge_sort_sorts_tuples_of_comparable_items():
    """Tuples are compared element-wise and sorted accordingly."""
    assert merge_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),                 # empty list
        ([42], [42]),             # single element
        ([1, 2], [1, 2]),         # already sorted, two elements
        ([2, 1], [1, 2]),         # reversed, two elements
        ([5, 5, 5, 5], [5, 5, 5, 5]),  # all duplicates
        ([-2, -5, -45], [-45, -5, -2]),  # all negative
        ([0], [0]),               # single zero
        ([3, 1, 2], [1, 2, 3]),   # odd length
        ([4, 3, 2, 1], [1, 2, 3, 4]),  # even length, reversed
    ],
)
def test_merge_sort_edge_cases(collection, expected):
    """Boundary inputs (empty, singleton, duplicates, negatives) sort correctly."""
    assert merge_sort(collection) == expected


def test_merge_sort_empty_list_returns_same_object():
    """For length <= 1 the input collection is returned as-is (identity)."""
    empty = []
    assert merge_sort(empty) is empty


def test_merge_sort_single_element_returns_same_object():
    """A one-element list is returned without copying."""
    single = [7]
    assert merge_sort(single) is single


def test_merge_sort_large_reversed_range():
    """A large reversed range is fully sorted (exercises deep recursion)."""
    data = list(range(500, 0, -1))
    assert merge_sort(data) == list(range(1, 501))


def test_merge_sort_random_data_matches_builtin_sorted():
    """Randomized data produces the same result as Python's built-in sorted."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert merge_sort(data) == sorted(data)


def test_merge_sort_is_stable_for_equal_keys():
    """Merge sort is stable: equal keys keep their original relative order."""
    data = [(1, "first"), (0, "a"), (1, "second"), (0, "b")]
    result = merge_sort(data)
    assert result == [(0, "a"), (0, "b"), (1, "first"), (1, "second")]


def test_merge_sort_raises_on_incomparable_items():
    """Mixing incomparable types raises TypeError during merging."""
    with pytest.raises(TypeError):
        merge_sort([1, "two", 3])


def test_merge_sort_rejects_tuple_input():
    """A tuple input is not supported; the merge step needs a mutable list."""
    with pytest.raises(AttributeError):
        merge_sort((3, 1, 2))


def test_merge_sort_handles_list_from_tuple():
    """Converting a tuple to a list first yields a correctly sorted list."""
    result = merge_sort(list((3, 1, 2)))
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_merge_sort_rejects_string_input():
    """A raw string is not supported; the merge step needs a mutable list."""
    with pytest.raises(AttributeError):
        merge_sort("cba")


def test_merge_sort_handles_list_from_string():
    """Converting a string to a list of characters sorts them."""
    assert merge_sort(list("cba")) == ["a", "b", "c"]


def test_merge_sort_does_not_require_mockable_io():
    """Sanity check: sorting performs no I/O (no input/print side effects)."""
    from unittest.mock import patch

    with patch("builtins.input") as mock_input, patch("builtins.print") as mock_print:
        merge_sort([3, 2, 1])
    mock_input.assert_not_called()
    mock_print.assert_not_called()
