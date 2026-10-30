"""Tests for the insertion_sort module.

The module under test is imported dynamically so that the test file works
regardless of whether the ``sorts`` directory is a package.
"""

import importlib.util
import random
import string
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by file path (no package assumptions).
# ---------------------------------------------------------------------------
_SOURCE_PATH = Path(__file__).resolve().parent / "insertion_sort.py"
if not _SOURCE_PATH.exists():  # pragma: no cover - fallback for odd layouts
    _SOURCE_PATH = Path(
        r"D:\Documents\Study\Projects\algorithms-bench\sorts\insertion_sort.py"
    )

_spec = importlib.util.spec_from_file_location("insertion_sort_under_test", _SOURCE_PATH)
insertion_sort_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = insertion_sort_module
_spec.loader.exec_module(insertion_sort_module)

insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------
def test_sorts_unsorted_integers():
    """A typical unsorted list of integers is sorted in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_sorts_already_sorted_list():
    """An already sorted list is returned unchanged."""
    data = [1, 2, 3, 4, 5]
    assert insertion_sort(data) == [1, 2, 3, 4, 5]


def test_sorts_reverse_sorted_list():
    """A reverse sorted list is fully reversed into ascending order."""
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_sorts_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same object."""
    data = [3, 1, 2]
    result = insertion_sort(data)
    assert result is data
    assert data == [1, 2, 3]


def test_works_with_mutable_sequence_other_than_list():
    """Any MutableSequence (e.g. a bytearray-like custom sequence) works."""
    from collections import UserList

    data = UserList([4, 2, 9, 1])
    result = insertion_sort(data)
    assert list(result) == [1, 2, 4, 9]
    assert result is data


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------
def test_empty_collection():
    """An empty collection is returned unchanged."""
    data = []
    assert insertion_sort(data) == []
    assert data == []


def test_single_element_collection():
    """A single element collection is returned unchanged."""
    assert insertion_sort([42]) == [42]


def test_two_element_collection_unsorted():
    """The smallest non-trivial case (two elements) is sorted."""
    assert insertion_sort([2, 1]) == [1, 2]


def test_two_element_collection_sorted():
    """A sorted two element collection stays sorted."""
    assert insertion_sort([1, 2]) == [1, 2]


def test_all_elements_equal():
    """A collection of identical elements is left as-is."""
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_negative_numbers():
    """Negative numbers are sorted correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_duplicates_preserved():
    """Duplicate values are all preserved after sorting."""
    result = insertion_sort([3, 1, 3, 2, 1, 3])
    assert result == [1, 1, 2, 3, 3, 3]
    assert len(result) == 6


def test_large_random_integers_matches_sorted():
    """A large random integer collection matches Python's built-in sorted."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_large_random_strings_matches_sorted():
    """A large random string collection matches Python's built-in sorted."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


# ---------------------------------------------------------------------------
# Parametrized equivalence tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection",
    [
        [],
        [1],
        [1, 2],
        [2, 1],
        [0, 5, 3, 2, 2],
        [-2, -5, -45],
        [10, -10, 0, 5, -5],
        [1, 1, 1, 1, 1],
        list(range(20)),
        list(range(20, 0, -1)),
        ["d", "a", "b", "e", "c"],
        ["z"],
        ["b", "a", "b", "a"],
    ],
)
def test_matches_builtin_sorted(collection):
    """insertion_sort output always equals Python's built-in sorted output."""
    original = list(collection)
    result = insertion_sort(collection)
    assert result == sorted(original)
    # The input list itself must have been mutated into sorted order.
    assert collection == sorted(original)


@pytest.mark.parametrize(
    "collection",
    [
        [3, 1, 2],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [2, 2, 1, 1],
    ],
)
def test_result_is_non_decreasing(collection):
    """Every adjacent pair in the result satisfies a <= b."""
    result = insertion_sort(collection)
    for left, right in zip(result, result[1:]):
        assert left <= right


def test_does_not_change_length():
    """Sorting never changes the number of elements."""
    data = [9, 3, 7, 1, 3]
    insertion_sort(data)
    assert len(data) == 5


def test_does_not_add_or_remove_elements():
    """Sorting preserves the multiset of elements."""
    data = [4, 4, 1, 9, 0, 4]
    insertion_sort(data)
    assert sorted(data) == [0, 1, 4, 4, 4, 9]