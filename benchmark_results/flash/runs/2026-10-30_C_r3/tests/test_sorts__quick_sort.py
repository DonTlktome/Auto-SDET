"""Tests for the pure Python quick sort implementation in sorts/quick_sort.py."""

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort


# Resolve the module object that actually defines ``quick_sort`` so that
# ``randrange`` can be patched on it directly.  Patching a dotted string such
# as ``"sorts.quick_sort.randrange"`` makes ``unittest.mock`` re-import the
# ``sorts`` package via ``pkgutil.resolve_name``, which fails with
# ``ModuleNotFoundError`` in this test environment.
quick_sort_module = sys.modules[quick_sort.__module__]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list():
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_list_type():
    """The function returns a list instance."""
    result = quick_sort([3, 1, 2])
    assert isinstance(result, list)


def test_quick_sort_pops_pivot_from_input():
    """The chosen pivot is popped from the caller's list, so it is mutated.

    ``quick_sort`` removes the pivot with ``collection.pop(pivot_index)``
    before partitioning, therefore the *top-level* input list loses exactly
    one element (the pivot) and keeps the relative order of the remaining
    items.  Only the **returned** list is guaranteed to be fully sorted.
    """
    original = [4, 2, 9, 1]
    snapshot = list(original)
    result = quick_sort(original)

    # The returned value is a brand new, correctly sorted list.
    assert result == [1, 2, 4, 9]
    assert result is not original

    # The input list lost exactly one element (the pivot) and kept its order.
    assert len(original) == len(snapshot) - 1
    assert any(
        snapshot[:index] + snapshot[index + 1 :] == original
        for index in range(len(snapshot))
    )


def test_quick_sort_handles_negative_numbers():
    """Negative values are ordered correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats():
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 3.75, 0.0]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings():
    """Strings are sorted lexicographically."""
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


def test_quick_sort_handles_tuples_of_comparable_items():
    """Tuples of comparable items are sorted element-wise."""
    assert quick_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([42], [42]),
        ([0], [0]),
        (["only"], ["only"]),
    ],
)
def test_quick_sort_base_cases(collection, expected):
    """Collections with fewer than two elements are returned unchanged."""
    assert quick_sort(collection) == expected


def test_quick_sort_base_case_returns_same_object():
    """For the base case the very same object is returned (no copy is made)."""
    single = [7]
    assert quick_sort(single) is single


@pytest.mark.parametrize(
    "collection",
    [
        [1, 1, 1, 1],
        [5, 5, 4, 4, 3, 3],
        [0, 0, -1, -1],
    ],
)
def test_quick_sort_all_duplicates(collection):
    """Lists containing only duplicates are sorted correctly."""
    expected = sorted(collection)
    assert quick_sort(list(collection)) == expected


def test_quick_sort_already_sorted():
    """An already sorted list stays sorted."""
    data = list(range(10))
    expected = list(data)
    assert quick_sort(list(data)) == expected


def test_quick_sort_reverse_sorted():
    """A reverse sorted list is fully reversed into ascending order."""
    data = list(range(10, 0, -1))
    assert quick_sort(list(data)) == list(range(1, 11))


def test_quick_sort_large_random_list():
    """A larger random list matches Python's built-in sorted output."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(data)
    assert quick_sort(list(data)) == expected


def test_quick_sort_extreme_values():
    """Very large and very small integers are handled without issue."""
    data = [10**18, -(10**18), 0, 1, -1]
    assert quick_sort(list(data)) == [-(10**18), -1, 0, 1, 10**18]


# ---------------------------------------------------------------------------
# Deterministic pivot selection via mocking
# ---------------------------------------------------------------------------


def test_quick_sort_with_patched_pivot_first_element():
    """With the pivot forced to index 0 the result is still correctly sorted."""
    with patch.object(quick_sort_module, "randrange", return_value=0):
        assert quick_sort([3, 1, 2]) == [1, 2, 3]


def test_quick_sort_with_patched_pivot_last_element():
    """With the pivot forced to the last index the result is still sorted."""
    with patch.object(quick_sort_module, "randrange", side_effect=lambda n: n - 1):
        assert quick_sort([9, 4, 7, 1]) == [1, 4, 7, 9]


def test_quick_sort_calls_randrange_for_non_trivial_input():
    """randrange is invoked with the current collection length as its bound."""
    with patch.object(
        quick_sort_module, "randrange", return_value=0
    ) as mock_randrange:
        quick_sort([3, 1, 2])
    assert mock_randrange.call_count >= 1
    for call in mock_randrange.call_args_list:
        (bound,) = call.args
        assert isinstance(bound, int)
        assert bound >= 1


def test_quick_sort_does_not_call_randrange_for_base_case():
    """No pivot selection happens for collections of length 0 or 1."""
    with patch.object(quick_sort_module, "randrange") as mock_randrange:
        quick_sort([])
        quick_sort([1])
    mock_randrange.assert_not_called()


def test_quick_sort_pivot_placed_between_partitions():
    """The pivot ends up between the lesser and greater partitions."""
    with patch.object(quick_sort_module, "randrange", return_value=0):
        # pivot is 5; lesser = [1, 2, 3], greater = [9]
        assert quick_sort([5, 1, 9, 2, 3]) == [1, 2, 3, 5, 9]


# ---------------------------------------------------------------------------
# Doctest-style consistency checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        [0, 5, 3, 2, 2],
        [],
        [-2, 5, 0, -45],
        [1],
        [2, 1],
        [1, 2],
    ],
)
def test_quick_sort_matches_builtin_sorted(collection):
    """Output always matches Python's built-in sorted() for the same input."""
    expected = sorted(collection)
    assert quick_sort(list(collection)) == expected
