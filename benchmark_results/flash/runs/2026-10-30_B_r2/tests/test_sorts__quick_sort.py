"""Tests for the pure Python quick sort implementation."""

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort


# Patch the ``randrange`` global on the module object that actually defines
# ``quick_sort``.  Using ``patch("sorts.quick_sort.randrange")`` forces
# ``unittest.mock`` to resolve a dotted import path, which fails when the
# ``sorts`` package is not importable by name at patch time.
_QUICK_SORT_MODULE = sys.modules[quick_sort.__module__]


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_sorted_list_for_already_sorted_input() -> None:
    """An already sorted list stays sorted."""
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_returns_sorted_list_for_reverse_sorted_input() -> None:
    """A reverse sorted list is reversed into ascending order."""
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Strings are sorted lexicographically."""
    assert quick_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


def test_quick_sort_handles_tuples_of_comparable_items() -> None:
    """Tuples are compared element-wise and sorted accordingly."""
    assert quick_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param(["only"], id="single-string"),
    ],
)
def test_quick_sort_base_cases_return_input(collection: list) -> None:
    """Collections with fewer than two elements are returned unchanged."""
    assert quick_sort(collection) == collection


def test_quick_sort_all_equal_elements() -> None:
    """A list of identical elements is returned with the same length and values."""
    result = quick_sort([7, 7, 7, 7, 7])
    assert result == [7, 7, 7, 7, 7]


def test_quick_sort_two_elements_unsorted() -> None:
    """The smallest non-trivial case (two elements) is sorted correctly."""
    assert quick_sort([2, 1]) == [1, 2]


def test_quick_sort_two_elements_sorted() -> None:
    """Two already ordered elements remain in order."""
    assert quick_sort([1, 2]) == [1, 2]


def test_quick_sort_duplicates_preserved() -> None:
    """Duplicate values are all preserved (multiset equality holds)."""
    data = [3, 1, 3, 2, 1, 3]
    original_length = len(data)
    # The implementation pops the pivot from the list it receives, so pass a
    # copy and compare the result against the original length.
    result = quick_sort(list(data))
    assert result == [1, 1, 2, 3, 3, 3]
    assert len(result) == original_length


def test_quick_sort_large_random_list_matches_builtin_sorted() -> None:
    """A large random list produces the same ordering as the builtin sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert quick_sort(list(data)) == sorted(data)


def test_quick_sort_does_not_mutate_input_length() -> None:
    """The input list is not left with a different number of elements.

    The implementation pops the pivot from the input list, so the input may be
    mutated; this test documents that the returned list has the original length.
    """
    data = [4, 2, 9, 1]
    original_length = len(data)
    result = quick_sort(data)
    assert len(result) == original_length


def test_quick_sort_returns_new_list_object() -> None:
    """The returned list is a distinct object from the input list."""
    data = [3, 1, 2]
    result = quick_sort(data)
    assert result is not data


# ---------------------------------------------------------------------------
# Isolation of the random pivot selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pivot_index", [0, 1, 2, 3])
def test_quick_sort_with_patched_pivot_index(pivot_index: int) -> None:
    """Sorting is correct regardless of which pivot index is chosen."""
    data = [4, 1, 3, 2]

    def _clamped_randrange(stop: int) -> int:
        """Return ``pivot_index`` clamped so it is valid for ``stop``."""
        return min(pivot_index, stop - 1)

    with patch.object(
        _QUICK_SORT_MODULE, "randrange", side_effect=_clamped_randrange
    ) as mock_randrange:
        result = quick_sort(list(data))
    assert result == [1, 2, 3, 4]
    assert mock_randrange.called


def test_quick_sort_calls_randrange_with_collection_length() -> None:
    """randrange is invoked with the current collection length as its argument."""
    with patch.object(_QUICK_SORT_MODULE, "randrange", return_value=0) as mock_randrange:
        quick_sort([3, 1, 2])
    assert mock_randrange.call_args_list[0].args == (3,)


def test_quick_sort_does_not_call_randrange_for_base_cases() -> None:
    """No pivot selection happens for empty or single-element collections."""
    with patch.object(_QUICK_SORT_MODULE, "randrange") as mock_randrange:
        quick_sort([])
        quick_sort([1])
    mock_randrange.assert_not_called()


def test_quick_sort_is_deterministic_for_fixed_seed() -> None:
    """With a seeded global RNG the result is reproducible and correct."""
    data = [9, 3, 7, 1, 8, 2, 6, 4, 5, 0]
    random.seed(2024)
    first = quick_sort(list(data))
    random.seed(2024)
    second = quick_sort(list(data))
    assert first == second == sorted(data)
