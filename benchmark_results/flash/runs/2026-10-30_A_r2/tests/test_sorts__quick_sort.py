"""Tests for the pure Python quick sort implementation."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------
# Make sure the directory that actually contains the ``sorts`` package is on
# sys.path before importing it, regardless of where pytest was started from.
_TEST_DIR = Path(__file__).resolve().parent
for _candidate in (_TEST_DIR, *_TEST_DIR.parents):
    if (_candidate / "sorts" / "quick_sort.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from sorts.quick_sort import quick_sort  # noqa: E402

# The module object that actually defines ``quick_sort``.  Patching attributes
# on this object (instead of using a dotted string target) avoids re-importing
# the package at patch time, which fails in some environments.
quick_sort_module = sys.modules[quick_sort.__module__]


def _pivot_selector(index: int):
    """Return a ``randrange`` replacement that imitates picking ``index``.

    ``index`` is clamped to the current collection length so that every
    recursive call still receives a valid pivot index.
    """

    def _choose(length: int) -> int:
        return min(index, length - 1)

    return _choose


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list():
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_sorted_list_and_preserves_elements():
    """The result is a freshly built, sorted list holding the same items.

    The implementation pops the pivot out of the collection while it
    partitions, so the caller's list is intentionally *not* preserved.
    We therefore only verify that a new list object is returned, that it is
    sorted and that no elements were silently lost.
    """
    original = [4, 1, 3, 2]
    expected_contents = sorted(original)
    result = quick_sort(original)
    assert result == [1, 2, 3, 4]
    assert result is not original
    assert sorted(result) == expected_contents
    assert len(result) == len(expected_contents)


def test_quick_sort_handles_negative_numbers():
    """Negative values are ordered correctly relative to positives."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats():
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings():
    """Strings are sorted lexicographically."""
    assert quick_sort(["banana", "apple", "cherry"]) == ["apple", "banana", "cherry"]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7], id="single-element-other"),
    ],
)
def test_quick_sort_base_cases_return_input(collection):
    """Collections with fewer than two elements are returned unchanged."""
    assert quick_sort(collection) == collection


def test_quick_sort_empty_list_returns_empty_list():
    """An empty list yields an empty list."""
    assert quick_sort([]) == []


def test_quick_sort_all_equal_elements():
    """A list of identical elements is preserved (no infinite recursion)."""
    assert quick_sort([5, 5, 5, 5, 5]) == [5, 5, 5, 5, 5]


def test_quick_sort_already_sorted():
    """An already sorted list stays sorted."""
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_reverse_sorted():
    """A reverse sorted list is fully reversed."""
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_two_elements_unsorted():
    """The smallest non-trivial case (two elements) is swapped correctly."""
    assert quick_sort([2, 1]) == [1, 2]


def test_quick_sort_two_elements_sorted():
    """Two already ordered elements remain in place."""
    assert quick_sort([1, 2]) == [1, 2]


def test_quick_sort_duplicates_and_negatives():
    """Duplicates mixed with negatives are grouped correctly."""
    assert quick_sort([3, -1, 3, -1, 0]) == [-1, -1, 0, 3, 3]


def test_quick_sort_large_random_list_matches_builtin_sorted():
    """A large random list produces the same result as the builtin sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Pivot selection / dependency isolation
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection():
    """The pivot index is chosen via random.randrange from the module namespace."""
    with patch.object(quick_sort_module, "randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])
    assert result == [1, 2, 3]
    assert mock_randrange.called


@pytest.mark.parametrize("pivot_index", [0, 1, 2, 3])
def test_quick_sort_correct_for_every_pivot_choice(pivot_index):
    """Sorting is correct regardless of which pivot index is selected."""
    data = [4, 2, 9, 1]
    with patch.object(
        quick_sort_module, "randrange", side_effect=_pivot_selector(pivot_index)
    ):
        assert quick_sort(list(data)) == [1, 2, 4, 9]


def test_quick_sort_randrange_called_with_collection_length():
    """randrange is invoked with the current collection length."""
    with patch.object(quick_sort_module, "randrange", return_value=0) as mock_randrange:
        quick_sort([10, 20, 30])
    assert mock_randrange.call_args_list[0].args == (3,)


def test_quick_sort_does_not_call_randrange_for_base_case():
    """No pivot selection happens for collections with fewer than two items."""
    with patch.object(quick_sort_module, "randrange") as mock_randrange:
        assert quick_sort([]) == []
        assert quick_sort([1]) == [1]
    mock_randrange.assert_not_called()
