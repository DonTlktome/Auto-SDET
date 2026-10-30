"""Tests for the pure Python quick sort implementation."""

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort

# ``quick_sort`` pops the pivot element out of the very list it is handed, so a
# single call shortens the caller's list by exactly one element.  Tests that
# need the original data afterwards therefore pass in a shallow copy.
#
# The module object is resolved through ``sys.modules`` instead of a dotted
# string such as "sorts.quick_sort" so that the module level ``randrange``
# symbol can always be patched, whatever the package layout is.
try:
    quick_sort_module = sys.modules[quick_sort.__module__]
except KeyError:  # pragma: no cover - defensive fallback
    import importlib

    quick_sort_module = importlib.import_module(quick_sort.__module__)


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_does_not_mutate_input() -> None:
    """Sorting a copy leaves the caller's own list untouched."""
    original = [4, 1, 3, 2]
    snapshot = list(original)

    result = quick_sort(list(original))

    assert result == [1, 2, 3, 4]
    assert result is not original
    assert original == snapshot


def test_quick_sort_pops_the_pivot_out_of_the_input_list() -> None:
    """The pivot element is removed from the list that was passed in."""
    data = [4, 1, 3, 2]

    quick_sort(data)

    assert len(data) == 3


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Any comparable items (e.g. strings) can be sorted."""
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7, 7], id="two-equal-elements"),
        pytest.param([1, 2], id="two-sorted-elements"),
        pytest.param([2, 1], id="two-reversed-elements"),
    ],
)
def test_quick_sort_small_collections(collection: list) -> None:
    """Tiny collections are ordered correctly."""
    expected = sorted(collection)
    assert quick_sort(list(collection)) == expected


def test_quick_sort_all_duplicates() -> None:
    """A list of identical values is returned with the same length and values."""
    assert quick_sort([5, 5, 5, 5, 5]) == [5, 5, 5, 5, 5]


def test_quick_sort_already_sorted() -> None:
    """An already sorted list stays sorted."""
    data = list(range(10))
    expected = list(data)

    assert quick_sort(data) == expected


def test_quick_sort_reverse_sorted() -> None:
    """A reverse sorted list is fully reversed into ascending order."""
    data = list(range(10, 0, -1))
    assert quick_sort(data) == list(range(1, 11))


def test_quick_sort_does_not_mutate_input_for_large_list() -> None:
    """Even for larger inputs the caller's list is left untouched when copied."""
    data = list(range(50, 0, -1))
    snapshot = list(data)

    result = quick_sort(list(data))

    assert result == list(range(1, 51))
    assert data == snapshot


def test_quick_sort_matches_builtin_sorted_on_random_data() -> None:
    """Randomized comparison against the standard library's sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-100, 100) for _ in range(200)]

    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Pivot selection (external dependency isolation)
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """The pivot index is chosen via random.randrange, which we can patch."""
    with patch.object(quick_sort_module, "randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    assert mock_randrange.called


def test_quick_sort_is_deterministic_when_pivot_is_fixed() -> None:
    """With a fixed pivot index the output is stable across runs."""
    data = [9, 3, 7, 1, 8, 2, 6, 4, 5]

    with patch.object(quick_sort_module, "randrange", return_value=0):
        first = quick_sort(list(data))
    with patch.object(quick_sort_module, "randrange", return_value=0):
        second = quick_sort(list(data))

    assert first == second == sorted(data)


def test_quick_sort_handles_worst_case_pivot_choice() -> None:
    """Always picking index 0 on sorted input still yields a correct result."""
    data = list(range(20))

    with patch.object(quick_sort_module, "randrange", return_value=0):
        result = quick_sort(list(data))

    assert result == data


def test_quick_sort_randrange_called_with_collection_length() -> None:
    """randrange receives the current collection length as its argument."""
    with patch.object(quick_sort_module, "randrange", return_value=0) as mock_randrange:
        quick_sort([2, 1])

    # The first call happens on the original two-element collection.
    assert mock_randrange.call_args_list[0].args == (2,)
