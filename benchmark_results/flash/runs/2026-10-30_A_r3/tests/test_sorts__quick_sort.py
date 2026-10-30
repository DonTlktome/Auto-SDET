"""Tests for the pure Python quick sort implementation in sorts/quick_sort.py."""

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort

# Resolve the module object that actually defines ``quick_sort`` so we can patch
# ``randrange`` on it directly.  String targets such as
# "sorts.quick_sort.randrange" make ``unittest.mock`` re-import the package by
# name, which fails in this environment even though the initial import above
# succeeds.
quick_sort_module = sys.modules.get(quick_sort.__module__)
if quick_sort_module is None:  # pragma: no cover - defensive fallback
    import importlib

    quick_sort_module = importlib.import_module(quick_sort.__module__)


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_does_not_alias_input() -> None:
    """A newly built, correctly sorted list is returned.

    Note: the implementation pops the chosen pivot out of the collection it is
    given, so the caller's list is consumed while sorting.  What must hold is
    that the *result* is a brand new list holding the sorted values.
    """
    original = [4, 1, 3, 2]

    result = quick_sort(original)

    assert result == [1, 2, 3, 4]
    assert result is not original


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are sorted correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are supported since they are comparable."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Strings are comparable and therefore sortable by this implementation."""
    assert quick_sort(["pear", "apple", "banana"]) == ["apple", "banana", "pear"]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7, 7], id="two-equal-elements"),
        pytest.param([1, 2, 3, 4, 5], id="already-sorted"),
        pytest.param([5, 4, 3, 2, 1], id="reverse-sorted"),
        pytest.param([3, 3, 3, 3], id="all-duplicates"),
        pytest.param([0, 0, -1, -1, 1, 1], id="duplicates-mixed-signs"),
    ],
)
def test_quick_sort_edge_cases(collection: list) -> None:
    """Edge-case inputs (empty, singleton, sorted, reversed, duplicates) sort correctly."""
    # ``quick_sort`` pops the pivot from the list it is given, so the expected
    # ordering has to be computed *before* the call mutates ``collection``.
    expected = sorted(collection)
    assert quick_sort(collection) == expected


def test_quick_sort_empty_list_returns_empty_list() -> None:
    """An empty list is returned unchanged (base case, no pivot selection)."""
    assert quick_sort([]) == []


def test_quick_sort_single_element_returns_same_element() -> None:
    """A one-element list is returned as-is without touching the RNG."""
    with patch.object(quick_sort_module, "randrange") as mock_randrange:
        assert quick_sort([9]) == [9]
    mock_randrange.assert_not_called()


def test_quick_sort_does_not_call_randrange_for_empty_list() -> None:
    """The base case for an empty list must not attempt to pick a pivot."""
    with patch.object(quick_sort_module, "randrange") as mock_randrange:
        assert quick_sort([]) == []
    mock_randrange.assert_not_called()


def test_quick_sort_large_random_list_matches_builtin_sorted() -> None:
    """A larger randomized list produces the same ordering as ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(300)]

    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Isolation of the external dependency (random.randrange)
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """The pivot index is chosen via ``random.randrange`` over the list length."""
    with patch.object(quick_sort_module, "randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    assert mock_randrange.call_count >= 1
    # The first call must be bounded by the length of the input collection.
    assert mock_randrange.call_args_list[0].args == (3,)


@pytest.mark.parametrize("pivot_index", [0, 1, 2, 3])
def test_quick_sort_is_correct_for_every_possible_pivot_choice(pivot_index: int) -> None:
    """Sorting is correct regardless of which pivot index is selected."""

    def fake_randrange(length: int) -> int:
        # Keep the requested pivot preference but stay inside the sub-list.
        return min(pivot_index, length - 1)

    with patch.object(quick_sort_module, "randrange", side_effect=fake_randrange):
        assert quick_sort([4, 2, 9, 1]) == [1, 2, 4, 9]


def test_quick_sort_with_deterministic_pivot_handles_duplicates() -> None:
    """Duplicates are partitioned into the 'lesser or equal' group and preserved."""
    with patch.object(quick_sort_module, "randrange", return_value=0):
        assert quick_sort([2, 2, 1, 2, 1]) == [1, 1, 2, 2, 2]


def test_quick_sort_randrange_receives_shrinking_lengths() -> None:
    """Recursive calls pass progressively smaller lengths to ``randrange``."""
    with patch.object(
        quick_sort_module, "randrange", return_value=0
    ) as mock_randrange:
        quick_sort([5, 4, 3, 2, 1])

    lengths = [call.args[0] for call in mock_randrange.call_args_list]
    assert lengths[0] == 5
    assert all(length >= 2 for length in lengths)
    assert lengths == sorted(lengths, reverse=True)


# ---------------------------------------------------------------------------
# Doctest parity
# ---------------------------------------------------------------------------


def test_quick_sort_matches_documented_doctest_examples() -> None:
    """The examples documented in the docstring hold true."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]
    assert quick_sort([]) == []
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]
