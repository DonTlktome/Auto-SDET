"""Tests for the pure Python quick sort implementation."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from quick_sort import quick_sort


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_mutates_input() -> None:
    """A new sorted list is returned; the input list is consumed (pivot popped)."""
    original = [4, 1, 3, 2]

    result = quick_sort(original)

    assert result == [1, 2, 3, 4]
    assert result is not original
    # The implementation pops the pivot from the passed-in list, so the
    # original list is mutated (one element removed).
    assert len(original) == 3


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Any comparable items (e.g. strings) are sorted lexicographically."""
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
    ],
)
def test_quick_sort_trivial_inputs(collection: list) -> None:
    """Collections with fewer than two elements are returned unchanged."""
    expected = list(collection)
    assert quick_sort(collection) == expected


def test_quick_sort_already_sorted() -> None:
    """An already sorted list stays sorted."""
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_reverse_sorted() -> None:
    """A descending list is reversed into ascending order."""
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_all_duplicates() -> None:
    """A list of identical values is returned with the same contents."""
    assert quick_sort([3, 3, 3, 3, 3]) == [3, 3, 3, 3, 3]


def test_quick_sort_many_duplicates() -> None:
    """Duplicates are preserved (multiset equality) and ordered correctly."""
    assert quick_sort([2, 1, 2, 1, 2, 1]) == [1, 1, 1, 2, 2, 2]


def test_quick_sort_large_random_list_matches_builtin_sorted() -> None:
    """A large random list produces the same result as the builtin sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(data)

    assert quick_sort(data) == expected


def test_quick_sort_mutates_input_for_large_list() -> None:
    """The original large list is consumed (pivot popped) after sorting."""
    rng = random.Random(99)
    data = [rng.randint(0, 100) for _ in range(200)]
    original_len = len(data)

    quick_sort(data)

    # The top-level pivot is popped from the input list.
    assert len(data) == original_len - 1


# ---------------------------------------------------------------------------
# Pivot selection (external dependency isolation)
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """The pivot index is chosen via random.randrange, called once per partition."""
    with patch("quick_sort.randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    # One call for the top-level list plus one for each non-trivial sublist.
    assert mock_randrange.call_count >= 1
    for call in mock_randrange.call_args_list:
        assert call.args[0] >= 1


def test_quick_sort_with_deterministic_pivot_is_stable_in_result() -> None:
    """Forcing the pivot to the last index still yields a correctly sorted list."""
    with patch("quick_sort.randrange", side_effect=lambda n: n - 1):
        assert quick_sort([9, 4, 7, 1, 4]) == [1, 4, 4, 7, 9]


def test_quick_sort_with_first_index_pivot() -> None:
    """Forcing the pivot to index 0 still yields a correctly sorted list."""
    with patch("quick_sort.randrange", return_value=0):
        assert quick_sort([5, 3, 8, 1]) == [1, 3, 5, 8]


def test_quick_sort_randrange_not_called_for_trivial_input() -> None:
    """No pivot is selected when the collection has fewer than two elements."""
    with patch("quick_sort.randrange") as mock_randrange:
        assert quick_sort([]) == []
        assert quick_sort([1]) == [1]

    mock_randrange.assert_not_called()
