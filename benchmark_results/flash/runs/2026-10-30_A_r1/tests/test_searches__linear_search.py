"""Tests for the linear search implementations in searches/linear_search.py."""

import pytest

from searches.linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence, target, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([42], 42, 0),
        ([42], 7, -1),
        ([3, 1, 4, 1, 5, 9, 2, 6], 9, 5),
        ([-10, -5, 0, 5, 10], -5, 1),
        ([1.5, 2.5, 3.5], 2.5, 1),
        (["a", "b", "c"], "c", 2),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_empty_sequence_returns_minus_one():
    """An empty sequence can never contain the target, so -1 is returned."""
    assert linear_search([], 1) == -1


def test_linear_search_returns_first_occurrence_of_duplicates():
    """When duplicates exist, the index of the first occurrence is returned."""
    assert linear_search([7, 3, 7, 7], 7) == 0
    assert linear_search([1, 2, 2, 3], 2) == 1


def test_linear_search_does_not_require_sorted_input():
    """Linear search works on unsorted sequences."""
    assert linear_search([9, 1, 8, 2, 7], 2) == 3


def test_linear_search_does_not_mutate_sequence():
    """The input sequence must be left untouched."""
    sequence = [4, 2, 9, 1]
    original = list(sequence)
    linear_search(sequence, 9)
    assert sequence == original


def test_linear_search_accepts_tuple():
    """Any iterable of comparable items (e.g. a tuple) is supported."""
    assert linear_search((10, 20, 30), 20) == 1


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence, low, high, target, expected",
    [
        ([0, 30, 500, 100, 700], 0, 4, 0, 0),
        ([0, 30, 500, 100, 700], 0, 4, 700, 4),
        ([0, 30, 500, 100, 700], 0, 4, 30, 1),
        ([0, 30, 500, 100, 700], 0, 4, -6, -1),
        ([0, 30, 500, 100, 700], 0, 4, 500, 2),
        ([0, 30, 500, 100, 700], 0, 4, 100, 3),
        ([5], 0, 0, 5, 0),
        # A missing target with a one-element window is only guaranteed to return
        # -1 when stepping out of the window keeps the indices in range (see
        # test_rec_linear_search_single_element_miss_raises_bounds_error for the
        # length-1-sequence edge case of this implementation).
        ([5, 6], 0, 1, 7, -1),
        ([1, 2, 3, 4, 5, 6], 1, 4, 4, 3),
        ([1, 2, 3, 4, 5, 6], 2, 3, 1, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search finds the target within the given bounds or returns -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """An inverted range (high < low) yields -1 without inspecting elements."""
    assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1


def test_rec_linear_search_single_element_range():
    """A range covering exactly one element behaves like a direct lookup."""
    assert rec_linear_search([1, 2, 3], 1, 1, 2) == 1
    assert rec_linear_search([1, 2, 3], 1, 1, 3) == -1


def test_rec_linear_search_single_element_miss_raises_bounds_error():
    """Documented quirk of the implementation.

    When the window is a single element at the very start of a length-1
    sequence, the recursive step (`low + 1`, `high - 1`) leaves the valid index
    range, so the bounds validation fires before the `high < low` termination
    check can return -1.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([5], 0, 0, 6)


@pytest.mark.parametrize(
    "sequence, low, high",
    [
        ([1, 2, 3], 0, 3),      # high out of range
        ([1, 2, 3], 0, -1),     # negative high
        ([1, 2, 3], -1, 2),     # negative low
        ([1, 2, 3], 3, 2),      # low out of range
        ([], 0, 0),             # empty sequence
    ],
)
def test_rec_linear_search_invalid_bounds_raise(sequence, low, high):
    """Out-of-range bounds raise an Exception with the documented message."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_sequence():
    """The recursive variant must not modify the input sequence."""
    sequence = [4, 2, 9, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, len(sequence) - 1, 9)
    assert sequence == original


def test_rec_linear_search_accepts_tuple():
    """The recursive variant also works with tuples."""
    assert rec_linear_search((10, 20, 30), 0, 2, 30) == 2
