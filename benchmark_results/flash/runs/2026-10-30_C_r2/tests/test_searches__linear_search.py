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
    """An empty sequence must yield -1 rather than raising."""
    assert linear_search([], 1) == -1


def test_linear_search_returns_first_occurrence():
    """When duplicates exist, the first matching index is returned."""
    assert linear_search([7, 7, 7], 7) == 0
    assert linear_search([1, 2, 2, 3], 2) == 1


def test_linear_search_does_not_require_sorted_input():
    """Linear search works on unsorted sequences."""
    assert linear_search([9, 3, 7, 1], 1) == 3


def test_linear_search_does_not_mutate_sequence():
    """The input sequence must be left untouched."""
    sequence = [4, 2, 8]
    linear_search(sequence, 8)
    assert sequence == [4, 2, 8]


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
        ([5, 9], 0, 1, 6, -1),
        ([1, 2, 3], 0, 2, 9, -1),
        ([1, 2, 3, 4], 1, 2, 3, 2),
        ([1, 2, 3, 4], 1, 2, 1, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search finds the target within the given bounds."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_single_element_window_miss():
    """A one-element window whose value does not match exhausts the bounds.

    ``rec_linear_search`` validates ``low``/``high`` *before* the
    ``high < low`` termination check, so recursing out of a single-element
    window (``low + 1`` / ``high - 1``) trips the documented bounds guard
    instead of returning -1.  This pins the current behaviour of the
    implementation for that corner case.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([5], 0, 0, 6)


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """An inverted (empty) range returns -1 without recursing."""
    assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1


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
    sequence = [4, 2, 8, 6]
    rec_linear_search(sequence, 0, 3, 8)
    assert sequence == [4, 2, 8, 6]


def test_rec_linear_search_matches_iterative_result():
    """Both implementations agree on the same inputs."""
    sequence = [3, 1, 4, 1, 5, 9, 2, 6]
    for target in sequence + [0, 100]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
