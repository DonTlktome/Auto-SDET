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
        ([], 1, -1),
        ([3, 3, 3], 3, 0),
        ([-5, -1, 0, 1], -1, 1),
        (["a", "b", "c"], "c", 2),
    ],
)
def test_linear_search(sequence, target, expected):
    """linear_search returns the index of the first match or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    """When duplicates exist, the lowest index must be returned."""
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_does_not_require_sorted_input():
    """Linear search works on unsorted sequences."""
    assert linear_search([9, 3, 7, 1, 5], 1) == 3


def test_linear_search_empty_sequence():
    """An empty sequence always yields -1."""
    assert linear_search([], 0) == -1


def test_linear_search_does_not_mutate_input():
    """The input sequence must be left untouched."""
    sequence = [4, 2, 8]
    original = list(sequence)
    linear_search(sequence, 8)
    assert sequence == original


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
        ([7], 0, 0, 7, 0),
        ([7, 8], 0, 1, 9, -1),
        ([1, 2, 3, 4], 1, 2, 3, 2),
        ([1, 2, 3, 4], 1, 2, 4, -1),
    ],
)
def test_rec_linear_search(sequence, low, high, target, expected):
    """rec_linear_search finds the target within the inclusive bounds."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_single_element_non_match_raises():
    """A single-element non-match currently recurses to invalid bounds.

    The implementation validates the bounds before the ``high < low`` check, so
    the recursive call with ``low=1`` and ``high=-1`` raises rather than
    returning -1.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([7], 0, 0, 8)


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """A valid but inverted range (high < low) yields -1."""
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
    """Out-of-range bounds must raise an Exception."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_input():
    """The recursive variant must not modify the input sequence."""
    sequence = [5, 1, 9, 3]
    original = list(sequence)
    rec_linear_search(sequence, 0, len(sequence) - 1, 9)
    assert sequence == original


def test_rec_linear_search_matches_iterative_result():
    """Both implementations agree on the same inputs."""
    sequence = [10, 20, 30, 40, 50]
    for target in sequence + [0, 25, 60]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
