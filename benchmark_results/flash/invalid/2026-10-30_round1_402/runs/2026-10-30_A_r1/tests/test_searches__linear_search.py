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
        ([3, 3, 3], 3, 0),
        ([-5, -1, 0, 1], -1, 1),
        ([1, 2, 3, 4, 5], 5, 4),
    ],
)
def test_linear_search_returns_index_or_minus_one(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_empty_sequence_returns_minus_one():
    """An empty sequence has no elements, so the result must be -1."""
    assert linear_search([], 1) == -1


def test_linear_search_returns_first_occurrence_of_duplicates():
    """When duplicates exist, the lowest index is returned."""
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_unsorted_sequence():
    """Linear search does not require a sorted input sequence."""
    assert linear_search([9, 3, 7, 1], 1) == 3


def test_linear_search_does_not_mutate_input():
    """The input sequence must be left untouched by the search."""
    sequence = [4, 8, 15, 16, 23, 42]
    original = list(sequence)
    linear_search(sequence, 16)
    assert sequence == original


def test_linear_search_works_with_strings():
    """The comparison is generic, so non-integer comparable items work too."""
    assert linear_search(["a", "b", "c"], "c") == 2
    assert linear_search(["a", "b", "c"], "z") == -1


def test_linear_search_accepts_tuple():
    """Any iterable of comparable items can be searched."""
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
        ([7], 0, 0, 7, 0),
        ([7, 8], 0, 1, 8, 1),
        ([7, 8], 0, 1, 9, -1),
        ([1, 2, 3, 4], 1, 2, 2, 1),
        ([1, 2, 3, 4], 1, 2, 4, -1),
    ],
)
def test_rec_linear_search_finds_target(sequence, low, high, target, expected):
    """rec_linear_search returns the index of the target within [low, high]."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_single_element_miss_raises_bounds_error():
    """A single-element range that misses does not return -1.

    The implementation checks the bounds at every call, so a miss on
    ``low == high`` recurses to ``(low=1, high=-1)`` and the guard raises
    before the ``high < low`` early return can produce -1.  This pins the
    documented behaviour of the shipped implementation.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([7], 0, 0, 8)


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """An inverted (empty) range yields -1 without raising."""
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


def test_rec_linear_search_does_not_mutate_input():
    """The recursive search must not modify the input sequence."""
    sequence = [5, 4, 3, 2, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, 4, 3)
    assert sequence == original


def test_rec_linear_search_works_with_strings():
    """Comparable non-integer items are supported by the recursive variant."""
    assert rec_linear_search(["x", "y", "z"], 0, 2, "z") == 2
    assert rec_linear_search(["x", "y", "z"], 0, 2, "q") == -1


def test_rec_linear_search_matches_linear_search_on_full_range():
    """Both implementations agree when searching the whole sequence."""
    sequence = [11, 4, 9, 4, 27, 0]
    for target in sequence + [99]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
