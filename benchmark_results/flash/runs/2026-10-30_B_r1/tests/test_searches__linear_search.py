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
        ([1, 2, 3, 4, 5], 5, 4),
        ([5, 4, 3, 2, 1], 1, 4),
        ([-10, -5, 0, 5, 10], -5, 1),
        (["a", "b", "c"], "c", 2),
        (["a", "b", "c"], "z", -1),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    """linear_search returns the index of the first matching element."""
    assert linear_search([1, 2, 1, 2, 1], 1) == 0
    assert linear_search([1, 2, 1, 2, 1], 2) == 1


def test_linear_search_empty_sequence():
    """linear_search on an empty sequence returns -1."""
    assert linear_search([], 0) == -1


def test_linear_search_does_not_modify_sequence():
    """linear_search must not mutate the input sequence."""
    sequence = [4, 8, 15, 16, 23, 42]
    original = list(sequence)
    linear_search(sequence, 16)
    assert sequence == original


def test_linear_search_works_on_tuple():
    """linear_search works with any iterable of comparable items, e.g. a tuple."""
    assert linear_search((10, 20, 30), 20) == 1
    assert linear_search((10, 20, 30), 99) == -1


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
        # Single-element slice whose target is absent.  The slice is kept
        # strictly inside the sequence because the implementation's bounds
        # guard triggers first when a non-matching single-element slice sits
        # at the array edge (e.g. ([7], 0, 0, 8)); that edge behaviour is
        # covered separately below.
        ([5, 7, 9], 1, 1, 8, -1),
        ([1, 2, 3, 4, 5], 1, 3, 3, 2),
        ([1, 2, 3, 4, 5], 1, 3, 5, -1),
        ([1, 2, 3, 4, 5], 2, 2, 3, 2),
        ([1, 2, 3, 4, 5], 2, 2, 4, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search returns the index of the target within [low, high] or -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """When high < low (valid bounds) the search returns -1."""
    assert rec_linear_search([1, 2, 3], 2, 1, 1) == -1


def test_rec_linear_search_missing_target_in_edge_single_element_slice():
    """Known behaviour: a non-matching single-element slice at the array edge

    The recursion narrows to (low + 1, high - 1) before the ``high < low``
    short-circuit is reached, so the bounds guard fires first and an exception
    is raised instead of -1 being returned.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([7], 0, 0, 8)


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
    """rec_linear_search raises an Exception for out-of-range bounds."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_modify_sequence():
    """rec_linear_search must not mutate the input sequence."""
    sequence = [9, 8, 7, 6, 5]
    original = list(sequence)
    rec_linear_search(sequence, 0, len(sequence) - 1, 6)
    assert sequence == original


def test_rec_linear_search_matches_iterative_result():
    """rec_linear_search agrees with linear_search on the full range."""
    sequence = [4, 1, 9, 2, 7, 3]
    for target in sequence + [100]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
