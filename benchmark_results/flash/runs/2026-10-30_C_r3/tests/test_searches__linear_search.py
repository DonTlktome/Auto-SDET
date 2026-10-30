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
        ([3, 1, 2], 2, 2),
        ([-5, -1, -10], -10, 2),
        ([1, 2, 3, 4, 5], 1, 0),
        ([1, 2, 3, 4, 5], 5, 4),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    """linear_search returns the index of the first matching element."""
    assert linear_search([1, 2, 2, 3, 2], 2) == 1


def test_linear_search_does_not_require_sorted_input():
    """linear_search works on unsorted sequences."""
    assert linear_search([9, 3, 7, 1, 5], 1) == 3


def test_linear_search_works_with_strings():
    """linear_search compares items with == so it works for non-int types too."""
    assert linear_search(["a", "b", "c"], "c") == 2
    assert linear_search(["a", "b", "c"], "z") == -1


def test_linear_search_empty_sequence_returns_minus_one():
    """An empty sequence yields -1 (boundary case)."""
    assert linear_search([], 0) == -1


def test_linear_search_does_not_mutate_sequence():
    """linear_search must not modify the input sequence."""
    sequence = [3, 1, 2]
    linear_search(sequence, 1)
    assert sequence == [3, 1, 2]


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
        ([1, 2, 3, 4], 1, 2, 3, 2),
        ([1, 2, 3, 4], 1, 2, 1, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search finds the target within [low, high] or returns -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_single_element_non_match_raises():
    """Known limitation: if the search range collapses to a single element that

    does not match, the recursive step ``(low + 1, high - 1)`` produces
    ``low=1, high=-1`` and the bounds guard fires instead of returning -1.
    This test pins that documented behaviour of the implementation.
    """
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search([7], 0, 0, 8)


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """When high < low the search space is empty and -1 is returned."""
    assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1


def test_rec_linear_search_single_element_range():
    """A single-element range is handled correctly (boundary case)."""
    assert rec_linear_search([5, 6, 7], 1, 1, 6) == 1
    assert rec_linear_search([5, 6, 7], 1, 1, 5) == -1


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
    """rec_linear_search raises Exception for out-of-range bounds."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_sequence():
    """rec_linear_search must not modify the input sequence."""
    sequence = [4, 3, 2, 1]
    rec_linear_search(sequence, 0, 3, 2)
    assert sequence == [4, 3, 2, 1]


def test_rec_linear_search_matches_iterative_result():
    """Both implementations agree on the same inputs."""
    sequence = [10, 20, 30, 40, 50]
    for target in [10, 20, 30, 40, 50, 99]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
