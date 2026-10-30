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
        (["a", "b", "c"], "c", 2),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_empty_sequence_returns_minus_one():
    """An empty sequence must yield -1 rather than raising."""
    assert linear_search([], 1) == -1


def test_linear_search_returns_first_occurrence_of_duplicates():
    """When duplicates exist, the first matching index is returned."""
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_does_not_mutate_input():
    """The input sequence must be left untouched."""
    sequence = [4, 3, 2, 1]
    original = list(sequence)
    linear_search(sequence, 2)
    assert sequence == original


def test_linear_search_accepts_tuple():
    """Any iterable with comparable items (e.g. a tuple) should work."""
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
        ([1, 2, 3, 4, 5, 6], 1, 4, 5, 4),
        ([1, 2, 3, 4, 5, 6], 1, 4, 6, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search finds the target within [low, high] or returns -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


@pytest.mark.xfail(
    reason=(
        "rec_linear_search raises 'Invalid upper or lower bound!' for a "
        "single-element sequence when the target is absent: after checking "
        "index 0 it recurses to low=1, high=-1, and the bounds check runs "
        "before the high < low guard."
    ),
)
def test_rec_linear_search_single_element_not_found_returns_minus_one():
    """A single-element sequence with an absent target should return -1."""
    assert rec_linear_search([7], 0, 0, 8) == -1


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
    sequence = [5, 4, 3, 2, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, 4, 3)
    assert sequence == original


def test_rec_linear_search_agrees_with_iterative():
    """Both implementations should agree on the same inputs."""
    sequence = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    for target in range(-1, 11):
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )