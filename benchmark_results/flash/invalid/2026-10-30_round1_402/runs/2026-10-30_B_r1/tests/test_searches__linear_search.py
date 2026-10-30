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
        ([1, 2, 3, 4, 5], 3, 2),
        ([-3, -2, -1], -2, 1),
        ([5, 4, 3, 2, 1], 1, 4),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    """linear_search returns the index of the target or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    """linear_search returns the index of the first matching element."""
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_empty_sequence():
    """linear_search on an empty sequence returns -1."""
    assert linear_search([], 0) == -1


def test_linear_search_does_not_mutate_sequence():
    """linear_search must not modify the input sequence."""
    sequence = [3, 1, 2]
    original = list(sequence)
    linear_search(sequence, 2)
    assert sequence == original


def test_linear_search_works_with_strings():
    """linear_search works with any comparable items, e.g. strings."""
    assert linear_search(["a", "b", "c"], "c") == 2
    assert linear_search(["a", "b", "c"], "z") == -1


def test_linear_search_works_with_tuple():
    """linear_search accepts any iterable, including tuples."""
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
        ([7, 8], 0, 1, 9, -1),
        ([1, 2, 3, 4], 1, 2, 3, 2),
        ([1, 2, 3, 4], 1, 2, 1, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    """rec_linear_search returns the index of the target within bounds or -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


@pytest.mark.xfail(
    reason=(
        "rec_linear_search validates the bounds before checking high < low, so a "
        "single-element range whose only element is not the target recurses to "
        "low=1, high=-1 and raises 'Invalid upper or lower bound!' instead of "
        "returning -1."
    ),
    raises=Exception,
    strict=True,
)
def test_rec_linear_search_single_element_missing_target():
    """A one-element range with a missing target should yield -1 (known defect)."""
    assert rec_linear_search([7], 0, 0, 8) == -1


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    """When high < low (valid bounds) the search returns -1."""
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
    """rec_linear_search raises an Exception for invalid bounds."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_sequence():
    """rec_linear_search must not modify the input sequence."""
    sequence = [4, 3, 2, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, 3, 2)
    assert sequence == original


def test_rec_linear_search_matches_iterative_result():
    """Recursive and iterative searches agree on the same inputs."""
    sequence = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    for target in range(-1, 11):
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
