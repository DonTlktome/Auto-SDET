"""Tests for the linear search implementations in searches/linear_search.py."""

import pytest

from searches.linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence, target, expected",
    [
        ([0, 5, 7, 10, 15], 0, 0),          # first element
        ([0, 5, 7, 10, 15], 15, 4),         # last element
        ([0, 5, 7, 10, 15], 5, 1),          # middle element
        ([0, 5, 7, 10, 15], 6, -1),         # missing element
        ([42], 42, 0),                      # single element, found
        ([42], 7, -1),                      # single element, not found
        ([], 1, -1),                        # empty sequence
        ([3, 1, 2], 2, 2),                  # unsorted sequence
        ([1, 2, 3, 2], 2, 1),               # duplicates -> first occurrence
        ([-5, -1, 0, 3], -1, 1),            # negative numbers
        (["a", "b", "c"], "c", 2),          # non-int comparable items
    ],
)
def test_linear_search(sequence, target, expected):
    """linear_search returns the index of the first match or -1 when absent."""
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    """When duplicates exist, the lowest index must be returned."""
    assert linear_search([9, 9, 9], 9) == 0


def test_linear_search_does_not_mutate_input():
    """The input sequence must be left untouched by the search."""
    sequence = [4, 8, 15, 16, 23, 42]
    original = list(sequence)
    linear_search(sequence, 16)
    assert sequence == original


def test_linear_search_accepts_tuple():
    """Any iterable of comparable items (e.g. a tuple) should work."""
    assert linear_search((10, 20, 30), 20) == 1


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence, low, high, target, expected",
    [
        ([0, 30, 500, 100, 700], 0, 4, 0, 0),      # first element
        ([0, 30, 500, 100, 700], 0, 4, 700, 4),    # last element
        ([0, 30, 500, 100, 700], 0, 4, 30, 1),     # second element
        ([0, 30, 500, 100, 700], 0, 4, 500, 2),    # middle element
        ([0, 30, 500, 100, 700], 0, 4, -6, -1),    # missing element
        ([7], 0, 0, 7, 0),                         # single element, found
        # Known source bug: for a single-element window that does not match,
        # the implementation recurses to `low=1, high=-1` and the bounds guard
        # raises "Invalid upper or lower bound!" instead of returning -1.
        # The assertion is kept (xfail, non-strict) so this case flips to
        # XPASS as soon as the implementation is fixed.
        pytest.param(
            [7], 0, 0, 3, -1,                      # single element, not found
            marks=pytest.mark.xfail(
                reason=(
                    "rec_linear_search recurses to (low=1, high=-1) and raises "
                    "'Invalid upper or lower bound!' instead of returning -1"
                ),
                raises=Exception,
                strict=False,
            ),
        ),
        ([1, 2, 3, 4], 1, 2, 3, 2),                # restricted sub-range
        ([1, 2, 3, 4], 1, 2, 4, -1),               # target outside sub-range
        ([1, 2, 3, 4], 2, 1, 2, -1),               # high < low -> -1
    ],
)
def test_rec_linear_search(sequence, low, high, target, expected):
    """rec_linear_search finds the index within [low, high] or returns -1."""
    assert rec_linear_search(sequence, low, high, target) == expected


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
def test_rec_linear_search_invalid_bounds(sequence, low, high):
    """Out-of-range bounds must raise an Exception."""
    with pytest.raises(Exception, match="Invalid upper or lower bound!"):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_input():
    """The recursive search must not modify the input sequence."""
    sequence = [5, 4, 3, 2, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, len(sequence) - 1, 3)
    assert sequence == original


def test_rec_linear_search_matches_iterative():
    """Both implementations must agree on the same inputs."""
    sequence = [11, 22, 33, 44, 55, 66]
    for target in sequence + [0, 100]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )
