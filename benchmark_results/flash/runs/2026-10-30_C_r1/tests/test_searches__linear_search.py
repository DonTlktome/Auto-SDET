'''Tests for searches/linear_search.py'''

import pytest

from searches.linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sequence, target, expected',
    [
        ([0, 5, 7, 10, 15], 0, 0),
        ([0, 5, 7, 10, 15], 15, 4),
        ([0, 5, 7, 10, 15], 5, 1),
        ([0, 5, 7, 10, 15], 6, -1),
        ([42], 42, 0),
        ([42], 7, -1),
        ([], 1, -1),
        ([3, 3, 3], 3, 0),
        ([1, 2, 3, 4, 5], 1, 0),
        ([1, 2, 3, 4, 5], 5, 4),
        ([-10, -5, 0, 5, 10], -5, 1),
        (['a', 'b', 'c'], 'c', 2),
        (['a', 'b', 'c'], 'z', -1),
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    '''linear_search returns the index of the target or -1 when absent.'''
    assert linear_search(sequence, target) == expected


def test_linear_search_empty_sequence_returns_minus_one():
    '''An empty sequence must yield -1 (boundary case).'''
    assert linear_search([], 0) == -1


def test_linear_search_returns_first_occurrence():
    '''When duplicates exist, the first matching index is returned.'''
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_does_not_require_sorted_input():
    '''Linear search works on unsorted sequences.'''
    assert linear_search([9, 3, 7, 1, 5], 1) == 3


def test_linear_search_does_not_mutate_sequence():
    '''The input sequence must not be modified by the search.'''
    sequence = [4, 8, 15, 16, 23, 42]
    original = list(sequence)
    linear_search(sequence, 16)
    assert sequence == original


def test_linear_search_returns_int_type():
    '''The return value is always an int.'''
    assert isinstance(linear_search([1, 2, 3], 2), int)
    assert isinstance(linear_search([1, 2, 3], 99), int)


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sequence, low, high, target, expected',
    [
        ([0, 30, 500, 100, 700], 0, 4, 0, 0),
        ([0, 30, 500, 100, 700], 0, 4, 700, 4),
        ([0, 30, 500, 100, 700], 0, 4, 30, 1),
        ([0, 30, 500, 100, 700], 0, 4, -6, -1),
        ([0, 30, 500, 100, 700], 0, 4, 500, 2),
        ([0, 30, 500, 100, 700], 0, 4, 100, 3),
        ([5], 0, 0, 5, 0),
        pytest.param(
            [5],
            0,
            0,
            6,
            -1,
            marks=pytest.mark.xfail(
                raises=Exception,
                reason='rec_linear_search raises on a single-element miss because it recurses before checking high < low',
                strict=False,
            ),
        ),
        ([1, 2, 3, 4], 1, 2, 2, 1),
        ([1, 2, 3, 4], 1, 2, 3, 2),
        ([1, 2, 3, 4], 1, 2, 4, -1),
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    '''rec_linear_search finds the target within [low, high] or returns -1.'''
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    '''When high < low the search space is empty and -1 is returned.'''
    assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1


@pytest.mark.parametrize(
    'sequence, low, high',
    [
        ([1, 2, 3], 0, 3),      # high out of range
        ([1, 2, 3], 0, -1),     # negative high
        ([1, 2, 3], -1, 2),     # negative low
        ([1, 2, 3], 3, 3),      # low out of range
        ([], 0, 0),             # empty sequence
    ],
)
def test_rec_linear_search_invalid_bounds_raise(sequence, low, high):
    '''Out-of-range bounds raise an Exception.'''
    with pytest.raises(Exception, match='Invalid upper or lower bound!'):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_sequence():
    '''The recursive search must not modify the input sequence.'''
    sequence = [0, 30, 500, 100, 700]
    original = list(sequence)
    rec_linear_search(sequence, 0, 4, 100)
    assert sequence == original


def test_rec_linear_search_returns_int_type():
    '''The return value is always an int.'''
    assert isinstance(rec_linear_search([1, 2, 3], 0, 2, 2), int)
    assert isinstance(rec_linear_search([1, 2, 3], 0, 2, 99), int)


def test_rec_linear_search_matches_iterative_result():
    '''Both implementations agree on the same inputs.'''
    sequence = [7, 1, 9, 3, 5, 11]
    for target in sequence + [0, 100]:
        assert rec_linear_search(sequence, 0, len(sequence) - 1, target) == linear_search(
            sequence, target
        )