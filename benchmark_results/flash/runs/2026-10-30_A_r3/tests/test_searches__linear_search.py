'''Tests for the linear search implementations in searches/linear_search.py.'''

import pytest

from searches.linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sequence, target, expected',
    [
        ([0, 5, 7, 10, 15], 0, 0),          # first element
        ([0, 5, 7, 10, 15], 15, 4),         # last element
        ([0, 5, 7, 10, 15], 5, 1),          # middle element
        ([0, 5, 7, 10, 15], 6, -1),         # missing element
        ([42], 42, 0),                      # single element, found
        ([42], 7, -1),                      # single element, not found
        ([], 1, -1),                        # empty sequence
        ([3, 3, 3], 3, 0),                  # duplicates -> first occurrence
        ([9, 8, 7, 6], 6, 3),               # unsorted sequence
        ([-5, -1, 0, 1], -5, 0),            # negative values
        (['a', 'b', 'c'], 'c', 2),          # non-int comparable items
    ],
)
def test_linear_search_returns_correct_index(sequence, target, expected):
    '''linear_search must return the index of the target or -1 when absent.'''
    assert linear_search(sequence, target) == expected


def test_linear_search_returns_first_occurrence():
    '''When duplicates exist, the lowest index must be returned.'''
    assert linear_search([1, 2, 2, 2, 3], 2) == 1


def test_linear_search_does_not_mutate_input():
    '''linear_search must leave the input sequence untouched.'''
    sequence = [4, 3, 2, 1]
    original = list(sequence)
    linear_search(sequence, 2)
    assert sequence == original


def test_linear_search_accepts_tuple():
    '''linear_search only iterates, so any iterable of comparable items works.'''
    assert linear_search((10, 20, 30), 20) == 1


def test_linear_search_returns_int_type():
    '''The return value must always be an int.'''
    assert isinstance(linear_search([1, 2, 3], 2), int)
    assert isinstance(linear_search([1, 2, 3], 99), int)


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sequence, low, high, target, expected',
    [
        ([0, 30, 500, 100, 700], 0, 4, 0, 0),      # first element
        ([0, 30, 500, 100, 700], 0, 4, 700, 4),    # last element
        ([0, 30, 500, 100, 700], 0, 4, 30, 1),     # second element
        ([0, 30, 500, 100, 700], 0, 4, 500, 2),    # middle element
        ([0, 30, 500, 100, 700], 0, 4, -6, -1),    # missing element
        ([7], 0, 0, 7, 0),                         # single element, found
        ([7, 8], 0, 1, 9, -1),                     # two elements, missing
        ([1, 2, 3, 4, 5], 1, 3, 3, 2),             # restricted sub-range
        ([1, 2, 3, 4, 5], 1, 3, 5, -1),            # target outside sub-range
        ([1, 2, 3, 4, 5], 2, 2, 3, 2),             # low == high, found (absolute index)
        ([1, 2, 3, 4, 5], 2, 2, 9, -1),            # low == high, not found
    ],
)
def test_rec_linear_search_returns_correct_index(sequence, low, high, target, expected):
    '''rec_linear_search must return the index of the target or -1 when absent.'''
    assert rec_linear_search(sequence, low, high, target) == expected


def test_rec_linear_search_single_element_not_found_raises():
    '''The implementation validates bounds on each recursive call; a one-element
    miss recurses to low=1, high=-1 and therefore raises instead of returning -1.
    '''
    with pytest.raises(Exception, match='Invalid upper or lower bound!'):
        rec_linear_search([7], 0, 0, 1)


def test_rec_linear_search_high_less_than_low_returns_minus_one():
    '''A valid but inverted range (high < low) must yield -1, not raise.'''
    assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1


@pytest.mark.parametrize(
    'sequence, low, high',
    [
        ([1, 2, 3], 0, 3),      # high out of range (== len)
        ([1, 2, 3], 0, -1),     # negative high
        ([1, 2, 3], -1, 2),     # negative low
        ([1, 2, 3], 3, 2),      # low out of range
        ([], 0, 0),             # empty sequence
    ],
)
def test_rec_linear_search_invalid_bounds_raise(sequence, low, high):
    '''Out-of-range bounds must raise an Exception with the documented message.'''
    with pytest.raises(Exception, match='Invalid upper or lower bound!'):
        rec_linear_search(sequence, low, high, 1)


def test_rec_linear_search_does_not_mutate_input():
    '''rec_linear_search must leave the input sequence untouched.'''
    sequence = [5, 4, 3, 2, 1]
    original = list(sequence)
    rec_linear_search(sequence, 0, 4, 3)
    assert sequence == original


def test_rec_linear_search_returns_int_type():
    '''The return value must always be an int.'''
    assert isinstance(rec_linear_search([1, 2, 3], 0, 2, 2), int)
    assert isinstance(rec_linear_search([1, 2, 3], 0, 2, 99), int)


# ---------------------------------------------------------------------------
# Consistency between the iterative and recursive implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sequence, target',
    [
        ([0, 5, 7, 10, 15], 0),
        ([0, 5, 7, 10, 15], 15),
        ([0, 5, 7, 10, 15], 6),
        ([9, 8, 7, 6], 6),
        ([1], 1),
        ([1, 2], 3),
    ],
)
def test_iterative_and_recursive_agree(sequence, target):
    '''Both implementations must agree on the full-range search result.'''
    assert linear_search(sequence, target) == rec_linear_search(
        sequence, 0, len(sequence) - 1, target
    )
