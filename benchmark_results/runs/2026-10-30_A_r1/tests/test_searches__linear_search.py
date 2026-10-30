'''
Pytest test suite for the linear search algorithms in linear_search.py.

Covers both the iterative `linear_search` and the recursive
`rec_linear_search` implementations, including happy paths, edge cases,
and boundary conditions.
'''

import pytest

from linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------

class TestLinearSearch:
    '''Tests for the iterative `linear_search` function.'''

    @pytest.mark.parametrize(
        'sequence, target, expected',
        [
            ([0, 5, 7, 10, 15], 0, 0),   # target at the beginning
            ([0, 5, 7, 10, 15], 15, 4),  # target at the end
            ([0, 5, 7, 10, 15], 5, 1),   # target in the middle
            ([42], 42, 0),               # single-element list, found
        ],
    )
    def test_found_returns_index(self, sequence, target, expected):
        '''Target present in the sequence returns its correct index.'''
        assert linear_search(sequence, target) == expected

    @pytest.mark.parametrize(
        'sequence, target',
        [
            ([0, 5, 7, 10, 15], 6),   # missing value between elements
            ([], 1),                  # empty sequence
            ([42], 7),                # single-element list, not found
            ([1, 2, 3], -1),          # negative target not present
        ],
    )
    def test_not_found_returns_minus_one(self, sequence, target):
        '''Target absent from the sequence returns -1.'''
        assert linear_search(sequence, target) == -1

    def test_returns_first_occurrence(self):
        '''When duplicates exist, the first matching index is returned.'''
        assert linear_search([1, 2, 2, 3], 2) == 1

    def test_works_with_strings(self):
        '''Comparable non-integer items are supported.'''
        assert linear_search(['a', 'b', 'c'], 'b') == 1


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------

class TestRecLinearSearch:
    '''Tests for the recursive `rec_linear_search` function.'''

    @pytest.mark.parametrize(
        'sequence, low, high, target, expected',
        [
            ([0, 30, 500, 100, 700], 0, 4, 0, 0),     # target at low bound
            ([0, 30, 500, 100, 700], 0, 4, 700, 4),   # target at high bound
            ([0, 30, 500, 100, 700], 0, 4, 30, 1),    # target in the middle
            ([0, 30, 500, 100, 700], 0, 4, -6, -1),   # target not present
        ],
    )
    def test_found_returns_index(self, sequence, low, high, target, expected):
        '''Target present within [low, high] returns its correct index.'''
        assert rec_linear_search(sequence, low, high, target) == expected

    def test_single_element_found(self):
        '''A single-element range with the target returns that index.'''
        assert rec_linear_search([7], 0, 0, 7) == 0

    def test_single_element_not_found(self):
        '''A single-element range without the target returns -1.'''
        assert rec_linear_search([1, 7, 3], 1, 1, 2) == -1

    def test_empty_range_returns_minus_one(self):
        '''When high < low, the search returns -1.'''
        assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1

    @pytest.mark.parametrize(
        'sequence, low, high',
        [
            ([1, 2, 3], -1, 2),   # low below zero
            ([1, 2, 3], 0, 3),    # high equal to len(sequence)
            ([1, 2, 3], 3, 2),    # low equal to len(sequence)
            ([1, 2, 3], 0, -1),   # high below zero
        ],
    )
    def test_invalid_bounds_raise_exception(self, sequence, low, high):
        '''Out-of-range bounds raise an Exception.'''
        with pytest.raises(Exception, match='Invalid upper or lower bound!'):
            rec_linear_search(sequence, low, high, 1)

    def test_duplicate_target_returns_first_from_low(self):
        '''With duplicates, the occurrence closest to `low` is returned.'''
        assert rec_linear_search([2, 2, 2, 2], 0, 3, 2) == 0
