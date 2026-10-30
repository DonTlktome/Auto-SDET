'''
Pytest test suite for the linear search module.

Covers both the iterative ``linear_search`` and the recursive
``rec_linear_search`` implementations, including happy paths, edge cases,
and boundary conditions.
'''

import pytest

from linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------

class TestLinearSearch:
    '''Tests for the iterative ``linear_search`` function.'''

    @pytest.mark.parametrize(
        'sequence, target, expected',
        [
            ([0, 5, 7, 10, 15], 0, 0),   # target at the beginning
            ([0, 5, 7, 10, 15], 15, 4),  # target at the end
            ([0, 5, 7, 10, 15], 5, 1),   # target in the middle
            ([0, 5, 7, 10, 15], 6, -1),  # target not present
        ],
    )
    def test_linear_search_finds_target(self, sequence, target, expected):
        '''Verify the correct index is returned for various target positions.'''
        assert linear_search(sequence, target) == expected

    def test_linear_search_single_element_found(self):
        '''Verify a single-element list returns index 0 when the target matches.'''
        assert linear_search([42], 42) == 0

    def test_linear_search_single_element_not_found(self):
        '''Verify a single-element list returns -1 when the target is absent.'''
        assert linear_search([42], 7) == -1

    def test_linear_search_empty_sequence(self):
        '''Verify an empty sequence returns -1.'''
        assert linear_search([], 10) == -1

    def test_linear_search_duplicate_values_returns_first(self):
        '''Verify the first occurrence is returned when duplicates exist.'''
        assert linear_search([1, 2, 2, 3], 2) == 1

    def test_linear_search_negative_numbers(self):
        '''Verify negative targets are handled correctly.'''
        assert linear_search([-5, -3, 0, 3, 5], -3) == 1
        assert linear_search([-5, -3, 0, 3, 5], -10) == -1


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------

class TestRecLinearSearch:
    '''Tests for the recursive ``rec_linear_search`` function.'''

    @pytest.mark.parametrize(
        'sequence, low, high, target, expected',
        [
            ([0, 30, 500, 100, 700], 0, 4, 0, 0),    # target at low bound
            ([0, 30, 500, 100, 700], 0, 4, 700, 4),  # target at high bound
            ([0, 30, 500, 100, 700], 0, 4, 30, 1),   # target in the middle
            ([0, 30, 500, 100, 700], 0, 4, -6, -1),  # target not present
        ],
    )
    def test_rec_linear_search_finds_target(
        self, sequence, low, high, target, expected
    ):
        '''Verify the correct index is returned for various target positions.'''
        assert rec_linear_search(sequence, low, high, target) == expected

    def test_rec_linear_search_single_element_found(self):
        '''Verify a single-element list returns index 0 when the target matches.'''
        assert rec_linear_search([42], 0, 0, 42) == 0

    def test_rec_linear_search_single_element_not_found(self):
        '''Verify a single-element list raises an exception when target is absent.'''
        with pytest.raises(Exception, match='Invalid upper or lower bound!'):
            rec_linear_search([42], 0, 0, 7)

    def test_rec_linear_search_high_less_than_low(self):
        '''Verify -1 is returned when the high bound is below the low bound.'''
        assert rec_linear_search([1, 2, 3], 2, 1, 3) == -1

    def test_rec_linear_search_low_out_of_range(self):
        '''Verify an exception is raised when the low bound is out of range.'''
        with pytest.raises(Exception, match='Invalid upper or lower bound!'):
            rec_linear_search([1, 2, 3], -1, 2, 1)

    def test_rec_linear_search_high_out_of_range(self):
        '''Verify an exception is raised when the high bound is out of range.'''
        with pytest.raises(Exception, match='Invalid upper or lower bound!'):
            rec_linear_search([1, 2, 3], 0, 3, 1)

    def test_rec_linear_search_empty_sequence(self):
        '''Verify an exception is raised for an empty sequence.'''
        with pytest.raises(Exception, match='Invalid upper or lower bound!'):
            rec_linear_search([], 0, 0, 1)

    def test_rec_linear_search_duplicate_values_returns_first(self):
        '''Verify the first occurrence is returned when duplicates exist.'''
        assert rec_linear_search([1, 2, 2, 3], 0, 3, 2) == 1
