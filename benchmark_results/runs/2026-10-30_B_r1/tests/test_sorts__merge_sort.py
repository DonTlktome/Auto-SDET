'''
Tests for the merge_sort module.

Covers the public ``merge_sort`` function with positive, edge, and
boundary cases, including parametrized inputs and isolation of the
interactive ``__main__`` block.
'''

import runpy

import pytest

from merge_sort import merge_sort


class TestMergeSort:
    '''Tests for the merge_sort function.'''

    @pytest.mark.parametrize(
        'unsorted, expected',
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([3, 1, 2], [1, 2, 3]),
            ([9, 7, 5, 3, 1, 0], [0, 1, 3, 5, 7, 9]),
        ],
    )
    def test_merge_sort_positive(self, unsorted, expected):
        '''Test that merge_sort correctly sorts various unsorted lists.'''
        assert merge_sort(unsorted) == expected

    @pytest.mark.parametrize(
        'unsorted, expected',
        [
            ([], []),
            ([42], [42]),
            ([-2, -5, -45], [-45, -5, -2]),
            ([0, 0, 0, 0], [0, 0, 0, 0]),
            ([1, -1, 1, -1], [-1, -1, 1, 1]),
        ],
    )
    def test_merge_sort_edge_cases(self, unsorted, expected):
        '''Test merge_sort on empty, single-element, negative, and duplicate inputs.'''
        assert merge_sort(unsorted) == expected

    def test_merge_sort_does_not_mutate_original(self):
        '''Test that merge_sort returns a new list and leaves the input unchanged.'''
        original = [3, 1, 2]
        result = merge_sort(original)
        assert result == [1, 2, 3]
        assert original == [3, 1, 2]

    def test_merge_sort_large_input(self):
        '''Test merge_sort on a larger reversed list to exercise recursion depth.'''
        unsorted = list(range(1000, 0, -1))
        assert merge_sort(unsorted) == list(range(1, 1001))

    def test_merge_sort_mixed_types_comparable(self):
        '''Test merge_sort with comparable mixed numeric types (int and float).'''
        unsorted = [3.5, 1, 2.0, 4]
        assert merge_sort(unsorted) == [1, 2.0, 3.5, 4]


class TestMainBlock:
    '''Tests for the interactive ``__main__`` block.'''

    def test_main_doctest_and_valid_input(self, monkeypatch, capsys):
        '''Test the __main__ block with valid comma-separated integer input.'''
        import merge_sort as ms

        monkeypatch.setattr('builtins.input', lambda _: '5,3,8,1\n')
        runpy.run_path(ms.__file__, run_name='__main__')
        captured = capsys.readouterr()
        assert '1,3,5,8' in captured.out

    def test_main_invalid_input(self, monkeypatch, capsys):
        '''Test the __main__ block handles invalid (non-integer) input gracefully.'''
        import merge_sort as ms

        monkeypatch.setattr('builtins.input', lambda _: 'a,b,c\n')
        runpy.run_path(ms.__file__, run_name='__main__')
        captured = capsys.readouterr()
        assert 'Invalid input' in captured.out
