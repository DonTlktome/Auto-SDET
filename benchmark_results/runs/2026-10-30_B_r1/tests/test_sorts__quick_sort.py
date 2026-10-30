'''
Pytest test suite for the quick_sort module.

Covers the public `quick_sort` function with positive, edge, and boundary
cases, and verifies sorting behaviour with mutable inputs.
'''

from __future__ import annotations

import pytest

from quick_sort import quick_sort


@pytest.mark.parametrize(
    'unsorted, expected',
    [
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([3, 1, 2], [1, 2, 3]),
        ([10, -1, 0, 7, 3], [-1, 0, 3, 7, 10]),
    ],
)
def test_quick_sort_positive_cases(unsorted, expected):
    '''Test that quick_sort correctly sorts various unsorted lists.'''
    assert quick_sort(unsorted) == expected


@pytest.mark.parametrize(
    'unsorted, expected',
    [
        ([], []),
        ([42], [42]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([-2, 5, 0, -45], [-45, -2, 0, 5]),
    ],
)
def test_quick_sort_edge_cases(unsorted, expected):
    '''Test quick_sort on empty, single-element, and duplicate-heavy lists.'''
    assert quick_sort(unsorted) == expected


def test_quick_sort_preserves_original_when_given_copy():
    '''Test that passing a copy to quick_sort leaves the original list unchanged.'''
    original = [3, 1, 2]
    snapshot = original[:]
    result = quick_sort(original[:])
    assert result == [1, 2, 3]
    assert original == snapshot


def test_quick_sort_large_list():
    '''Test quick_sort on a larger list of integers.'''
    unsorted = list(range(100, 0, -1))
    assert quick_sort(unsorted) == list(range(1, 101))


def test_quick_sort_with_strings():
    '''Test quick_sort works with comparable non-integer items (strings).'''
    unsorted = ['banana', 'apple', 'cherry', 'apple']
    assert quick_sort(unsorted) == ['apple', 'apple', 'banana', 'cherry']
