'''Tests for the pure Python quick sort implementation.'''

from __future__ import annotations

import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort

# The module object that owns ``quick_sort``. Patching attributes on this
# object avoids the string-based import resolution performed by
# ``unittest.mock.patch('sorts.quick_sort.randrange')``, which fails when the
# package is only importable through pytest's sys.path manipulation.
quick_sort_module = sys.modules[quick_sort.__module__]


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list():
    '''A typical unsorted list of integers is returned in ascending order.'''
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_mutates_input_by_consuming_pivot():
    '''quick_sort() pops the pivot from the caller list, so the input is mutated.'''
    original = [4, 1, 3, 2]
    snapshot = list(original)

    result = quick_sort(original)

    assert result == [1, 2, 3, 4]
    # Exactly one element (the pivot) is removed from the caller's list.
    assert len(original) == len(snapshot) - 1


def test_quick_sort_handles_negative_numbers():
    '''Negative values are ordered correctly relative to positives.'''
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats():
    '''Floating point values are sorted in ascending order.'''
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings():
    '''Strings are sorted lexicographically.'''
    assert quick_sort(['pear', 'apple', 'banana']) == ['apple', 'banana', 'pear']


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'collection',
    [
        pytest.param([], id='empty'),
        pytest.param([42], id='single-element'),
    ],
)
def test_quick_sort_base_cases_return_input(collection):
    '''Collections with fewer than two elements are returned unchanged.'''
    assert quick_sort(collection) == collection


def test_quick_sort_base_case_returns_same_object():
    '''The base case returns the very same list object (no copy is made).'''
    collection = [7]
    assert quick_sort(collection) is collection


def test_quick_sort_already_sorted():
    '''An already sorted list stays sorted.'''
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_reverse_sorted():
    '''A reverse sorted list is fully reversed into ascending order.'''
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_all_equal_elements():
    '''A list of identical elements is returned with the same length.'''
    result = quick_sort([3, 3, 3, 3])
    assert result == [3, 3, 3, 3]
    assert len(result) == 4


def test_quick_sort_duplicates_preserved():
    '''Duplicate values are all preserved (multiset equality holds).'''
    data = [2, 1, 2, 1, 2]
    snapshot = list(data)
    result = quick_sort(data)
    assert result == [1, 1, 2, 2, 2]
    assert sorted(result) == sorted(snapshot)


def test_quick_sort_large_random_list_matches_builtin_sorted():
    '''A large random list sorts identically to Python built-in sorted.'''
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Pivot selection (external dependency isolation)
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection():
    '''randrange is called to pick the pivot index for non-trivial input.'''
    with patch.object(quick_sort_module, 'randrange', return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    assert mock_randrange.called


def test_quick_sort_deterministic_with_patched_pivot():
    '''With a fixed pivot index the output is deterministic and correct.'''
    with patch.object(quick_sort_module, 'randrange', return_value=1):
        assert quick_sort([9, 4, 7, 1]) == [1, 4, 7, 9]


def test_quick_sort_does_not_call_randrange_for_base_cases():
    '''randrange must not be invoked when the collection has < 2 elements.'''
    with patch.object(quick_sort_module, 'randrange') as mock_randrange:
        assert quick_sort([]) == []
        assert quick_sort([1]) == [1]

    mock_randrange.assert_not_called()


def test_quick_sort_randrange_called_with_current_length():
    '''randrange receives the length of the collection being partitioned.'''
    with patch.object(quick_sort_module, 'randrange', return_value=0) as mock_randrange:
        quick_sort([5, 6, 7])

    # First call is on the full 3-element list.
    assert mock_randrange.call_args_list[0].args == (3,)


# ---------------------------------------------------------------------------
# Type / error behaviour
# ---------------------------------------------------------------------------


def test_quick_sort_raises_on_incomparable_types():
    '''Mixing incomparable types raises TypeError during partitioning.'''
    with pytest.raises(TypeError):
        quick_sort([1, 'a', 2])


def test_quick_sort_raises_on_non_list_input():
    '''Passing a non-list (e.g. a tuple) raises AttributeError on .pop().'''
    with pytest.raises(AttributeError):
        quick_sort((3, 1, 2))
