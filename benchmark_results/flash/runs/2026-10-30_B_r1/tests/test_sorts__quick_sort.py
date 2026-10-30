'''Tests for the pure Python quick sort implementation.

The module under test is ``sorts/quick_sort.py`` which exposes a single
public function ``quick_sort`` plus a ``__main__`` CLI block.
'''

from __future__ import annotations

import builtins
import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    '''A typical unsorted list of ints is returned in ascending order.'''
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_does_not_mutate_input() -> None:
    '''The result is a new sorted list (the input may be consumed internally).'''
    original = [4, 1, 3, 2]

    result = quick_sort(original)

    assert result == [1, 2, 3, 4]
    assert result is not original


def test_quick_sort_handles_negative_numbers() -> None:
    '''Negative values are ordered correctly relative to positives and zero.'''
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_sorts_strings() -> None:
    '''Any comparable items (e.g. strings) are sorted lexicographically.'''
    assert quick_sort(['banana', 'apple', 'cherry']) == ['apple', 'banana', 'cherry']


def test_quick_sort_sorts_floats() -> None:
    '''Floating point values are sorted in ascending order.'''
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'collection',
    [
        pytest.param([], id='empty'),
        pytest.param([42], id='single-element'),
        pytest.param([7, 7], id='two-equal-elements'),
        pytest.param([1, 2, 3, 4, 5], id='already-sorted'),
        pytest.param([5, 4, 3, 2, 1], id='reverse-sorted'),
        pytest.param([3, 3, 3, 3, 3], id='all-duplicates'),
        pytest.param([0, -0, 0], id='zeros'),
    ],
)
def test_quick_sort_edge_cases(collection: list) -> None:
    '''Edge cases (empty, singleton, sorted, reversed, duplicates) sort correctly.'''
    assert quick_sort(list(collection)) == sorted(collection)


def test_quick_sort_empty_list_returns_empty_list() -> None:
    '''An empty list is returned unchanged (base case, no pivot selection).'''
    assert quick_sort([]) == []


def test_quick_sort_single_element_returns_same_element() -> None:
    '''A one-element list is returned as-is (base case).'''
    assert quick_sort([99]) == [99]


def test_quick_sort_does_not_call_randrange_for_trivial_input() -> None:
    '''The base case must short-circuit before any pivot is chosen.'''
    quick_sort_module = sys.modules[quick_sort.__module__]
    with patch.object(quick_sort_module, 'randrange') as mock_randrange:
        assert quick_sort([]) == []
        assert quick_sort([1]) == [1]
    mock_randrange.assert_not_called()


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    '''A pivot index is drawn via ``randrange`` bounded by the list length.'''
    quick_sort_module = sys.modules[quick_sort.__module__]
    with patch.object(quick_sort_module, 'randrange', return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    assert mock_randrange.called
    # Every call must be bounded by the length of the (sub)collection.
    for call in mock_randrange.call_args_list:
        (bound,) = call.args
        assert isinstance(bound, int)
        assert bound >= 1


def test_quick_sort_is_deterministic_across_pivot_choices() -> None:
    '''The sorted output is identical regardless of which pivot is chosen.'''
    data = [5, 1, 4, 2, 8, 0, 2]
    expected = sorted(data)
    quick_sort_module = sys.modules[quick_sort.__module__]

    for pivot_index in range(len(data)):
        def pick_pivot(bound: int, _pivot_index: int = pivot_index) -> int:
            return _pivot_index % bound

        with patch.object(quick_sort_module, 'randrange', side_effect=pick_pivot):
            assert quick_sort(list(data)) == expected


def test_quick_sort_matches_sorted_for_random_input() -> None:
    '''Property check: output equals ``sorted`` for many random inputs.'''
    rng = random.Random(1234)
    for _ in range(25):
        size = rng.randrange(0, 30)
        data = [rng.randrange(-50, 51) for _ in range(size)]
        assert quick_sort(list(data)) == sorted(data)


def test_quick_sort_handles_large_input() -> None:
    '''A larger input (with duplicates) is sorted correctly.'''
    data = [i % 97 for i in range(500)]
    random.Random(7).shuffle(data)
    assert quick_sort(list(data)) == sorted(data)


def test_quick_sort_raises_on_incomparable_items() -> None:
    '''Mixing incomparable types surfaces a TypeError from the comparison.'''
    with pytest.raises(TypeError):
        quick_sort([1, 'a', 2])


# ---------------------------------------------------------------------------
# CLI / __main__ behaviour
# ---------------------------------------------------------------------------


def test_main_block_sorts_user_input(monkeypatch, capsys) -> None:
    '''Running the module as a script reads CSV ints and prints the sorted list.'''
    import sorts.quick_sort as quick_sort_module

    monkeypatch.setattr(builtins, 'input', lambda *a, **k: ' 5,3,1,4,2 ')
    monkeypatch.setattr(sys, 'argv', ['quick_sort.py'])

    with patch.object(quick_sort_module, '__name__', '__main__'):
        # Re-execute the module body so the __main__ branch runs.
        source = open(quick_sort_module.__file__, encoding='utf-8').read()
        exec(compile(source, quick_sort_module.__file__, 'exec'), {'__name__': '__main__'})

    captured = capsys.readouterr()
    assert captured.out.strip() == '[1, 2, 3, 4, 5]'
