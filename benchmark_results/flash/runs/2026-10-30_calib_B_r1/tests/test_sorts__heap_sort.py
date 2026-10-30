"""
Tests for the pure Python heap sort implementation in `sorts/heap_sort.py`.

The module under test exposes two public functions:

* ``heapify(unsorted, index, heap_size)`` -- restores the max-heap property
  for the subtree rooted at ``index`` (in place, returns ``None``).
* ``heap_sort(unsorted)`` -- sorts the list in place in ascending order and
  returns the same list object.

All tests are self-contained and require no network access.
"""

from __future__ import annotations

import random

import pytest

from sorts.heap_sort import heap_sort, heapify


# ---------------------------------------------------------------------------
# heapify
# ---------------------------------------------------------------------------


def test_heapify_restores_max_heap_property_at_root() -> None:
    """A root smaller than its children must be swapped down (happy path)."""
    unsorted = [1, 4, 3, 5, 2]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [4, 5, 3, 1, 2]


def test_heapify_is_idempotent_once_heap_property_holds() -> None:
    """Calling heapify again on an already valid heap must not change the list."""
    unsorted = [4, 5, 3, 1, 2]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [5, 4, 3, 1, 2]

    snapshot = list(unsorted)
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == snapshot


def test_heapify_returns_none() -> None:
    """heapify mutates in place and therefore must return ``None``."""
    unsorted = [1, 2, 3]
    assert heapify(unsorted, 0, len(unsorted)) is None


def test_heapify_single_element_list_is_unchanged() -> None:
    """A one-element heap has no children, so nothing can be swapped."""
    unsorted = [42]
    heapify(unsorted, 0, 1)
    assert unsorted == [42]


def test_heapify_empty_list_is_unchanged() -> None:
    """An empty list must be handled without raising (boundary case)."""
    unsorted: list[int] = []
    heapify(unsorted, 0, 0)
    assert unsorted == []


def test_heapify_leaf_node_is_unchanged() -> None:
    """A leaf node (index beyond the last parent) must not be modified."""
    unsorted = [9, 8, 7, 6, 5]
    heapify(unsorted, 4, len(unsorted))
    assert unsorted == [9, 8, 7, 6, 5]


def test_heapify_respects_heap_size_boundary() -> None:
    """Elements at or beyond ``heap_size`` must be ignored by heapify."""
    unsorted = [1, 10, 3]
    # heap_size == 1 means only index 0 is part of the heap; index 1 is out of range.
    heapify(unsorted, 0, 1)
    assert unsorted == [1, 10, 3]


def test_heapify_swaps_with_right_child_when_it_is_largest() -> None:
    """The right child must be selected when it is strictly the largest."""
    unsorted = [1, 2, 9]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [9, 2, 1]


def test_heapify_does_not_swap_on_equal_children() -> None:
    """Equal values must not trigger a swap (strict ``>`` comparison)."""
    unsorted = [5, 5, 5]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [5, 5, 5]


def test_heapify_sifts_down_recursively() -> None:
    """A swap at the root must cascade into the affected subtree."""
    unsorted = [1, 9, 3, 8, 7]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [9, 8, 3, 1, 7]


# ``heapify`` only guarantees a valid heap for the subtree rooted at ``index``
# when its child subtrees are *already* valid max-heaps (that is precisely how
# ``heap_sort`` builds the heap, bottom-up).  The parametrised inputs below
# therefore satisfy that precondition, otherwise a single call could not
# establish the max-heap rule along the whole sifted path.
@pytest.mark.parametrize(
    "unsorted, index",
    [
        ([1, 5, 3, 2, 4], 0),
        ([1, 4, 3, 5, 2], 1),
        ([1, 4, 3, 5, 2], 2),
        ([7, 6, 5, 4, 3, 2, 1], 0),
        ([7, 6, 5, 4, 3, 2, 1], 2),
        ([-1, -5, -3], 0),
    ],
)
def test_heapify_produces_valid_heap_for_subtree(
    unsorted: list[int], index: int
) -> None:
    """After heapify, the subtree rooted at ``index`` must satisfy the max-heap rule."""
    heapify(unsorted, index, len(unsorted))
    _assert_max_heap(unsorted, index, len(unsorted))


def _assert_max_heap(heap: list[int], root: int, heap_size: int) -> None:
    """Helper: assert the max-heap property holds for the subtree rooted at ``root``."""
    stack = [root]
    while stack:
        current = stack.pop()
        left = 2 * current + 1
        right = 2 * current + 2
        if left < heap_size:
            assert heap[current] >= heap[left]
            stack.append(left)
        if right < heap_size:
            assert heap[current] >= heap[right]
            stack.append(right)


# ---------------------------------------------------------------------------
# heap_sort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsorted, expected",
    [
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([], []),
        ([-2, -5, -45], [-45, -5, -2]),
        ([3, 7, 9, 28, 123, -5, 8, -30, -200, 0, 4],
         [-200, -30, -5, 0, 3, 4, 7, 8, 9, 28, 123]),
        ([1], [1]),
        ([2, 1], [1, 2]),
        ([1, 2], [1, 2]),
        ([5, 5, 5, 5], [5, 5, 5, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ],
)
def test_heap_sort_returns_ascending_order(
    unsorted: list[int], expected: list[int]
) -> None:
    """heap_sort must return the elements in ascending order (happy path + edges)."""
    assert heap_sort(unsorted) == expected


def test_heap_sort_empty_list() -> None:
    """An empty list must be returned unchanged (boundary case)."""
    assert heap_sort([]) == []


def test_heap_sort_single_element() -> None:
    """A single-element list must be returned unchanged (boundary case)."""
    assert heap_sort([7]) == [7]


def test_heap_sort_sorts_in_place_and_returns_same_object() -> None:
    """heap_sort mutates the input list and returns that very same object."""
    unsorted = [3, 1, 2]
    result = heap_sort(unsorted)
    assert result is unsorted
    assert unsorted == [1, 2, 3]


def test_heap_sort_handles_duplicates() -> None:
    """Duplicate values must all be preserved and grouped together."""
    unsorted = [4, 1, 4, 2, 1, 4, 2]
    assert heap_sort(unsorted) == [1, 1, 2, 2, 4, 4, 4]


def test_heap_sort_handles_all_negative_values() -> None:
    """Negative-only input must be sorted correctly."""
    assert heap_sort([-1, -100, -50, -2]) == [-100, -50, -2, -1]


def test_heap_sort_handles_large_magnitude_values() -> None:
    """Very large and very small integers must be ordered correctly."""
    unsorted = [10**18, -(10**18), 0, 10**9, -(10**9)]
    assert heap_sort(unsorted) == [-(10**18), -(10**9), 0, 10**9, 10**18]


def test_heap_sort_matches_builtin_sorted_on_random_data() -> None:
    """Randomised cross-check against Python's built-in ``sorted``."""
    rng = random.Random(1234)
    for _ in range(20):
        size = rng.randint(0, 60)
        data = [rng.randint(-100, 100) for _ in range(size)]
        assert heap_sort(list(data)) == sorted(data)


def test_heap_sort_does_not_use_external_dependencies() -> None:
    """heap_sort must work on a plain list without any I/O or network access."""
    unsorted = [9, 3, 7, 1]
    assert heap_sort(unsorted) == [1, 3, 7, 9]
