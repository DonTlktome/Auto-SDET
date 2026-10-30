"""
Tests for the pure Python heap sort implementation in `sorts/heap_sort.py`.

The tests cover:
  * `heapify`  - sift-down behaviour on a max-heap
  * `heap_sort` - full sorting behaviour, edge cases and boundaries

All tests are self-contained and require no network access.
"""

import random

import pytest

from sorts.heap_sort import heap_sort, heapify


# --------------------------------------------------------------------------- #
# heapify
# --------------------------------------------------------------------------- #
def test_heapify_sifts_root_down_to_restore_max_heap():
    """A root smaller than its children must be swapped down (happy path)."""
    unsorted = [1, 4, 3, 5, 2]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [4, 5, 3, 1, 2]


def test_heapify_is_idempotent_once_heap_property_holds():
    """Calling heapify again on an already valid max-heap changes nothing."""
    unsorted = [4, 5, 3, 1, 2]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [5, 4, 3, 1, 2]

    snapshot = list(unsorted)
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == snapshot


def test_heapify_returns_none():
    """`heapify` mutates in place and must return None."""
    unsorted = [1, 2, 3]
    assert heapify(unsorted, 0, len(unsorted)) is None


def test_heapify_leaf_node_is_noop():
    """A leaf node has no children, so the list must stay untouched."""
    unsorted = [9, 8, 7, 6, 5]
    heapify(unsorted, 4, len(unsorted))
    assert unsorted == [9, 8, 7, 6, 5]


def test_heapify_single_element_list():
    """A one-element heap is trivially valid and must not be modified."""
    unsorted = [42]
    heapify(unsorted, 0, 1)
    assert unsorted == [42]


def test_heapify_empty_list():
    """An empty list must not raise even though index 0 does not exist."""
    unsorted = []
    heapify(unsorted, 0, 0)
    assert unsorted == []


def test_heapify_respects_heap_size_boundary():
    """Elements at or beyond `heap_size` must be ignored by the sift-down."""
    unsorted = [1, 2, 100]
    # heap_size == 2 means index 2 (value 100) is outside the heap.
    heapify(unsorted, 0, 2)
    assert unsorted == [2, 1, 100]


def test_heapify_chooses_larger_right_child():
    """When the right child is the largest, it must be promoted to the root."""
    unsorted = [1, 2, 3]
    heapify(unsorted, 0, 3)
    assert unsorted == [3, 2, 1]


def test_heapify_chooses_larger_left_child():
    """When the left child is the largest, it must be promoted to the root."""
    unsorted = [1, 3, 2]
    heapify(unsorted, 0, 3)
    assert unsorted == [3, 1, 2]


def test_heapify_equal_children_do_not_swap():
    """Equal children are not strictly greater, so no swap should occur."""
    unsorted = [1, 5, 5]
    heapify(unsorted, 0, 3)
    assert unsorted == [5, 1, 5]


def test_heapify_deep_recursion_restores_property():
    """A small value at the root of a deep heap must sink all the way down."""
    unsorted = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    heapify(unsorted, 0, len(unsorted))
    # Root must now be the maximum of the whole heap.
    assert unsorted[0] == 9
    # And the max-heap property must hold for every internal node.
    for i in range(len(unsorted) // 2):
        left, right = 2 * i + 1, 2 * i + 2
        if left < len(unsorted):
            assert unsorted[i] >= unsorted[left]
        if right < len(unsorted):
            assert unsorted[i] >= unsorted[right]


@pytest.mark.parametrize(
    "unsorted, index, heap_size",
    [
        ([1, 4, 3, 5, 2], 0, 5),
        ([1, 4, 3, 5, 2], 1, 5),
        ([1, 4, 3, 5, 2], 2, 5),
        ([5, 4, 3, 2, 1], 0, 5),
        ([1, 2, 3, 4, 5], 0, 5),
        ([7], 0, 1),
        ([], 0, 0),
    ],
)
def test_heapify_never_breaks_max_heap_property(unsorted, index, heap_size):
    """After heapify, the subtree rooted at `index` must satisfy the max-heap property.

    `heapify` (a sift-down) only guarantees the invariant when the child
    subtrees of `index` are already valid heaps, so those subtrees are built
    bottom-up first.  Without that precondition a single sift-down pass cannot
    restore the property for arbitrary input, which is not what is asserted
    here.
    """
    # Build the child subtrees of `index` into proper max-heaps (bottom-up).
    for i in range(heap_size // 2 - 1, index, -1):
        heapify(unsorted, i, heap_size)

    heapify(unsorted, index, heap_size)

    # Every node from `index` downwards must now be >= both of its children.
    for i in range(index, heap_size // 2):
        left, right = 2 * i + 1, 2 * i + 2
        if left < heap_size:
            assert unsorted[i] >= unsorted[left]
        if right < heap_size:
            assert unsorted[i] >= unsorted[right]


# --------------------------------------------------------------------------- #
# heap_sort
# --------------------------------------------------------------------------- #
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
def test_heap_sort_returns_ascending_order(unsorted, expected):
    """`heap_sort` must return the collection ordered ascending (happy path + edges)."""
    assert heap_sort(unsorted) == expected


def test_heap_sort_empty_list():
    """An empty list must be returned as an empty list."""
    assert heap_sort([]) == []


def test_heap_sort_single_element():
    """A single-element list is already sorted."""
    assert heap_sort([7]) == [7]


def test_heap_sort_two_elements_unsorted():
    """The smallest non-trivial case: two elements out of order."""
    assert heap_sort([2, 1]) == [1, 2]


def test_heap_sort_all_negative_numbers():
    """Negative numbers must sort correctly (boundary around zero)."""
    assert heap_sort([-1, -100, -50, -2]) == [-100, -50, -2, -1]


def test_heap_sort_mixed_signs_including_zero():
    """Zero must be placed between negative and positive values."""
    assert heap_sort([1, -1, 0]) == [-1, 0, 1]


def test_heap_sort_duplicates_preserved():
    """Duplicate values must all be retained (multiset preserved)."""
    unsorted = [3, 1, 3, 2, 1, 3]
    result = heap_sort(unsorted)
    assert result == [1, 1, 2, 3, 3, 3]
    assert len(result) == len(unsorted)


def test_heap_sort_sorts_in_place_and_returns_same_object():
    """`heap_sort` mutates the input list and returns that same object."""
    unsorted = [3, 1, 2]
    result = heap_sort(unsorted)
    assert result is unsorted
    assert unsorted == [1, 2, 3]


def test_heap_sort_does_not_change_length():
    """Sorting must never add or drop elements."""
    unsorted = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    original_length = len(unsorted)
    heap_sort(unsorted)
    assert len(unsorted) == original_length


def test_heap_sort_large_random_list_matches_builtin_sorted():
    """A larger pseudo-random list must match Python's `sorted` output."""
    rng = random.Random(12345)
    unsorted = [rng.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(unsorted)
    assert heap_sort(unsorted) == expected


def test_heap_sort_already_sorted_input():
    """Already sorted input must remain sorted (best-case-ish boundary)."""
    unsorted = list(range(50))
    assert heap_sort(unsorted) == list(range(50))


def test_heap_sort_reverse_sorted_input():
    """Reverse sorted input must be fully reversed into ascending order."""
    unsorted = list(range(50, 0, -1))
    assert heap_sort(unsorted) == list(range(1, 51))


def test_heap_sort_does_not_mutate_other_lists():
    """Sorting one list must not affect an unrelated list."""
    unsorted = [3, 1, 2]
    other = [9, 8, 7]
    heap_sort(unsorted)
    assert other == [9, 8, 7]


def test_heap_sort_works_with_floats():
    """The algorithm is comparison based, so floats must sort correctly too."""
    assert heap_sort([2.5, -1.0, 0.0, 3.25]) == [-1.0, 0.0, 2.5, 3.25]


def test_heap_sort_works_with_strings():
    """Comparable heterogeneous items (strings) must sort lexicographically."""
    assert heap_sort(["pear", "apple", "banana"]) == ["apple", "banana", "pear"]
