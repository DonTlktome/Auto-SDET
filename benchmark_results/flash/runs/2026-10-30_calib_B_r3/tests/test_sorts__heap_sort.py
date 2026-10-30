"""
Tests for the pure Python heap sort implementation in `sorts/heap_sort.py`.

The module is imported by file path so the test file works regardless of the
package layout of the project (no `__init__.py` assumptions, no network access).
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Module loading
# --------------------------------------------------------------------------- #

_TEST_DIR = Path(__file__).resolve().parent


def _candidate_paths():
    """Yield plausible locations for `heap_sort.py`, nearest first."""
    bases = [_TEST_DIR, *_TEST_DIR.parents]
    for base in bases:
        yield base / "sorts" / "heap_sort.py"
        yield base / "heap_sort.py"


def _locate_module_path() -> Path:
    """Return the absolute path of `heap_sort.py`, searching the tree."""
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate

    # Last resort: walk the tree from the test directory upwards.
    for base in [_TEST_DIR, *_TEST_DIR.parents]:
        try:
            entries = list(base.rglob("heap_sort.py"))
        except OSError:  # pragma: no cover - unreadable directory
            continue
        for entry in entries:
            if entry.is_file():
                return entry

    raise FileNotFoundError(
        "Could not locate heap_sort.py starting from {!r}".format(str(_TEST_DIR))
    )


_MODULE_PATH = _locate_module_path()


def _load_module():
    """Load `heap_sort.py` from its absolute path as a standalone module."""
    module_dir = str(_MODULE_PATH.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("heap_sort_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


heap_sort_module = _load_module()
heapify = heap_sort_module.heapify
heap_sort = heap_sort_module.heap_sort


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _is_max_heap(values: list[int], index: int, heap_size: int) -> bool:
    """Return True if the subtree rooted at `index` satisfies the max-heap property."""
    left = 2 * index + 1
    right = 2 * index + 2
    if left < heap_size and values[left] > values[index]:
        return False
    if right < heap_size and values[right] > values[index]:
        return False
    ok = True
    if left < heap_size:
        ok = ok and _is_max_heap(values, left, heap_size)
    if right < heap_size:
        ok = ok and _is_max_heap(values, right, heap_size)
    return ok


# --------------------------------------------------------------------------- #
# heapify
# --------------------------------------------------------------------------- #

def test_heapify_positive_sift_down():
    """heapify must move the largest child up when the root is too small."""
    unsorted = [1, 4, 3, 5, 2]
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [4, 5, 3, 1, 2]


def test_heapify_repeated_call_reaches_heap_property():
    """Calling heapify twice on the same list must yield a valid max-heap."""
    unsorted = [1, 4, 3, 5, 2]
    heapify(unsorted, 0, len(unsorted))
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == [5, 4, 3, 1, 2]
    assert _is_max_heap(unsorted, 0, len(unsorted))


def test_heapify_returns_none():
    """heapify mutates in place and must return None."""
    unsorted = [3, 1, 2]
    assert heapify(unsorted, 0, len(unsorted)) is None


def test_heapify_already_valid_heap_is_unchanged():
    """A list that already satisfies the heap property must not be modified."""
    unsorted = [9, 5, 7, 1, 2]
    snapshot = list(unsorted)
    heapify(unsorted, 0, len(unsorted))
    assert unsorted == snapshot


def test_heapify_single_element_list():
    """A one-element list is trivially a heap and must stay untouched."""
    unsorted = [42]
    heapify(unsorted, 0, 1)
    assert unsorted == [42]


def test_heapify_empty_list_is_noop():
    """heapify on an empty list must not raise and must leave it empty."""
    unsorted: list[int] = []
    heapify(unsorted, 0, 0)
    assert unsorted == []


def test_heapify_leaf_index_is_noop():
    """heapify on a leaf node (no children) must not change the list."""
    unsorted = [10, 9, 8, 7]
    snapshot = list(unsorted)
    heapify(unsorted, 3, len(unsorted))
    assert unsorted == snapshot


def test_heapify_respects_heap_size_boundary():
    """Children at or beyond `heap_size` must be ignored."""
    unsorted = [1, 100, 2]
    # heap_size == 1 means only index 0 is part of the heap -> no swap allowed.
    heapify(unsorted, 0, 1)
    assert unsorted == [1, 100, 2]


def test_heapify_swaps_with_right_child_when_it_is_largest():
    """The right child must be chosen when it is strictly the largest."""
    unsorted = [1, 2, 3]
    heapify(unsorted, 0, 3)
    assert unsorted == [3, 2, 1]


def test_heapify_does_not_swap_on_equal_children():
    """Equal values must not trigger a swap (strict `>` comparison)."""
    unsorted = [5, 5, 5]
    snapshot = list(unsorted)
    heapify(unsorted, 0, 3)
    assert unsorted == snapshot


def test_heapify_negative_values():
    """heapify must work with negative integers."""
    unsorted = [-10, -3, -7]
    heapify(unsorted, 0, 3)
    assert unsorted == [-3, -10, -7]


@pytest.mark.parametrize(
    "unsorted, index, heap_size",
    [
        # NOTE: a single sift-down (as implemented by `heapify`) only yields a
        # valid heap when the child subtrees are already heaps, so the input
        # collections below are chosen to satisfy that precondition.
        ([1, 5, 3, 4, 2], 0, 5),
        ([1, 4, 3, 5, 2], 1, 5),
        ([1, 4, 3, 5, 2], 2, 5),
        ([7, 6, 5, 4, 3, 2, 1], 0, 7),
        ([2, 9, 7, 6, 5], 0, 5),
        ([0], 0, 1),
        ([], 0, 0),
    ],
)
def test_heapify_produces_valid_subtree_heap(unsorted, index, heap_size):
    """After heapify, the subtree rooted at `index` must satisfy the max-heap property."""
    heapify(unsorted, index, heap_size)
    assert _is_max_heap(unsorted, index, heap_size)


# --------------------------------------------------------------------------- #
# heap_sort
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "unsorted, expected",
    [
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([], []),
        ([1], [1]),
        ([-2, -5, -45], [-45, -5, -2]),
        ([3, 7, 9, 28, 123, -5, 8, -30, -200, 0, 4],
         [-200, -30, -5, 0, 3, 4, 7, 8, 9, 28, 123]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([2, 2, 2, 2], [2, 2, 2, 2]),
        ([0, 0, -1, 0], [-1, 0, 0, 0]),
    ],
)
def test_heap_sort_returns_ascending_order(unsorted, expected):
    """heap_sort must return the collection sorted in ascending order."""
    assert heap_sort(unsorted) == expected


def test_heap_sort_empty_list():
    """Sorting an empty list must return an empty list."""
    assert heap_sort([]) == []


def test_heap_sort_single_element():
    """Sorting a single-element list must return that same element."""
    assert heap_sort([7]) == [7]


def test_heap_sort_two_elements_unsorted():
    """Boundary case: the smallest non-trivial list must be sorted correctly."""
    assert heap_sort([2, 1]) == [1, 2]


def test_heap_sort_two_elements_sorted():
    """Boundary case: an already sorted two-element list stays unchanged."""
    assert heap_sort([1, 2]) == [1, 2]


def test_heap_sort_returns_same_object():
    """heap_sort sorts in place and returns the very same list object."""
    unsorted = [3, 1, 2]
    result = heap_sort(unsorted)
    assert result is unsorted


def test_heap_sort_mutates_input_in_place():
    """The caller's list must be mutated to the sorted order."""
    unsorted = [9, 1, 5, 3]
    heap_sort(unsorted)
    assert unsorted == [1, 3, 5, 9]


def test_heap_sort_does_not_change_length():
    """Sorting must preserve the number of elements."""
    unsorted = [4, 4, 1, 8, 8, 2]
    assert len(heap_sort(unsorted)) == 6


def test_heap_sort_preserves_multiset_of_values():
    """Sorting must be a permutation of the input (no lost or invented values)."""
    unsorted = [5, -1, 5, 0, 12, -1, 7]
    original = sorted(unsorted)
    assert heap_sort(unsorted) == original


def test_heap_sort_large_random_list_matches_builtin_sorted():
    """A larger pseudo-random list must match Python's `sorted` output."""
    rng = random.Random(1234)
    unsorted = [rng.randint(-1000, 1000) for _ in range(500)]
    expected = sorted(unsorted)
    assert heap_sort(unsorted) == expected


def test_heap_sort_all_equal_values():
    """A list of identical values must be returned unchanged."""
    unsorted = [3] * 50
    assert heap_sort(unsorted) == [3] * 50


def test_heap_sort_already_sorted_large_list():
    """An already ascending list must remain sorted (best-case-ish input)."""
    unsorted = list(range(100))
    assert heap_sort(unsorted) == list(range(100))


def test_heap_sort_reverse_sorted_large_list():
    """A descending list must be reversed into ascending order."""
    unsorted = list(range(100, 0, -1))
    assert heap_sort(unsorted) == list(range(1, 101))


def test_heap_sort_extreme_integer_values():
    """Very large and very small integers must be ordered correctly."""
    unsorted = [10**18, -(10**18), 0, 10**18 - 1, -(10**18) + 1]
    assert heap_sort(unsorted) == [-(10**18), -(10**18) + 1, 0, 10**18 - 1, 10**18]


def test_heap_sort_does_not_use_builtin_sorted(monkeypatch):
    """heap_sort must not delegate to the builtin `sorted` (isolation check)."""
    import builtins

    def _boom(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("builtin sorted() must not be used by heap_sort")

    monkeypatch.setattr(builtins, "sorted", _boom)
    assert heap_sort([3, 1, 2]) == [1, 2, 3]


def test_heap_sort_uses_heapify(monkeypatch):
    """heap_sort must drive the algorithm through the module-level `heapify`."""
    calls = []
    original = heap_sort_module.heapify

    def spy(unsorted, index, heap_size):
        calls.append((index, heap_size))
        return original(unsorted, index, heap_size)

    monkeypatch.setattr(heap_sort_module, "heapify", spy)
    heap_sort([3, 1, 2])
    assert calls, "heapify should have been called at least once"


def test_module_doctests_pass():
    """The doctests embedded in the module must all pass."""
    import doctest

    results = doctest.testmod(heap_sort_module, verbose=False)
    assert results.failed == 0
