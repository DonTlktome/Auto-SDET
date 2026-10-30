"""Tests for the insertion_sort module.

The module under test is imported by file path so that the test suite does not
depend on the package layout of the project.
"""

from __future__ import annotations

import importlib.util
import random
import string
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Locate the module under test by absolute path, searching upward from the
# test file so the suite works regardless of the checkout layout.
# ---------------------------------------------------------------------------
def _find_source_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = []
    for base in (here, *here.parents):
        candidates.append(base / "sorts" / "insertion_sort.py")
        candidates.append(base / "insertion_sort.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(
        r"D:\Documents\Study\Projects\algorithms-bench\sorts\insertion_sort.py"
    )


_SOURCE_PATH = _find_source_path()


def _load_module():
    """Load the insertion_sort module from its source file."""
    spec = importlib.util.spec_from_file_location("insertion_sort_under_test", _SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


insertion_sort_module = _load_module()
insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------
def test_sorts_unsorted_list_of_ints():
    """A plain unsorted list of integers is sorted in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_sorts_list_of_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_negative_numbers():
    """Negative integers are handled correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([3.5, -1.25, 0.0, 2.0]) == [-1.25, 0.0, 2.0, 3.5]


def test_sorts_tuples_by_first_element():
    """Tuples are compared lexicographically, so sorting works on them too."""
    data = [(2, "b"), (1, "a"), (2, "a")]
    assert insertion_sort(data) == [(1, "a"), (2, "a"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([1, 1, 1, 1], id="all-equal"),
        pytest.param([1, 2, 3, 4, 5], id="already-sorted"),
        pytest.param([5, 4, 3, 2, 1], id="reverse-sorted"),
        pytest.param([0, -1, 0, -1], id="duplicates-with-negatives"),
    ],
)
def test_edge_case_collections(collection):
    """Edge-case collections are sorted identically to the builtin ``sorted``."""
    assert insertion_sort(list(collection)) == sorted(collection)


def test_empty_list_returns_empty_list():
    """An empty collection is returned unchanged."""
    result = insertion_sort([])
    assert result == []


def test_single_element_list_is_unchanged():
    """A one-element collection is returned as-is."""
    assert insertion_sort([7]) == [7]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same object."""
    original = [3, 1, 2]
    result = insertion_sort(original)
    assert result is original
    assert original == [1, 2, 3]


def test_does_not_change_length():
    """Sorting never adds or removes elements."""
    data = [4, 4, 2, 9, 0, 2]
    length_before = len(data)
    insertion_sort(data)
    assert len(data) == length_before


def test_works_with_mutable_sequence_subclass():
    """Any ``MutableSequence`` implementation (e.g. ``bytearray``) is supported."""
    data = bytearray(b"cab")
    result = insertion_sort(data)
    assert result is data
    assert bytes(data) == b"abc"


# ---------------------------------------------------------------------------
# Randomized / property-style tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 42, 1234])
def test_matches_builtin_sorted_for_random_ints(seed):
    """Random integer collections are sorted exactly like ``sorted``."""
    rng = random.Random(seed)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


@pytest.mark.parametrize("seed", [0, 7, 99])
def test_matches_builtin_sorted_for_random_strings(seed):
    """Random string collections are sorted exactly like ``sorted``."""
    rng = random.Random(seed)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


def test_result_is_non_decreasing():
    """Every adjacent pair in the result satisfies ``a <= b``."""
    rng = random.Random(2024)
    collection = [rng.randint(-1000, 1000) for _ in range(200)]
    insertion_sort(collection)
    assert all(a <= b for a, b in zip(collection, collection[1:]))


def test_result_is_a_permutation_of_input():
    """Sorting preserves the multiset of elements."""
    rng = random.Random(5)
    collection = [rng.randint(0, 20) for _ in range(150)]
    expected = sorted(collection)
    insertion_sort(collection)
    assert collection == expected


# ---------------------------------------------------------------------------
# Comparison-count / algorithm-behaviour checks
# ---------------------------------------------------------------------------
def test_already_sorted_input_performs_no_shifts():
    """A sorted input requires only one comparison per element (no shifting)."""

    class CountingInt(int):
        comparisons = 0

        def __lt__(self, other, /):
            type(self).comparisons += 1
            return int.__lt__(self, other)

    CountingInt.comparisons = 0
    data = [CountingInt(i) for i in range(10)]
    insertion_sort(data)
    assert data == list(range(10))
    # n - 1 comparisons for an already sorted sequence.
    assert CountingInt.comparisons == 9


def test_reverse_sorted_input_performs_quadratic_comparisons():
    """A reverse-sorted input triggers the worst-case comparison count."""

    class CountingInt(int):
        comparisons = 0

        def __lt__(self, other, /):
            type(self).comparisons += 1
            return int.__lt__(self, other)

    n = 10
    CountingInt.comparisons = 0
    data = [CountingInt(i) for i in range(n, 0, -1)]
    insertion_sort(data)
    assert data == list(range(1, n + 1))
    assert CountingInt.comparisons == n * (n - 1) // 2


# ---------------------------------------------------------------------------
# Isolation / no side effects
# ---------------------------------------------------------------------------
def test_does_not_mutate_other_objects():
    """Sorting one collection leaves unrelated collections untouched."""
    target = [3, 1, 2]
    other = [9, 8, 7]
    insertion_sort(target)
    assert other == [9, 8, 7]


def test_does_not_call_input_or_print():
    """The sorting function itself performs no I/O."""
    with patch("builtins.input") as mock_input, patch("builtins.print") as mock_print:
        insertion_sort([2, 1])
    mock_input.assert_not_called()
    mock_print.assert_not_called()


def test_raises_type_error_for_incomparable_items():
    """Mixing incomparable types raises ``TypeError`` from the comparison."""
    with pytest.raises(TypeError):
        insertion_sort([1, "a", 2])


def test_raises_type_error_for_immutable_sequence():
    """Passing an immutable sequence (tuple) fails on item assignment."""
    with pytest.raises(TypeError):
        insertion_sort((3, 1, 2))


# ---------------------------------------------------------------------------
# Doctest coverage
# ---------------------------------------------------------------------------
def test_module_doctests_pass():
    """All doctests embedded in the module execute successfully."""
    import doctest

    results = doctest.testmod(insertion_sort_module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
