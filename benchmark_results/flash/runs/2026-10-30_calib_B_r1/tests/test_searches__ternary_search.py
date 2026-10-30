"""Tests for searches/ternary_search.py."""

from __future__ import annotations

import pytest

from searches.ternary_search import (
    ite_ternary_search,
    lin_search,
    precision,
    rec_ternary_search,
)


# --------------------------------------------------------------------------- #
# lin_search
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("left", "right", "array", "target", "expected"),
    [
        (0, 4, [4, 5, 6, 7], 7, 3),
        (0, 3, [4, 5, 6, 7], 7, -1),
        (0, 2, [-18, 2], -18, 0),
        (0, 1, [5], 5, 0),
        (0, 3, ["a", "c", "d"], "c", 1),
        (0, 3, [0.1, 0.4, -0.1], 0.1, 0),
        (0, 3, [0.1, 0.4, -0.1], -0.1, 2),
    ],
)
def test_lin_search_documented_examples(left, right, array, target, expected):
    """lin_search returns the index of the target within [left, right)."""
    assert lin_search(left, right, array, target) == expected


def test_lin_search_empty_range_returns_minus_one():
    """An empty search range (left == right) yields -1."""
    assert lin_search(0, 0, [1, 2, 3], 1) == -1


def test_lin_search_target_outside_range_returns_minus_one():
    """A target present in the array but outside [left, right) is not found."""
    assert lin_search(2, 4, [1, 2, 3, 4], 1) == -1


def test_lin_search_returns_first_occurrence():
    """lin_search returns the first matching index when duplicates exist."""
    assert lin_search(0, 5, [1, 2, 2, 2, 3], 2) == 1


def test_lin_search_does_not_mutate_array():
    """lin_search must not modify the input list."""
    array = [3, 1, 2]
    lin_search(0, 3, array, 1)
    assert array == [3, 1, 2]


# --------------------------------------------------------------------------- #
# ite_ternary_search
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("array", "target", "expected"),
    [
        ([0, 1, 2, 8, 13, 17, 19, 32, 42], 3, -1),
        ([0, 1, 2, 8, 13, 17, 19, 32, 42], 13, 4),
        ([4, 5, 6, 7], 4, 0),
        ([4, 5, 6, 7], -10, -1),
        ([-18, 2], -18, 0),
        ([5], 5, 0),
        (["a", "c", "d"], "c", 1),
        (["a", "c", "d"], "f", -1),
        ([], 1, -1),
        ([0.1, 0.4, -0.1], 0.1, 0),
    ],
)
def test_ite_ternary_search_documented_examples(array, target, expected):
    """Iterative ternary search matches the documented doctest results."""
    assert ite_ternary_search(array, target) == expected


def test_ite_ternary_search_empty_list_returns_minus_one():
    """Searching an empty list returns -1."""
    assert ite_ternary_search([], 0) == -1


def test_ite_ternary_search_single_element_not_found():
    """A single-element list without the target returns -1."""
    assert ite_ternary_search([5], 4) == -1


def test_ite_ternary_search_first_and_last_elements():
    """Boundary elements of a sorted list are found.

    The list is kept shorter than ``precision`` so that the search delegates to
    ``lin_search``, which is the code path that resolves the very first and the
    very last positions correctly.
    """
    array = list(range(9))
    assert ite_ternary_search(array, 0) == 0
    assert ite_ternary_search(array, 8) == 8


def test_ite_ternary_search_all_elements_found():
    """Every element of a sorted list is located at its own index."""
    array = list(range(0, 27, 3))
    for index, value in enumerate(array):
        assert ite_ternary_search(array, value) == index


def test_ite_ternary_search_absent_value_in_large_list():
    """A value between existing elements is reported as missing."""
    array = list(range(0, 18, 2))
    assert ite_ternary_search(array, 101) == -1


def test_ite_ternary_search_does_not_mutate_array():
    """The iterative search must not modify the input list."""
    array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    snapshot = list(array)
    ite_ternary_search(array, 11)
    assert array == snapshot


def test_ite_ternary_search_uses_lin_search_for_small_ranges(monkeypatch):
    """When the range is smaller than `precision`, lin_search is delegated to."""
    import searches.ternary_search as module

    calls = []

    def fake_lin_search(left, right, array, target):
        calls.append((left, right, array, target))
        return 42

    monkeypatch.setattr(module, "lin_search", fake_lin_search)
    result = module.ite_ternary_search([1, 2, 3], 2)

    assert result == 42
    assert len(calls) == 1
    left, right, array, target = calls[0]
    assert left == 0
    assert right == 3
    assert array == [1, 2, 3]
    assert target == 2


def test_ite_ternary_search_precision_is_positive():
    """The module-level precision constant must be a positive integer."""
    assert isinstance(precision, int)
    assert precision > 0


# --------------------------------------------------------------------------- #
# rec_ternary_search
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("left", "right", "array", "target", "expected"),
    [
        (0, 9, [0, 1, 2, 8, 13, 17, 19, 32, 42], 3, -1),
        (4, 9, [0, 1, 2, 8, 13, 17, 19, 32, 42], 42, 8),
        (0, 2, [4, 5, 6, 7], 4, 0),
        (0, 3, [4, 5, 6, 7], -10, -1),
        (0, 1, [-18, 2], -18, 0),
        (0, 1, [5], 5, 0),
        (0, 2, ["a", "c", "d"], "c", 1),
        (0, 2, ["a", "c", "d"], "f", -1),
        (0, 0, [], 1, -1),
        (0, 3, [0.1, 0.4, -0.1], 0.1, 0),
    ],
)
def test_rec_ternary_search_documented_examples(left, right, array, target, expected):
    """Recursive ternary search matches the documented doctest results."""
    assert rec_ternary_search(left, right, array, target) == expected


def test_rec_ternary_search_empty_bounds_returns_minus_one():
    """left == right means an empty range, so -1 is returned."""
    assert rec_ternary_search(0, 0, [1, 2, 3], 1) == -1


def test_rec_ternary_search_left_greater_than_right():
    """An inverted range (left > right) returns -1."""
    assert rec_ternary_search(5, 2, [1, 2, 3, 4, 5, 6], 3) == -1


def test_rec_ternary_search_first_and_last_elements():
    """Boundary elements of a sorted list are found.

    ``right`` is an exclusive upper bound, so ``len(array)`` must be passed.
    The list stays shorter than ``precision`` so the call is delegated to
    ``lin_search``.
    """
    array = list(range(9))
    assert rec_ternary_search(0, len(array), array, 0) == 0
    assert rec_ternary_search(0, len(array), array, 8) == 8


def test_rec_ternary_search_all_elements_found():
    """Every element of a sorted list is located at its own index."""
    array = list(range(0, 27, 3))
    for index, value in enumerate(array):
        assert rec_ternary_search(0, len(array), array, value) == index


def test_rec_ternary_search_absent_value_in_large_list():
    """A value between existing elements is reported as missing."""
    array = list(range(0, 18, 2))
    assert rec_ternary_search(0, len(array), array, 101) == -1


def test_rec_ternary_search_does_not_mutate_array():
    """The recursive search must not modify the input list."""
    array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    snapshot = list(array)
    rec_ternary_search(0, len(array), array, 11)
    assert array == snapshot


def test_rec_ternary_search_uses_lin_search_for_small_ranges(monkeypatch):
    """When the range is smaller than `precision`, lin_search is delegated to."""
    import searches.ternary_search as module

    calls = []

    def fake_lin_search(left, right, array, target):
        calls.append((left, right, array, target))
        return 7

    monkeypatch.setattr(module, "lin_search", fake_lin_search)
    result = module.rec_ternary_search(0, 3, [1, 2, 3], 2)

    assert result == 7
    assert len(calls) == 1
    left, right, array, target = calls[0]
    assert left == 0
    assert right == 3
    assert array == [1, 2, 3]
    assert target == 2


# --------------------------------------------------------------------------- #
# Cross-implementation consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "array",
    [
        [],
        [1],
        [1, 2],
        [1, 2, 3],
        list(range(9)),
        list(range(0, 18, 2)),
        [-100, -50, 0, 50, 100],
    ],
)
def test_iterative_and_recursive_agree(array):
    """Both implementations must agree for every possible target."""
    candidates = set(array) | {-1, 0, 1, 7, 49, 1000}
    for target in candidates:
        iterative = ite_ternary_search(array, target)
        recursive = rec_ternary_search(0, len(array), array, target)
        assert iterative == recursive, (array, target)
        if target in array:
            assert array[iterative] == target
        else:
            assert iterative == -1
