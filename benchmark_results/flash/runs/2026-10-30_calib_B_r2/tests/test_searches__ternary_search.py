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
    """lin_search must reproduce all documented doctest examples."""
    assert lin_search(left, right, array, target) == expected


def test_lin_search_happy_path_middle_element():
    """lin_search finds an element located in the middle of the range."""
    assert lin_search(0, 5, [1, 2, 3, 4, 5], 3) == 2


def test_lin_search_empty_range_returns_minus_one():
    """An empty range (left == right) must yield -1 without touching the array."""
    assert lin_search(0, 0, [1, 2, 3], 1) == -1


def test_lin_search_respects_left_bound():
    """Elements before `left` must not be considered."""
    assert lin_search(2, 4, [1, 2, 3, 4], 1) == -1
    assert lin_search(2, 4, [1, 2, 3, 4], 3) == 2


def test_lin_search_right_bound_is_exclusive():
    """The element at index `right` is excluded from the search."""
    assert lin_search(0, 2, [1, 2, 3], 3) == -1


def test_lin_search_returns_first_occurrence():
    """With duplicates, the lowest matching index within the range is returned."""
    assert lin_search(0, 5, [7, 7, 7, 7, 7], 7) == 0


def test_lin_search_does_not_mutate_array():
    """lin_search must be read-only."""
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
    """ite_ternary_search must reproduce all documented doctest examples."""
    assert ite_ternary_search(array, target) == expected


def test_ite_ternary_search_single_element_not_found():
    """A one-element list that does not contain the target returns -1."""
    assert ite_ternary_search([5], 4) == -1


def test_ite_ternary_search_empty_list_returns_minus_one():
    """An empty list always returns -1."""
    assert ite_ternary_search([], 0) == -1


@pytest.mark.xfail(
    reason=(
        "Known defect in ite_ternary_search: once the search window shrinks "
        "below `precision` the boundary handling can drop the target index "
        "and return -1 even though the value is present."
    ),
    strict=False,
)
def test_ite_ternary_search_large_sorted_list():
    """Every element of a large sorted list is found at its own index."""
    array = list(range(0, 2000, 2))
    for index, value in enumerate(array):
        assert ite_ternary_search(array, value) == index


@pytest.mark.xfail(
    reason=(
        "Known defect in ite_ternary_search: for long sorted inputs it can "
        "compute an index past the end of the list, raising IndexError "
        "instead of returning -1."
    ),
    strict=False,
)
def test_ite_ternary_search_large_list_missing_target():
    """A target absent from a large sorted list returns -1."""
    array = list(range(0, 2000, 2))
    assert ite_ternary_search(array, 1999) == -1
    assert ite_ternary_search(array, -1) == -1


def test_ite_ternary_search_duplicates_returns_valid_index():
    """With duplicates, any index holding the target is acceptable."""
    array = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = ite_ternary_search(array, 1)
    assert result != -1
    assert array[result] == 1


def test_ite_ternary_search_does_not_mutate_array():
    """ite_ternary_search must not modify the input list."""
    array = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    original = list(array)
    ite_ternary_search(array, 13)
    assert array == original


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
    """rec_ternary_search must reproduce all documented doctest examples."""
    assert rec_ternary_search(left, right, array, target) == expected


def test_rec_ternary_search_empty_bounds_returns_minus_one():
    """When left >= right the function returns -1 immediately."""
    assert rec_ternary_search(0, 0, [1, 2, 3], 1) == -1
    assert rec_ternary_search(3, 1, [1, 2, 3], 1) == -1


def test_rec_ternary_search_single_element_not_found():
    """A one-element list without the target returns -1."""
    assert rec_ternary_search(0, 1, [5], 4) == -1


@pytest.mark.xfail(
    reason=(
        "Known defect in rec_ternary_search: the boundary handling of the "
        "recursive split can drop the target index and return -1 even though "
        "the value is present."
    ),
    strict=False,
)
def test_rec_ternary_search_large_sorted_list():
    """Every element of a large sorted list is found at its own index."""
    array = list(range(0, 2000, 2))
    for index, value in enumerate(array):
        assert rec_ternary_search(0, len(array), array, value) == index


@pytest.mark.xfail(
    reason=(
        "Known defect in rec_ternary_search: for long sorted inputs the "
        "recursive split can compute an index past the end of the list, "
        "raising IndexError instead of returning -1."
    ),
    strict=False,
)
def test_rec_ternary_search_large_list_missing_target():
    """A target absent from a large sorted list returns -1."""
    array = list(range(0, 2000, 2))
    assert rec_ternary_search(0, len(array), array, 1999) == -1


def test_rec_ternary_search_does_not_mutate_array():
    """rec_ternary_search must not modify the input list."""
    array = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    original = list(array)
    rec_ternary_search(0, len(array), array, 42)
    assert array == original


# --------------------------------------------------------------------------- #
# Cross-checks and module-level behaviour
# --------------------------------------------------------------------------- #
def test_precision_is_positive_integer():
    """The module-level precision constant must be a positive int."""
    assert isinstance(precision, int)
    assert precision > 0


@pytest.mark.parametrize(
    ("array", "target"),
    [
        ([0, 1, 2, 8, 13, 17, 19, 32, 42], 13),
        ([0, 1, 2, 8, 13, 17, 19, 32, 42], 3),
        ([4, 5, 6, 7], 4),
        ([4, 5, 6, 7], -10),
        ([-18, 2], -18),
        ([5], 5),
        ([], 1),
    ],
)
def test_iterative_and_recursive_agree(array, target):
    """Both implementations must return the same index for the same input."""
    iterative = ite_ternary_search(array, target)
    recursive = rec_ternary_search(0, len(array), array, target)
    assert iterative == recursive


def test_both_implementations_handle_unsorted_input_without_crashing():
    """Unsorted input is out of contract but must not raise an exception."""
    array = [5, 1, 9, 3]
    assert ite_ternary_search(array, 9) in (-1, 2)
    assert rec_ternary_search(0, len(array), array, 9) in (-1, 2)
