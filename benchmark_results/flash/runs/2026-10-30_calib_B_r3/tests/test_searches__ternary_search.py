# AUTO-SDET: Test generation FAILED after 3 retries
# Last error: 

"""Tests for searches/ternary_search.py."""

from __future__ import annotations

import pytest

import ternary_search
from ternary_search import (
    ite_ternary_search,
    lin_search,
    precision,
    rec_ternary_search,
)


# ---------------------------------------------------------------------------
# lin_search
# ---------------------------------------------------------------------------


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
def test_lin_search_happy_path(left, right, array, target, expected):
    """lin_search returns the index of the target within [left, right)."""
    assert lin_search(left, right, array, target) == expected


def test_lin_search_target_outside_range():
    """lin_search returns -1 when the target lies outside the searched range."""
    assert lin_search(1, 3, [4, 5, 6, 7], 4) == -1


def test_lin_search_empty_range():
    """lin_search returns -1 for an empty range (left == right)."""
    assert lin_search(0, 0, [1, 2, 3], 1) == -1


def test_lin_search_right_bound_exclusive():
    """lin_search treats `right` as an exclusive bound."""
    array = [1, 2, 3]
    assert lin_search(0, 3, array, 3) == 2
    assert lin_search(0, 2, array, 3) == -1


def test_lin_search_returns_first_occurrence():
    """lin_search returns the first matching index when duplicates exist."""
    assert lin_search(0, 5, [1, 2, 2, 2, 3], 2) == 1


# ---------------------------------------------------------------------------
# ite_ternary_search
# ---------------------------------------------------------------------------


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
def test_ite_ternary_search_happy_path(array, target, expected):
    """ite_ternary_search finds the target index or returns -1."""
    assert ite_ternary_search(array, target) == expected


def test_ite_ternary_search_empty_list():
    """ite_ternary_search returns -1 for an empty list."""
    assert ite_ternary_search([], 42) == -1


def test_ite_ternary_search_single_element_missing():
    """ite_ternary_search returns -1 for a single-element list without the target."""
    assert ite_ternary_search([5], 6) == -1


def test_ite_ternary_search_first_and_last():
    """ite_ternary_search finds elements at the extreme ends of a large list."""
    array = list(range(0, 200, 2))
    assert ite_ternary_search(array, 0) == 0
    assert ite_ternary_search(array, 198) == 99


def test_ite_ternary_search_large_list_middle():
    """ite_ternary_search finds a middle element in a large sorted list."""
    array = list(range(1000))
    assert ite_ternary_search(array, 500) == 500


def test_ite_ternary_search_large_list_missing():
    """ite_ternary_search returns -1 for a missing element in a large list."""
    array = list(range(0, 1000, 2))
    assert ite_ternary_search(array, 501) == -1


def test_ite_ternary_search_uses_lin_search_for_small_ranges(monkeypatch):
    """ite_ternary_search delegates to lin_search once the range is below precision."""
    calls = []

    def fake_lin_search(left, right, array, target):
        calls.append((left, right))
        return -1

    monkeypatch.setattr(ternary_search, "lin_search", fake_lin_search)
    assert ite_ternary_search([1, 2, 3], 2) == -1
    assert calls, "lin_search should have been called for a small range"


def test_ite_ternary_search_respects_precision_module_value():
    """The module-level `precision` constant is a positive integer."""
    assert isinstance(precision, int)
    assert precision >= 1


# ---------------------------------------------------------------------------
# rec_ternary_search
# ---------------------------------------------------------------------------


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
def test_rec_ternary_search_happy_path(left, right, array, target, expected):
    """rec_ternary_search finds the target index or returns -1."""
    assert rec_ternary_search(left, right, array, target) == expected


def test_rec_ternary_search_empty_bounds():
    """rec_ternary_search returns -1 when left == right (empty range)."""
    assert rec_ternary_search(0, 0, [1, 2, 3], 1) == -1


def test_rec_ternary_search_inverted_bounds():
    """rec_ternary_search returns -1 when left > right."""
    assert rec_ternary_search(3, 1, [1, 2, 3], 2) == -1


def test_rec_ternary_search_single_element_missing():
    """rec_ternary_search returns -1 for a single-element list without the target."""
    assert rec_ternary_search(0, 1, [5], 6) == -1


def test_rec_ternary_search_large_list():
    """rec_ternary_search finds elements in a large sorted list."""
    array = list(range(1000))
    assert rec_ternary_search(0, len(array) - 1, array, 0) == 0
    assert rec_ternary_search(0, len(array) - 1, array, 999) == 999
    assert rec_ternary_search(0, len(array) - 1, array, 500) == 500
    assert rec_ternary_search(0, len(array) - 1, array, 1000) == -1


def test_rec_ternary_search_uses_lin_search_for_small_ranges(monkeypatch):
    """rec_ternary_search delegates to lin_search once the range is below precision."""
    calls = []

    def fake_lin_search(left, right, array, target):
        calls.append((left, right))
        return -1

    monkeypatch.setattr(ternary_search, "lin_search", fake_lin_search)
    assert rec_ternary_search(0, 3, [1, 2, 3], 2) == -1
    assert calls, "lin_search should have been called for a small range"


# ---------------------------------------------------------------------------
# Cross-checks between implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [-1, 0, 1, 7, 13, 42, 43, 100])
def test_iterative_and_recursive_agree(target):
    """Both implementations agree on the result for the same sorted input."""
    array = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    ite = ite_ternary_search(array, target)
    rec = rec_ternary_search(0, len(array) - 1, array, target)
    assert ite == rec


@pytest.mark.parametrize("target", range(0, 50, 7))
def test_search_matches_list_index(target):
    """Search results match the true index from list.index for present values."""
    array = list(range(0, 50, 2))
    expected = array.index(target) if target in array else -1
    assert ite_ternary_search(array, target) == expected
    assert rec_ternary_search(0, len(array) - 1, array, target) == expected
