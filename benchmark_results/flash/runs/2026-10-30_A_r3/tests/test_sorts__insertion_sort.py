"""Tests for the insertion_sort module.

These tests exercise the pure-Python insertion sort implementation found in
``sorts/insertion_sort.py``.  The module has no external dependencies, so all
tests run fully offline.
"""

from __future__ import annotations

import random
import string

import pytest

from sorts.insertion_sort import insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_insertion_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is sorted in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_insertion_sort_returns_same_object() -> None:
    """The function sorts in place and returns the very same collection."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_insertion_sort_matches_builtin_sorted_on_random_ints() -> None:
    """Sorting a random sample of integers matches Python's ``sorted``."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_insertion_sort_matches_builtin_sorted_on_random_strings() -> None:
    """Sorting random characters matches Python's ``sorted``."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([1, 1, 1, 1], id="all-equal"),
        pytest.param([1, 2, 3, 4, 5], id="already-sorted"),
        pytest.param([5, 4, 3, 2, 1], id="reverse-sorted"),
        pytest.param([-2, -5, -45], id="all-negative"),
        pytest.param([0, -1, 1, -1, 0], id="mixed-signs-with-duplicates"),
        pytest.param([2, 1], id="two-elements-unsorted"),
        pytest.param([1, 2], id="two-elements-sorted"),
    ],
)
def test_insertion_sort_edge_cases(collection: list[int]) -> None:
    """Various boundary inputs are sorted correctly and match ``sorted``."""
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


def test_insertion_sort_empty_list_returns_empty_list() -> None:
    """An empty collection is returned unchanged (and is the same object)."""
    collection: list[int] = []
    result = insertion_sort(collection)
    assert result == []
    assert result is collection


def test_insertion_sort_single_element_unchanged() -> None:
    """A one-element collection is returned unchanged."""
    collection = [7]
    assert insertion_sort(collection) == [7]


def test_insertion_sort_strings() -> None:
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_insertion_sort_floats() -> None:
    """Floating point values are sorted correctly."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.14]) == [-1.0, 0.0, 2.5, 3.14]


def test_insertion_sort_tuples_by_first_element() -> None:
    """Tuples are compared lexicographically, as with ``sorted``."""
    collection = [(2, "b"), (1, "a"), (2, "a")]
    assert insertion_sort(collection) == sorted(collection)


def test_insertion_sort_works_with_mutable_sequence_subclass() -> None:
    """Any ``MutableSequence`` (e.g. ``bytearray``) is supported."""
    collection = bytearray(b"cab")
    result = insertion_sort(collection)
    assert result is collection
    assert bytes(collection) == b"abc"


def test_insertion_sort_large_reverse_range() -> None:
    """A larger reverse-ordered input is sorted correctly."""
    collection = list(range(200, 0, -1))
    assert insertion_sort(collection) == list(range(1, 201))


def test_insertion_sort_does_not_mutate_other_references_unexpectedly() -> None:
    """Sorting in place affects the caller's list, as documented."""
    original = [4, 2, 9, 1]
    alias = original
    insertion_sort(original)
    assert alias == [1, 2, 4, 9]


def test_insertion_sort_uses_only_less_than_comparisons() -> None:
    """The algorithm relies solely on ``__lt__`` for ordering."""

    class OnlyLt:
        def __init__(self, value: int) -> None:
            self.value = value

        def __lt__(self, other: "OnlyLt", /) -> bool:
            return self.value < other.value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, OnlyLt) and self.value == other.value

        def __repr__(self) -> str:  # pragma: no cover - debugging aid
            return f"OnlyLt({self.value})"

    collection = [OnlyLt(3), OnlyLt(1), OnlyLt(2)]
    result = insertion_sort(collection)
    assert [item.value for item in result] == [1, 2, 3]