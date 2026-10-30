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

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by absolute path (no package assumptions).
# ---------------------------------------------------------------------------
_SOURCE_PATH = Path(__file__).resolve().parent.parent / "sorts" / "insertion_sort.py"
if not _SOURCE_PATH.exists():  # pragma: no cover - fallback for other layouts
    _candidates = [
        Path(__file__).resolve().parent / "insertion_sort.py",
        Path.cwd() / "sorts" / "insertion_sort.py",
        Path.cwd() / "insertion_sort.py",
    ]
    for _candidate in _candidates:
        if _candidate.exists():
            _SOURCE_PATH = _candidate
            break

_spec = importlib.util.spec_from_file_location("insertion_sort_under_test", _SOURCE_PATH)
assert _spec is not None and _spec.loader is not None
insertion_sort_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("insertion_sort_under_test", insertion_sort_module)
_spec.loader.exec_module(insertion_sort_module)

insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------
def test_sorts_unsorted_list_of_ints():
    """A plain unsorted list of integers is sorted in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_sorts_list_of_strings():
    """Strings are sorted according to their natural ordering."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_negative_numbers():
    """Negative integers are handled correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([3.5, -1.25, 0.0, 2.0]) == [-1.25, 0.0, 2.0, 3.5]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same object."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_matches_builtin_sorted_on_random_data():
    """Random integer data produces the same result as the builtin sorted()."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(list(collection)) == sorted(collection)


def test_matches_builtin_sorted_on_random_strings():
    """Random alphanumeric data produces the same result as the builtin sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(list(collection)) == sorted(collection)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7, 7], id="two-equal"),
        pytest.param([1, 2, 3, 4, 5], id="already-sorted"),
        pytest.param([5, 4, 3, 2, 1], id="reverse-sorted"),
        pytest.param([2, 2, 2, 2], id="all-equal"),
        pytest.param([0, -0, 0], id="zeros"),
        pytest.param([1, 3, 2, 3, 1], id="duplicates"),
    ],
)
def test_edge_case_collections(collection):
    """Edge-case collections are sorted exactly like the builtin sorted()."""
    expected = sorted(collection)
    assert insertion_sort(list(collection)) == expected


def test_empty_list_returns_empty_list():
    """An empty list is returned unchanged (and is the same object)."""
    collection: list[int] = []
    result = insertion_sort(collection)
    assert result == []
    assert result is collection


def test_single_element_list_unchanged():
    """A one-element list is returned unchanged."""
    assert insertion_sort([99]) == [99]


def test_large_collection_is_sorted():
    """A large collection is sorted correctly (exercises the inner while loop)."""
    rng = random.Random(2024)
    collection = [rng.randint(-1000, 1000) for _ in range(500)]
    assert insertion_sort(list(collection)) == sorted(collection)


def test_works_with_mutable_sequence_subclass():
    """Any MutableSequence implementation (here: a list subclass) is supported."""

    class MyList(list):
        pass

    collection = MyList([4, 1, 3, 2])
    result = insertion_sort(collection)
    assert result is collection
    assert list(result) == [1, 2, 3, 4]


def test_works_with_custom_comparable_objects():
    """Objects implementing only ``__lt__`` are sorted correctly."""

    class Item:
        def __init__(self, value: int) -> None:
            self.value = value

        def __lt__(self, other: "Item", /) -> bool:
            return self.value < other.value

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"Item({self.value})"

    items = [Item(3), Item(1), Item(2)]
    insertion_sort(items)
    assert [item.value for item in items] == [1, 2, 3]


def test_does_not_mutate_input_when_already_sorted():
    """Sorting an already sorted list leaves its contents untouched."""
    collection = [1, 2, 3, 4, 5]
    insertion_sort(collection)
    assert collection == [1, 2, 3, 4, 5]


def test_raises_type_error_for_non_mutable_sequence():
    """Passing an immutable sequence (tuple) raises a TypeError."""
    with pytest.raises(TypeError):
        insertion_sort((3, 1, 2))


def test_raises_type_error_for_incomparable_items():
    """Mixing incomparable types raises a TypeError from the comparison."""
    with pytest.raises(TypeError):
        insertion_sort([1, "a", 2])


# ---------------------------------------------------------------------------
# Isolation / mocking tests
# ---------------------------------------------------------------------------
def test_comparisons_are_performed_via_lt_operator():
    """The algorithm relies on ``__lt__``; verify it is actually invoked."""
    calls: list[tuple[int, int]] = []

    class Tracked:
        def __init__(self, value: int) -> None:
            self.value = value

        def __lt__(self, other: "Tracked", /) -> bool:
            calls.append((self.value, other.value))
            return self.value < other.value

    items = [Tracked(2), Tracked(1)]
    insertion_sort(items)
    assert [item.value for item in items] == [1, 2]
    assert calls, "expected at least one comparison to be performed"


def test_uses_len_and_getitem_of_collection():
    """The algorithm only needs ``len``, ``__getitem__`` and ``__setitem__``.

    Built-in ``list`` objects do not allow their dunder methods to be patched
    (their ``__len__`` slot is read-only), so a ``list`` subclass that records
    the calls is used as the spy instead.
    """
    calls = {"len": 0, "getitem": 0, "setitem": 0}

    class SpyList(list):
        def __len__(self):
            calls["len"] += 1
            return super().__len__()

        def __getitem__(self, index):
            calls["getitem"] += 1
            return super().__getitem__(index)

        def __setitem__(self, index, value):
            calls["setitem"] += 1
            return super().__setitem__(index, value)

    collection = SpyList([3, 1, 2])
    insertion_sort(collection)

    assert calls["len"] > 0
    assert calls["getitem"] > 0
    assert collection == [1, 2, 3]


def test_doctest_examples_pass():
    """The doctests embedded in the module all pass."""
    import doctest

    results = doctest.testmod(insertion_sort_module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
