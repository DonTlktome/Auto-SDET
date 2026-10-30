"""Tests for the pure Python quick sort implementation.

The module under test is ``sorts/quick_sort.py`` which exposes a single
public function ``quick_sort`` plus a ``__main__`` demo block.
"""

from __future__ import annotations

import builtins
import importlib
import random
import runpy
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort


# Resolve the actual module object that defines ``quick_sort``. Patching
# attributes on this object directly (rather than via a dotted-name string)
# keeps the tests independent of how the module happens to be importable in
# the test runner's environment.
_QUICK_SORT_MODULE = sys.modules.get(quick_sort.__module__)
if _QUICK_SORT_MODULE is None:  # pragma: no cover - defensive fallback
    _QUICK_SORT_MODULE = importlib.import_module(quick_sort.__module__)


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of ints is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_sorted_list() -> None:
    """The result is a freshly built, sorted list."""
    source = [3, 1, 2]
    snapshot = list(source)

    result = quick_sort(list(source))

    assert result == [1, 2, 3]
    assert source == snapshot
    assert result is not source


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are supported since they are comparable."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Any comparable items work, e.g. strings sorted lexicographically."""
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


def test_quick_sort_handles_tuples_of_comparable_items() -> None:
    """Tuples compare lexicographically, so they can be sorted too."""
    assert quick_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param(["only"], id="single-string"),
    ],
)
def test_quick_sort_base_cases_return_input_unchanged(collection: list) -> None:
    """Collections with fewer than two elements are returned as-is."""
    assert quick_sort(collection) == collection


def test_quick_sort_empty_list_returns_same_object() -> None:
    """The base case returns the very same object for an empty list."""
    empty: list = []
    assert quick_sort(empty) is empty


def test_quick_sort_already_sorted_list() -> None:
    """An already sorted list stays sorted."""
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_reverse_sorted_list() -> None:
    """A descending list is fully reversed into ascending order."""
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_all_equal_elements() -> None:
    """All-equal input exercises the `item <= pivot` branch for every element."""
    assert quick_sort([7, 7, 7, 7, 7]) == [7, 7, 7, 7, 7]


def test_quick_sort_duplicates_preserved() -> None:
    """Duplicates are all retained (multiset of values is unchanged)."""
    data = [4, 1, 4, 2, 1, 4]
    result = quick_sort(list(data))
    assert result == [1, 1, 2, 4, 4, 4]
    assert sorted(result) == sorted(data)


def test_quick_sort_two_elements_ordered() -> None:
    """Smallest non-trivial input: exactly two elements out of order."""
    assert quick_sort([2, 1]) == [1, 2]


def test_quick_sort_two_elements_already_ordered() -> None:
    """Two elements already in order remain unchanged."""
    assert quick_sort([1, 2]) == [1, 2]


def test_quick_sort_large_random_list_matches_builtin_sorted() -> None:
    """A larger randomized list produces the same result as `sorted`."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Pivot selection / randomness isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pivot_index", [0, 1, 2, 3, 4])
def test_quick_sort_result_independent_of_pivot_choice(pivot_index: int) -> None:
    """Forcing a deterministic pivot still yields a correctly sorted list."""
    data = [5, 3, 8, 1, 9]

    def _clamped_index(stop: int) -> int:
        # Keep the chosen index within the bounds of the current collection.
        return pivot_index % stop

    with patch.object(_QUICK_SORT_MODULE, "randrange", side_effect=_clamped_index):
        assert quick_sort(list(data)) == sorted(data)


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """`randrange` is called with the current collection length."""
    with patch.object(_QUICK_SORT_MODULE, "randrange", return_value=0) as mock_randrange:
        quick_sort([3, 1, 2])

    assert mock_randrange.call_count >= 1
    # The first call must be bounded by the length of the original collection.
    assert mock_randrange.call_args_list[0].args == (3,)


def test_quick_sort_works_when_randrange_always_picks_last_index() -> None:
    """Worst-case-ish pivot choice (always the last element) still sorts correctly."""
    data = list(range(20, 0, -1))

    def _last_index(stop: int) -> int:
        return stop - 1

    with patch.object(_QUICK_SORT_MODULE, "randrange", side_effect=_last_index):
        assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# __main__ demo block
# ---------------------------------------------------------------------------


def test_main_block_parses_input_and_prints_sorted_list(capsys) -> None:
    """Running the module as a script reads comma separated ints and prints them sorted."""
    with patch.object(builtins, "input", return_value=" 3,1,2 \n"):
        runpy.run_path(_QUICK_SORT_MODULE.__file__, run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "[1, 2, 3]"


def test_main_block_handles_single_value(capsys) -> None:
    """The demo block works with a single comma-free value."""
    with patch.object(builtins, "input", return_value="42"):
        runpy.run_path(_QUICK_SORT_MODULE.__file__, run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out.strip() == "[42]"


def test_main_block_raises_on_non_numeric_input() -> None:
    """Non-numeric input makes the int() conversion raise ValueError."""
    with patch.object(builtins, "input", return_value="a,b"):
        with pytest.raises(ValueError):
            runpy.run_path(_QUICK_SORT_MODULE.__file__, run_name="__main__")
