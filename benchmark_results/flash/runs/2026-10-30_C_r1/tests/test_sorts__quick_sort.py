"""Tests for the pure Python quick sort implementation.

The module under test is ``sorts/quick_sort.py`` which exposes a single
public function ``quick_sort`` plus a ``__main__`` interactive block.
"""

from __future__ import annotations

import builtins
import random
import runpy
import sys
from unittest.mock import patch

import pytest

try:  # pragma: no cover - depends on how the project root is laid out
    from sorts.quick_sort import quick_sort
except ImportError:  # pragma: no cover
    from quick_sort import quick_sort


# The module object that actually defines ``quick_sort``.  Patching attributes on
# this object avoids relying on a particular dotted import path being resolvable
# by ``unittest.mock`` (which re-imports the target by name).
QUICK_SORT_MODULE = sys.modules[quick_sort.__module__]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of ints is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_does_not_mutate_input() -> None:
    """The input list must not be mutated by the sort (pivot is popped from a copy)."""
    original = [4, 1, 3, 2]
    snapshot = list(original)
    result = quick_sort(list(original))
    assert result == [1, 2, 3, 4]
    assert original == snapshot


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats() -> None:
    """Floating point values are sorted in ascending order."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings() -> None:
    """Strings are comparable and therefore sortable by the same routine."""
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7, 7], id="two-equal-elements"),
        pytest.param([1, 1, 1, 1, 1], id="all-equal"),
    ],
)
def test_quick_sort_trivial_inputs(collection: list) -> None:
    """Empty, single-element and all-equal inputs are returned unchanged."""
    assert quick_sort(list(collection)) == collection


def test_quick_sort_already_sorted() -> None:
    """An already sorted list stays sorted."""
    data = list(range(10))
    assert quick_sort(list(data)) == data


def test_quick_sort_reverse_sorted() -> None:
    """A reverse sorted list is fully reversed into ascending order."""
    data = list(range(10, 0, -1))
    assert quick_sort(list(data)) == list(range(1, 11))


def test_quick_sort_duplicates_preserved() -> None:
    """Duplicate values are all retained (multiset is preserved)."""
    data = [3, 1, 3, 2, 1, 3]
    result = quick_sort(list(data))
    assert result == [1, 1, 2, 3, 3, 3]
    assert sorted(result) == sorted(data)


def test_quick_sort_large_random_list_matches_builtin_sorted() -> None:
    """A larger pseudo-random list produces the same result as ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]
    assert quick_sort(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Determinism / pivot selection behaviour
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """``randrange`` is used to pick the pivot index; patching it must be honoured."""
    with patch.object(QUICK_SORT_MODULE, "randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])
    assert result == [1, 2, 3]
    assert mock_randrange.called


def test_quick_sort_result_independent_of_pivot_choice() -> None:
    """Whatever pivot index is chosen, the sorted output is identical."""
    data = [5, 3, 8, 1, 9, 2]
    expected = sorted(data)
    for pivot_index in range(len(data)):
        # ``side_effect`` keeps the chosen index inside the current collection
        # (recursive calls receive shorter collections).
        with patch.object(
            QUICK_SORT_MODULE,
            "randrange",
            side_effect=lambda n, _i=pivot_index: _i % n,
        ):
            assert quick_sort(list(data)) == expected


def test_quick_sort_randrange_called_with_collection_length() -> None:
    """``randrange`` receives the current collection length as its argument."""
    with patch.object(QUICK_SORT_MODULE, "randrange", return_value=0) as mock_randrange:
        quick_sort([4, 2, 6])
    assert mock_randrange.call_args_list[0].args == (3,)


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------


def test_quick_sort_raises_on_non_comparable_items() -> None:
    """Mixing incomparable types (int vs str) raises a TypeError."""
    with pytest.raises(TypeError):
        quick_sort([1, "a", 2])


def test_quick_sort_raises_on_non_list_input() -> None:
    """A tuple has no ``pop`` method, so passing one raises AttributeError."""
    with pytest.raises(AttributeError):
        quick_sort((3, 1, 2))


# ---------------------------------------------------------------------------
# ``__main__`` interactive block
# ---------------------------------------------------------------------------


def test_main_block_sorts_user_input(capsys: pytest.CaptureFixture[str]) -> None:
    """Running the module as a script reads comma separated ints and prints the sorted list."""
    with patch.object(builtins, "input", return_value=" 5, 3, 1, 4 \n"):
        runpy.run_path(str(QUICK_SORT_MODULE.__file__), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.strip() == "[1, 3, 4, 5]"


def test_main_block_handles_single_value(capsys: pytest.CaptureFixture[str]) -> None:
    """The script block works for a single input value."""
    with patch.object(builtins, "input", return_value="42"):
        runpy.run_path(str(QUICK_SORT_MODULE.__file__), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.strip() == "[42]"


def test_main_block_raises_on_non_numeric_input() -> None:
    """Non-numeric user input causes a ValueError from the int conversion."""
    with patch.object(builtins, "input", return_value="a,b"):
        with pytest.raises(ValueError):
            runpy.run_path(str(QUICK_SORT_MODULE.__file__), run_name="__main__")
