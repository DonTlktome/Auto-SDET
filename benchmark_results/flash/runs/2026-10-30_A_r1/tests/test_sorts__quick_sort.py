"""Tests for the pure Python quick sort implementation.

The module under test is ``sorts/quick_sort.py`` which exposes a single
public function ``quick_sort`` plus a ``__main__`` demo block.

Implementation note: ``quick_sort`` pops the pivot out of the collection it
is handed, so the caller's list is mutated (it loses one element per call).
Expected values must therefore always be computed *before* calling it.
"""

from __future__ import annotations

import builtins
import random
import sys
from unittest.mock import patch

import pytest

from sorts.quick_sort import quick_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module():
    """Return the already imported module object that defines ``quick_sort``."""
    return sys.modules[quick_sort.__module__]


def _read_source() -> str:
    """Return the source text of the module under test."""
    with open(_module().__file__, "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list() -> None:
    """A typical unsorted list of integers is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_mutates_input_by_removing_the_pivot() -> None:
    """The pivot is popped from the input list, and a new list is returned."""
    original = [4, 1, 3, 2]
    expected = sorted(original)

    result = quick_sort(original)

    assert result == expected == [1, 2, 3, 4]
    # Exactly one element (the pivot) is removed from the caller's list.
    assert len(original) == len(expected) - 1
    # The returned list is the newly built result, not the caller's list.
    assert result is not original


def test_quick_sort_handles_negative_numbers() -> None:
    """Negative values are ordered correctly relative to positives and zero."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats_and_strings() -> None:
    """Any comparable items (floats, strings) are sorted ascending."""
    assert quick_sort([2.5, 1.1, 3.9, 0.0]) == [0.0, 1.1, 2.5, 3.9]
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


# ---------------------------------------------------------------------------
# Edge / boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([], id="empty"),
        pytest.param([42], id="single-element"),
        pytest.param([7], id="single-element-other"),
    ],
)
def test_quick_sort_base_cases_return_input_unchanged(collection: list) -> None:
    """Collections with fewer than two elements are returned as-is."""
    result = quick_sort(collection)

    assert result == collection
    # Base case returns the very same object without touching the RNG.
    assert result is collection


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param([1, 1, 1, 1], id="all-equal"),
        pytest.param([5, 5, 4, 4, 3, 3], id="duplicate-pairs"),
        pytest.param([0, 0], id="two-equal"),
    ],
)
def test_quick_sort_handles_duplicates(collection: list) -> None:
    """Duplicate values are preserved and grouped together."""
    # Compute the expectation first: quick_sort mutates the input list.
    expected = sorted(collection)

    assert quick_sort(collection) == expected


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param(list(range(10)), id="already-sorted"),
        pytest.param(list(range(10, 0, -1)), id="reverse-sorted"),
        pytest.param([3, 1, 2], id="three-elements"),
        pytest.param([2, 1], id="two-elements"),
    ],
)
def test_quick_sort_boundary_orders(collection: list) -> None:
    """Already sorted, reverse sorted and tiny inputs all sort correctly."""
    expected = sorted(collection)

    assert quick_sort(collection) == expected


def test_quick_sort_large_random_input_matches_builtin_sorted() -> None:
    """A larger pseudo-random input produces the same result as ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(300)]
    expected = sorted(data)

    assert quick_sort(data) == expected


# ---------------------------------------------------------------------------
# Pivot selection / RNG interaction
# ---------------------------------------------------------------------------


def test_quick_sort_uses_randrange_for_pivot_selection() -> None:
    """The pivot index is chosen via the module-level ``randrange``."""
    module = _module()

    with patch.object(module, "randrange", return_value=0) as mock_randrange:
        result = quick_sort([3, 1, 2])

    assert result == [1, 2, 3]
    assert mock_randrange.called
    # First call is made with the length of the original collection.
    assert mock_randrange.call_args_list[0].args == (3,)


def test_quick_sort_works_with_deterministic_pivot() -> None:
    """Forcing the pivot to the last index still yields a sorted list."""
    module = _module()

    with patch.object(module, "randrange", side_effect=lambda n: n - 1):
        assert quick_sort([9, 4, 7, 1, 4]) == [1, 4, 4, 7, 9]


def test_quick_sort_randrange_receives_current_length_each_recursion() -> None:
    """``randrange`` is called with the shrinking sub-list lengths."""
    lengths: list[int] = []

    def fake_randrange(n: int) -> int:
        lengths.append(n)
        return 0

    module = _module()

    with patch.object(module, "randrange", side_effect=fake_randrange):
        quick_sort([3, 2, 1])

    # Every recorded length must be a positive int within the sub-list size.
    assert lengths
    assert all(isinstance(n, int) and n >= 1 for n in lengths)
    assert lengths[0] == 3


# ---------------------------------------------------------------------------
# ``__main__`` demo block behaviour
# ---------------------------------------------------------------------------


def test_main_block_parses_input_and_prints_sorted_list(capsys) -> None:
    """Running the module as a script reads CSV ints and prints the sorted list."""
    module_globals = {"__name__": "__main__", "__builtins__": builtins}

    try:
        with patch.object(builtins, "input", return_value=" 3,1,2 \n"):
            exec(compile(_read_source(), "quick_sort.py", "exec"), module_globals)
    except SystemExit:  # pragma: no cover - the demo block has no sys.exit
        pass

    captured = capsys.readouterr()
    assert captured.out.strip() == "[1, 2, 3]"


def test_main_block_handles_single_value(capsys) -> None:
    """A single comma-free value is parsed and echoed back as a one-item list."""
    module_globals = {"__name__": "__main__", "__builtins__": builtins}

    try:
        with patch.object(builtins, "input", return_value="42"):
            exec(compile(_read_source(), "quick_sort.py", "exec"), module_globals)
    except SystemExit:  # pragma: no cover - the demo block has no sys.exit
        pass

    captured = capsys.readouterr()
    assert captured.out.strip() == "[42]"
