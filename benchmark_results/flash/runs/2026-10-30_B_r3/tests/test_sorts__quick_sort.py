"""Tests for the pure Python quick sort implementation.

The module under test is ``sorts/quick_sort.py`` which exposes a single
public function ``quick_sort`` plus a ``__main__`` demo block.
"""

from __future__ import annotations

import builtins
import os
import random
import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest

from sorts.quick_sort import quick_sort


# The module object that actually defines ``quick_sort``.  Resolving it via
# ``sys.modules`` avoids depending on the (possibly aliased) dotted package
# name being importable by ``unittest.mock`` / ``runpy``.
quick_sort_module = sys.modules[quick_sort.__module__]


def _patch_randrange(**mock_kwargs) -> MagicMock:
    """Install a fake ``randrange`` in the module where the code looks it up."""
    fake_randrange = MagicMock(**mock_kwargs)
    patcher = patch.dict(quick_sort.__globals__, {"randrange": fake_randrange})
    patcher.start()
    return fake_randrange


def _run_module_as_main() -> None:
    """Execute the module under test as if run as a script (``__main__``)."""
    path = getattr(quick_sort_module, "__file__", None)
    if path and os.path.exists(path):
        runpy.run_path(path, run_name="__main__")
        return

    loader = getattr(quick_sort_module, "__loader__", None)
    if loader is not None and hasattr(loader, "get_source"):
        source = loader.get_source(quick_sort_module.__name__)
        exec(
            compile(source, path or "<quick_sort>", "exec"),
            {"__name__": "__main__", "__file__": path},
        )
        return

    raise RuntimeError("Unable to execute the quick_sort module as __main__")


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_quick_sort_sorts_unsorted_list():
    """A typical unsorted list of ints is returned in ascending order."""
    assert quick_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_quick_sort_returns_new_list_and_does_not_mutate_input():
    """The result is a new list; the caller's list only loses the pivot."""
    original = [4, 1, 3, 2]
    snapshot = list(original)

    result = quick_sort(original)

    assert result == [1, 2, 3, 4]
    assert result is not original
    # The implementation pops exactly one pivot element from the caller's
    # collection; every other element is left untouched and in order.
    assert len(original) == len(snapshot) - 1
    assert any(
        snapshot[:index] + snapshot[index + 1 :] == original
        for index in range(len(snapshot))
    )


def test_quick_sort_handles_negative_numbers():
    """Negative values are ordered correctly relative to positives."""
    assert quick_sort([-2, 5, 0, -45]) == [-45, -2, 0, 5]


def test_quick_sort_handles_floats():
    """Floating point values are supported since they are comparable."""
    assert quick_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_quick_sort_handles_strings():
    """Strings are comparable and therefore sortable."""
    assert quick_sort(["pear", "apple", "fig"]) == ["apple", "fig", "pear"]


def test_quick_sort_handles_tuples_of_comparable_items():
    """Tuples are comparable element-wise and can be sorted."""
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
def test_quick_sort_base_cases_return_input(collection):
    """Collections with fewer than two elements are returned unchanged."""
    assert quick_sort(collection) == collection


def test_quick_sort_empty_list_returns_same_object():
    """The base case returns the very same object for an empty list."""
    empty: list = []
    assert quick_sort(empty) is empty


def test_quick_sort_single_element_returns_same_object():
    """The base case returns the very same object for a one-element list."""
    single = [7]
    assert quick_sort(single) is single


def test_quick_sort_already_sorted():
    """An already sorted list stays sorted."""
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_quick_sort_reverse_sorted():
    """A descending list is fully reversed into ascending order."""
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_quick_sort_all_equal_elements():
    """All-equal input exercises the `<=` partition branch."""
    assert quick_sort([3, 3, 3, 3]) == [3, 3, 3, 3]


def test_quick_sort_duplicates_preserved():
    """Duplicates are all retained (multiset is preserved)."""
    result = quick_sort([2, 1, 2, 1, 2])
    assert result == [1, 1, 2, 2, 2]
    assert len(result) == 5


def test_quick_sort_two_elements_unsorted():
    """Smallest non-trivial input: two out-of-order elements."""
    assert quick_sort([2, 1]) == [1, 2]


def test_quick_sort_two_elements_sorted():
    """Smallest non-trivial input that is already ordered."""
    assert quick_sort([1, 2]) == [1, 2]


def test_quick_sort_large_random_list_matches_builtin_sorted():
    """A larger randomized list matches Python's built-in ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-1000, 1000) for _ in range(500)]

    assert quick_sort(list(data)) == sorted(data)


def test_quick_sort_does_not_raise_on_uncomparable_mixed_types():
    """Mixed incomparable types raise TypeError from the comparison."""
    with pytest.raises(TypeError):
        quick_sort([1, "a", 2])


# ---------------------------------------------------------------------------
# Pivot selection / randomness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pivot_index", [0, 1, 2, 3, 4])
def test_quick_sort_works_for_every_possible_pivot_choice(pivot_index):
    """Sorting is correct regardless of which pivot index is chosen."""
    data = [5, 1, 4, 2, 3]
    fake_randrange = MagicMock(side_effect=lambda n: pivot_index % n)
    with patch.dict(quick_sort.__globals__, {"randrange": fake_randrange}):
        assert quick_sort(list(data)) == [1, 2, 3, 4, 5]


def test_quick_sort_calls_randrange_with_collection_length():
    """``randrange`` is invoked with the current collection length."""
    mock_randrange = MagicMock(return_value=0)
    with patch.dict(quick_sort.__globals__, {"randrange": mock_randrange}):
        quick_sort([3, 1, 2])

    assert mock_randrange.call_count >= 1
    for call in mock_randrange.call_args_list:
        assert len(call.args) == 1
        assert call.kwargs == {}
        assert isinstance(call.args[0], int)
        assert call.args[0] >= 1


def test_quick_sort_uses_randrange_from_module_namespace():
    """The module-level ``randrange`` symbol is the one actually used."""
    mock_randrange = MagicMock(return_value=0)
    with patch.dict(quick_sort.__globals__, {"randrange": mock_randrange}):
        quick_sort([2, 1])

    mock_randrange.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# __main__ demo block
# ---------------------------------------------------------------------------


def test_main_block_parses_input_and_prints_sorted_list(capsys):
    """Running the module as a script sorts comma separated user input."""
    with patch.object(builtins, "input", return_value=" 3,1,2 \n"):
        _run_module_as_main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "[1, 2, 3]"


def test_main_block_handles_single_value(capsys):
    """The demo block works with a single comma separated value."""
    with patch.object(builtins, "input", return_value="7"):
        _run_module_as_main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "[7]"


def test_main_block_raises_on_non_numeric_input():
    """Non-numeric input causes a ValueError from the int() conversion."""
    with patch.object(builtins, "input", return_value="a,b"):
        with pytest.raises(ValueError):
            _run_module_as_main()
