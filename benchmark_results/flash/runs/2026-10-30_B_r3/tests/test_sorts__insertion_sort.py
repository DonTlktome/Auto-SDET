"""Tests for the insertion_sort module.

The module under test is imported by file path so that the test suite does not
depend on the package layout of the project.
"""

from __future__ import annotations

import importlib.util
import random
import runpy
import string
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _find_source_path() -> Path:
    """Locate ``insertion_sort.py`` regardless of where the test file lives."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "sorts" / "insertion_sort.py",
        here.parent / "sorts" / "insertion_sort.py",
        here / "insertion_sort.py",
        here.parent / "insertion_sort.py",
        Path.cwd() / "sorts" / "insertion_sort.py",
        Path.cwd() / "insertion_sort.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for root in (here, here.parent, Path.cwd()):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("insertion_sort.py")):
            if path.is_file():
                return path

    raise FileNotFoundError(
        "Could not locate insertion_sort.py relative to {}".format(here)
    )


SOURCE_PATH = _find_source_path()


def _load_module():
    """Load the insertion_sort module from its file path."""
    spec = importlib.util.spec_from_file_location("insertion_sort_under_test", SOURCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _exec_module_as_main(name: str):
    """Execute the module so that its ``if __name__ == "__main__"`` block runs.

    ``runpy.run_path`` is used because a ``SourceFileLoader`` bound to a
    non-``__main__`` module name refuses to execute the code once
    ``module.__name__`` has been changed to ``"__main__"``.
    """
    saved = sys.modules.get(name)
    try:
        namespace = runpy.run_path(str(SOURCE_PATH), run_name="__main__")
        sys.modules[name] = namespace
        return namespace
    finally:
        if saved is None:
            pass
        else:
            sys.modules[name] = saved


insertion_sort_module = _load_module()
insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_sorts_simple_unsorted_list():
    """A plain unsorted list of ints is returned in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same collection object."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_sorts_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_sorts_tuples_inside_list():
    """Tuples are comparable, so a list of tuples is sorted element-wise."""
    assert insertion_sort([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


def test_sorts_mutable_sequence_subclass():
    """Any MutableSequence (e.g. a list subclass) is supported."""
    class MyList(list):
        pass

    collection = MyList([4, 2, 9, 1])
    result = insertion_sort(collection)
    assert isinstance(result, MyList)
    assert list(result) == [1, 2, 4, 9]


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
        pytest.param([-2, -5, -45], id="all-negative"),
        pytest.param([0, -0, 0], id="zeros"),
        pytest.param([3, -1, 0, -1, 3], id="mixed-signs-with-duplicates"),
    ],
)
def test_edge_case_collections(collection):
    """Edge-case collections are sorted exactly like Python's built-in sorted()."""
    expected = sorted(collection)
    assert insertion_sort(collection) == expected


def test_empty_list_is_returned_unchanged():
    """An empty list stays empty and the same object is returned."""
    collection = []
    result = insertion_sort(collection)
    assert result is collection
    assert result == []


def test_single_element_list_unchanged():
    """A one-element list is trivially sorted and untouched."""
    collection = [99]
    assert insertion_sort(collection) == [99]


def test_large_reverse_sorted_list():
    """A large reverse-sorted list is sorted correctly (worst-case input)."""
    collection = list(range(500, 0, -1))
    assert insertion_sort(collection) == list(range(1, 501))


def test_random_integers_match_sorted():
    """Random integers match the result of the built-in sorted()."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_random_ascii_letters_and_digits_match_sorted():
    """Random mixed strings match the result of the built-in sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


def test_does_not_mutate_other_references_unexpectedly():
    """Sorting in place is visible through an alias, as documented."""
    collection = [3, 1, 2]
    alias = collection
    insertion_sort(collection)
    assert alias == [1, 2, 3]


def test_uses_only_less_than_comparisons():
    """The algorithm relies solely on ``<`` between elements.

    A custom comparable object that only implements ``__lt__`` must still be
    sortable, proving no other comparison operators are required.
    """
    class OnlyLt:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

        def __repr__(self):
            return f"OnlyLt({self.value})"

    collection = [OnlyLt(3), OnlyLt(1), OnlyLt(2)]
    insertion_sort(collection)
    assert [item.value for item in collection] == [1, 2, 3]


def test_comparison_count_is_quadratic_worst_case():
    """Reverse-sorted input triggers the documented O(n^2) comparison behaviour."""
    class Counting:
        comparisons = 0

        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            Counting.comparisons += 1
            return self.value < other.value

    n = 10
    collection = [Counting(i) for i in range(n, 0, -1)]
    insertion_sort(collection)
    assert [item.value for item in collection] == list(range(1, n + 1))
    # Worst case for insertion sort is n * (n - 1) / 2 comparisons.
    assert Counting.comparisons == n * (n - 1) // 2


def test_raises_type_error_for_incomparable_items():
    """Mixing incomparable types raises TypeError, mirroring sorted()."""
    with pytest.raises(TypeError):
        insertion_sort([1, "a", 2])


# ---------------------------------------------------------------------------
# Isolation / mocking tests
# ---------------------------------------------------------------------------


def test_sorting_does_not_touch_stdout(capsys):
    """The sorting function itself performs no I/O."""
    insertion_sort([3, 1, 2])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_block_reads_input_and_prints_result():
    """Running the module as a script reads comma-separated numbers and prints the result.

    ``input`` is patched so no real stdin is required, and the module is
    executed with ``__name__ == "__main__"``.
    """
    with patch("builtins.input", return_value="3, 1, 2") as mock_input, patch(
        "builtins.print"
    ) as mock_print:
        _exec_module_as_main("insertion_sort_as_main")

    mock_input.assert_called_once()
    outputs = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
    assert any(
        "insertion_sort(unsorted)" in output and "[1, 2, 3]" in output
        for output in outputs
    )


def test_main_block_handles_whitespace_around_numbers():
    """Whitespace around the comma-separated numbers is stripped before parsing."""
    with patch("builtins.input", return_value="  10 , -2 ,5  "), patch(
        "builtins.print"
    ) as mock_print:
        _exec_module_as_main("insertion_sort_as_main_ws")

    outputs = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
    assert any("[-2, 5, 10]" in output for output in outputs)


def test_doctests_pass():
    """The module's own doctests must all pass."""
    import doctest

    results = doctest.testmod(insertion_sort_module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
