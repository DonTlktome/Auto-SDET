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
from unittest.mock import patch

import pytest

MODULE_NAME = "insertion_sort.py"
_HERE = Path(__file__).resolve().parent


def _find_source() -> Path:
    """Locate ``insertion_sort.py`` regardless of where the repo root sits."""
    roots = [_HERE, _HERE.parent, _HERE.parent.parent, Path.cwd()]
    candidates = []
    for root in roots:
        candidates.append(root / "sorts" / MODULE_NAME)
        candidates.append(root / MODULE_NAME)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - defensive
            continue
    searched = "\n".join(str(c) for c in candidates)
    raise ImportError(f"Cannot locate {MODULE_NAME}; searched:\n{searched}")


SOURCE_PATH = _find_source()


def _load_module():
    """Load the insertion_sort module from its absolute file path."""
    spec = importlib.util.spec_from_file_location("insertion_sort_under_test", SOURCE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module from {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


insertion_sort_module = _load_module()
insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_sorts_unsorted_integers():
    """A typical unsorted list of integers is sorted in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_sorts_already_sorted_list():
    """An already sorted list is returned unchanged."""
    data = [1, 2, 3, 4, 5]
    assert insertion_sort(data) == [1, 2, 3, 4, 5]


def test_sorts_reverse_sorted_list():
    """A reverse sorted list is fully reversed into ascending order."""
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_sorts_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([2.5, -1.0, 0.0, 3.75]) == [-1.0, 0.0, 2.5, 3.75]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same object."""
    data = [3, 1, 2]
    result = insertion_sort(data)
    assert result is data
    assert data == [1, 2, 3]


def test_works_with_mutable_sequence_subclass():
    """Any MutableSequence implementation (e.g. a list subclass) is supported."""
    class MyList(list):
        pass

    data = MyList([4, 2, 9, 1])
    result = insertion_sort(data)
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
        pytest.param([0, 0, 0, 0], id="all-equal"),
    ],
)
def test_trivial_collections(collection):
    """Empty, single-element and all-equal collections are returned unchanged."""
    original = list(collection)
    assert insertion_sort(collection) == original


def test_negative_numbers():
    """Negative numbers are ordered correctly."""
    assert insertion_sort([-2, -5, -45]) == [-45, -5, -2]


def test_mixed_sign_numbers():
    """A mix of negative, zero and positive values is sorted correctly."""
    assert insertion_sort([3, -1, 0, -10, 8]) == [-10, -1, 0, 3, 8]


def test_duplicates_preserved():
    """Duplicate values are all preserved (multiset is unchanged)."""
    data = [3, 1, 3, 2, 1, 3]
    assert insertion_sort(data) == [1, 1, 2, 3, 3, 3]


def test_large_random_integers_matches_sorted():
    """A large random integer collection matches Python's built-in sorted()."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_large_random_strings_matches_sorted():
    """A large random string collection matches Python's built-in sorted()."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


def test_stability_of_equal_keys():
    """Equal-comparing items keep their relative order (stable sort)."""
    data = [(1, "a"), (0, "b"), (1, "c"), (0, "d")]
    assert insertion_sort(data) == [(0, "b"), (0, "d"), (1, "a"), (1, "c")]


def test_does_not_mutate_beyond_sorting():
    """The sorted result contains exactly the same elements as the input."""
    data = [9, 3, 7, 1, 5]
    original = sorted(data)
    insertion_sort(data)
    assert data == original
    assert len(data) == 5


# ---------------------------------------------------------------------------
# Interaction with external dependencies (mocked)
# ---------------------------------------------------------------------------


def test_uses_only_lt_comparisons():
    """Sorting relies on the ``<`` operator of the elements (Comparable protocol)."""
    class Tracked:
        def __init__(self, value):
            self.value = value
            self.lt_calls = 0

        def __lt__(self, other):
            self.lt_calls += 1
            return self.value < other.value

        def __repr__(self):  # pragma: no cover - debugging helper
            return f"Tracked({self.value})"

    items = [Tracked(3), Tracked(1), Tracked(2)]
    insertion_sort(items)
    assert [item.value for item in items] == [1, 2, 3]
    assert sum(item.lt_calls for item in items) > 0


def test_main_block_reads_input_and_prints_result():
    """Running the module as a script reads comma separated numbers and prints the result."""
    import runpy

    with patch("builtins.input", return_value="3,1,2") as mock_input, patch(
        "builtins.print"
    ) as mock_print:
        runpy.run_path(str(SOURCE_PATH), run_name="__main__")

    mock_input.assert_called_once()
    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
    assert "[1, 2, 3]" in printed
