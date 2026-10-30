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


# ---------------------------------------------------------------------------
# Locate and import the module under test by absolute path.
# ---------------------------------------------------------------------------
def _find_source_path() -> Path:
    """Search the test file's directory and all of its ancestors for the module.

    The test file may live directly in the project root (next to ``sorts/``)
    or inside a sub-package, so every plausible location is probed instead of
    relying on a single hard-coded relative path.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for base in (here.parent, *here.parents):
        candidates.append(base / "sorts" / "insertion_sort.py")
        candidates.append(base / "insertion_sort.py")
    # Last-resort fallback for the original authoring machine.
    candidates.append(
        Path(r"D:\Documents\Study\Projects\algorithms-bench\sorts\insertion_sort.py")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate insertion_sort.py; searched: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


_SOURCE_PATH = _find_source_path()


def _load_module():
    """Load ``insertion_sort.py`` as a module object without importing a package."""
    spec = importlib.util.spec_from_file_location("insertion_sort_under_test", _SOURCE_PATH)
    assert spec is not None and spec.loader is not None
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
    """A typical unsorted list of integers is returned in ascending order."""
    assert insertion_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_sorts_strings():
    """Strings are sorted lexicographically."""
    assert insertion_sort(["d", "a", "b", "e", "c"]) == ["a", "b", "c", "d", "e"]


def test_sorts_floats():
    """Floating point values are sorted correctly."""
    assert insertion_sort([3.5, -1.25, 0.0, 2.0]) == [-1.25, 0.0, 2.0, 3.5]


def test_sorts_tuples():
    """Tuples (comparable, immutable elements) are sorted correctly."""
    data = [(2, "b"), (1, "z"), (2, "a")]
    assert insertion_sort(data) == [(1, "z"), (2, "a"), (2, "b")]


def test_returns_same_object_in_place():
    """The function sorts in place and returns the very same object."""
    data = [4, 1, 3, 2]
    result = insertion_sort(data)
    assert result is data
    assert data == [1, 2, 3, 4]


def test_works_with_mutable_sequence_subclass():
    """Any MutableSequence implementation (e.g. a list subclass) is supported."""

    class MyList(list):
        pass

    data = MyList([3, 1, 2])
    result = insertion_sort(data)
    assert isinstance(result, MyList)
    assert list(result) == [1, 2, 3]


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
        pytest.param([0, -0.0, 0], id="zeros"),
        pytest.param([1, -1, 0], id="mixed-sign"),
    ],
)
def test_edge_case_collections(collection):
    """Edge-case collections are sorted exactly like Python's built-in ``sorted``."""
    expected = sorted(collection)
    assert insertion_sort(list(collection)) == expected


def test_empty_list_is_returned_unchanged():
    """An empty list is returned as the same empty list object."""
    data = []
    result = insertion_sort(data)
    assert result is data
    assert result == []


def test_single_element_list_unchanged():
    """A one-element list is returned untouched."""
    data = [99]
    assert insertion_sort(data) == [99]


def test_duplicates_preserved():
    """Duplicate values are all preserved (multiset of elements is unchanged)."""
    data = [3, 1, 3, 2, 1, 3]
    result = insertion_sort(data)
    assert result == [1, 1, 2, 3, 3, 3]
    assert sorted(result) == sorted(data)


def test_large_random_integers_matches_sorted():
    """A large random integer collection matches ``sorted`` output."""
    rng = random.Random(1234)
    collection = rng.sample(range(-500, 500), 200)
    assert insertion_sort(collection) == sorted(collection)


def test_large_random_strings_matches_sorted():
    """A large random string collection matches ``sorted`` output."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=200)
    assert insertion_sort(collection) == sorted(collection)


def test_stability_of_equal_keys():
    """Equal keys keep their relative order (insertion sort is stable)."""
    data = [(1, "a"), (0, "x"), (1, "b"), (0, "y")]
    result = insertion_sort(data)
    assert result == [(0, "x"), (0, "y"), (1, "a"), (1, "b")]


def test_does_not_mutate_beyond_sorting():
    """The sorted result contains exactly the original elements."""
    data = [5, 3, 8, 1]
    original = list(data)
    insertion_sort(data)
    assert sorted(data) == sorted(original)
    assert len(data) == len(original)


# ---------------------------------------------------------------------------
# Interaction with external dependencies (mocked)
# ---------------------------------------------------------------------------
def test_uses_only_lt_comparisons():
    """The algorithm relies solely on ``__lt__`` for comparisons."""

    class Tracked:
        def __init__(self, value):
            self.value = value
            self.lt_calls = 0

        def __lt__(self, other):
            self.lt_calls += 1
            return self.value < other.value

        def __repr__(self):  # pragma: no cover - debugging helper
            return f"Tracked({self.value})"

    items = [Tracked(v) for v in (3, 1, 2)]
    insertion_sort(items)
    assert [item.value for item in items] == [1, 2, 3]
    assert sum(item.lt_calls for item in items) > 0


def test_raises_type_error_for_incomparable_items():
    """Comparing incomparable types propagates a ``TypeError``."""
    with pytest.raises(TypeError):
        insertion_sort([1, "a"])


def test_raises_type_error_for_immutable_sequence():
    """Passing an immutable sequence (tuple) fails when assignment is attempted."""
    with pytest.raises(TypeError):
        insertion_sort((3, 1, 2))


def test_raises_type_error_for_none():
    """Passing ``None`` raises ``TypeError`` because it has no length."""
    with pytest.raises(TypeError):
        insertion_sort(None)


def test_raises_index_error_for_non_sequence():
    """Passing a non-subscriptable object raises ``TypeError``."""
    with pytest.raises(TypeError):
        insertion_sort(42)


# ---------------------------------------------------------------------------
# Mocked I/O for the ``__main__`` block
# ---------------------------------------------------------------------------
def test_main_block_reads_input_and_prints_sorted(monkeypatch, capsys):
    """Running the module as a script reads comma separated numbers and prints them sorted."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "3,1,2")
    runpy.run_path(str(_SOURCE_PATH), run_name="__main__")
    captured = capsys.readouterr()
    assert "insertion_sort(unsorted) = [1, 2, 3]" in captured.out


def test_main_block_handles_whitespace_input(monkeypatch, capsys):
    """Whitespace around the input is stripped before parsing."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "  5, -1 , 0  ")
    runpy.run_path(str(_SOURCE_PATH), run_name="__main__")
    captured = capsys.readouterr()
    assert "[-1, 0, 5]" in captured.out


def test_main_block_raises_on_invalid_input(monkeypatch):
    """Non-numeric input raises ``ValueError`` from ``int`` conversion."""
    import runpy

    monkeypatch.setattr("builtins.input", lambda *a, **k: "a,b")
    with pytest.raises(ValueError):
        runpy.run_path(str(_SOURCE_PATH), run_name="__main__")


# ---------------------------------------------------------------------------
# Doctests
# ---------------------------------------------------------------------------
def test_doctests_pass():
    """All doctests embedded in the module pass."""
    import doctest

    results = doctest.testmod(insertion_sort_module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
