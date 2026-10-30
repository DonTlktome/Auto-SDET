"""Tests for the insertion_sort module.

The module under test is imported by file path so that the test suite does not
depend on the package layout of the project.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import string
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SOURCE_FILENAME = "insertion_sort.py"
_MODULE_NAME = "insertion_sort_under_test"


def _candidate_roots() -> list[Path]:
    """Return every directory that may contain the module under test.

    The test file may sit next to the ``sorts`` directory, inside it, in a
    sibling ``tests`` directory, or in a temporary directory created by the
    test runner, so the ancestors of this file, the current working directory
    and every ``sys.path`` entry are considered.
    """
    candidates: list[Path] = list(Path(__file__).resolve().parents)
    candidates.append(Path.cwd())
    for entry in list(sys.path):
        try:
            candidates.append(Path(entry or ".").resolve())
        except (OSError, TypeError):  # pragma: no cover - defensive
            continue

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _is_searchable(root: Path) -> bool:
    """Only recurse into directories that are cheap and safe to scan."""
    if not root.is_dir():
        return False
    if root == Path(root.anchor):  # filesystem root, e.g. "/" or "C:\\"
        return False
    if root.name in {"site-packages", "dist-packages", "__pycache__"}:
        return False
    try:
        base_prefix = Path(sys.base_prefix).resolve()
    except OSError:  # pragma: no cover - defensive
        return True
    return not (root == base_prefix or base_prefix in root.parents)


def _find_source_path() -> Path:
    """Locate the ``insertion_sort`` source file without a package layout."""
    roots = _candidate_roots()

    # 1. cheap, exact lookups: <root>/sorts/insertion_sort.py then <root>/insertion_sort.py
    for root in roots:
        for candidate in (root / "sorts" / _SOURCE_FILENAME, root / _SOURCE_FILENAME):
            if candidate.is_file():
                return candidate

    # 2. recursive fallback below each plausible root, preferring a "sorts" folder
    fallback: Path | None = None
    for root in roots:
        if not _is_searchable(root):
            continue
        for candidate in sorted(root.rglob(_SOURCE_FILENAME)):
            if not candidate.is_file():
                continue
            if candidate.parent.name == "sorts":
                return candidate
            if fallback is None:
                fallback = candidate
    if fallback is not None:
        return fallback

    raise FileNotFoundError(f"could not locate {_SOURCE_FILENAME}")


def _load_module(path: Path):
    """Load the insertion_sort module from its absolute file path."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _import_module_fallback():
    """Last-resort import via ``sys.path`` for packaged layouts."""
    search_roots = [Path.cwd(), *Path(__file__).resolve().parents]
    for root in reversed(search_roots):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    last_error: Exception | None = None
    for name in ("sorts.insertion_sort", "insertion_sort"):
        try:
            return importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - defensive
            last_error = exc
    raise FileNotFoundError(f"could not import {_SOURCE_FILENAME}") from last_error


try:
    SOURCE_PATH = _find_source_path()
except FileNotFoundError:  # pragma: no cover - fallback for packaged layouts
    insertion_sort_module = _import_module_fallback()
    SOURCE_PATH = Path(insertion_sort_module.__file__).resolve()
else:
    insertion_sort_module = _load_module(SOURCE_PATH)

insertion_sort = insertion_sort_module.insertion_sort


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "expected"),
    [
        ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([-2, -5, -45], [-45, -5, -2]),
        ([3], [3]),
        (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),
        ([2.5, -1.0, 0.0, 3.75], [-1.0, 0.0, 2.5, 3.75]),
    ],
)
def test_insertion_sort_sorts_ascending(collection, expected):
    """insertion_sort returns the collection ordered by ascending."""
    assert insertion_sort(collection) == expected


def test_insertion_sort_returns_same_object():
    """insertion_sort sorts in place and returns the very same object."""
    collection = [3, 1, 2]
    result = insertion_sort(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_insertion_sort_matches_builtin_sorted_on_random_ints():
    """insertion_sort agrees with sorted() on a random sample of integers."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert insertion_sort(collection) == sorted(collection)


def test_insertion_sort_matches_builtin_sorted_on_random_strings():
    """insertion_sort agrees with sorted() on random alphanumeric strings."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert insertion_sort(collection) == sorted(collection)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


def test_insertion_sort_empty_collection():
    """An empty collection is returned unchanged."""
    collection: list[int] = []
    assert insertion_sort(collection) == []
    assert collection == []


def test_insertion_sort_single_element_is_unchanged():
    """A one-element collection is already sorted."""
    collection = [42]
    assert insertion_sort(collection) == [42]


def test_insertion_sort_two_elements_unsorted():
    """The smallest non-trivial case swaps two out-of-order elements."""
    assert insertion_sort([2, 1]) == [1, 2]


def test_insertion_sort_two_elements_sorted():
    """The smallest non-trivial case leaves an ordered pair untouched."""
    assert insertion_sort([1, 2]) == [1, 2]


def test_insertion_sort_all_equal_elements():
    """A collection of identical elements stays identical."""
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_insertion_sort_preserves_length_and_multiset():
    """Sorting preserves the length and the multiset of elements."""
    collection = [4, 1, 3, 1, 2, 4, 0]
    original = sorted(collection)
    result = insertion_sort(collection)
    assert len(result) == len(original)
    assert result == original


def test_insertion_sort_works_with_mutable_sequence_subclass():
    """insertion_sort accepts any MutableSequence, e.g. a list subclass."""

    class MyList(list):
        pass

    collection = MyList([3, 1, 2])
    result = insertion_sort(collection)
    assert isinstance(result, MyList)
    assert list(result) == [1, 2, 3]


def test_insertion_sort_uses_only_less_than_comparisons():
    """insertion_sort relies solely on the ``<`` operator of the elements."""

    class OnlyLt:
        def __init__(self, value: int) -> None:
            self.value = value

        def __lt__(self, other: "OnlyLt", /) -> bool:
            return self.value < other.value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, OnlyLt) and self.value == other.value

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"OnlyLt({self.value})"

    collection = [OnlyLt(3), OnlyLt(1), OnlyLt(2)]
    result = insertion_sort(collection)
    assert [item.value for item in result] == [1, 2, 3]


def test_insertion_sort_does_not_mutate_input_when_already_sorted():
    """An already sorted collection is left byte-for-byte identical."""
    collection = [1, 2, 3, 4, 5]
    snapshot = list(collection)
    insertion_sort(collection)
    assert collection == snapshot


def test_insertion_sort_large_reversed_collection():
    """A large reversed collection is fully sorted (boundary on size)."""
    collection = list(range(200, 0, -1))
    assert insertion_sort(collection) == list(range(1, 201))


# ---------------------------------------------------------------------------
# Module-level behaviour
# ---------------------------------------------------------------------------


def test_module_exposes_insertion_sort_callable():
    """The module exposes a callable named insertion_sort."""
    assert callable(insertion_sort_module.insertion_sort)


def test_doctests_pass():
    """All doctests embedded in the module pass."""
    import doctest

    results = doctest.testmod(insertion_sort_module, verbose=False)
    assert results.failed == 0


def test_main_block_reads_input_and_prints_sorted_result():
    """Running the module as a script sorts comma-separated user input."""
    import runpy

    with patch("builtins.input", return_value="3,1,2"), patch(
        "builtins.print"
    ) as mock_print:
        runpy.run_path(str(SOURCE_PATH), run_name="__main__")

    printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list)
    assert "[1, 2, 3]" in printed
