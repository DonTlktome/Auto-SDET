"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
# Insert both the test file's directory (in case it sits next to the module)
# and its parent (in case the module lives in a `sorts` package directory).
_TEST_DIR = Path(__file__).resolve().parent
for _path in (_TEST_DIR, _TEST_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


# ---------------------------------------------------------------------------
# bubble_sort_iterative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([-2, -45, -5], [-45, -5, -2]),
        ([-23, 0, 6, -4, 34], [-23, -4, 0, 6, 34]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([3, 3, 3, 3], [3, 3, 3, 3]),
        ([56], [56]),
        (["d", "a", "b", "e"], ["a", "b", "d", "e"]),
        (["z", "a", "y", "b", "x", "c"], ["a", "b", "c", "x", "y", "z"]),
        ([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_bubble_sort_iterative_happy_path(collection, expected):
    """Iterative bubble sort returns the collection in ascending order."""
    assert bubble_sort_iterative(collection) == expected


def test_bubble_sort_iterative_empty_list():
    """An empty list is a valid boundary input and must be returned unchanged."""
    assert bubble_sort_iterative([]) == []


def test_bubble_sort_iterative_single_element():
    """A one-element list is already sorted and must be returned as-is."""
    assert bubble_sort_iterative([42]) == [42]


def test_bubble_sort_iterative_already_sorted():
    """An already sorted list must be returned unchanged (early-exit path)."""
    data = list(range(10))
    assert bubble_sort_iterative(data) == list(range(10))


def test_bubble_sort_iterative_reverse_sorted():
    """A reverse-sorted list is the worst case and must be fully reversed."""
    data = list(range(10, 0, -1))
    assert bubble_sort_iterative(data) == list(range(1, 11))


def test_bubble_sort_iterative_sorts_in_place_and_returns_same_object():
    """The function mutates and returns the very same list object."""
    data = [3, 1, 2]
    result = bubble_sort_iterative(data)
    assert result is data
    assert data == [1, 2, 3]


def test_bubble_sort_iterative_matches_builtin_sorted_random():
    """Randomized comparison against the built-in ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-100, 100) for _ in range(200)]
    assert bubble_sort_iterative(data[:]) == sorted(data)


def test_bubble_sort_iterative_mixed_strings_and_digits():
    """Heterogeneous-but-comparable string data sorts like ``sorted``."""
    rng = random.Random(99)
    data = rng.choices(string.ascii_letters + string.digits, k=100)
    assert bubble_sort_iterative(data[:]) == sorted(data)


def test_bubble_sort_iterative_raises_on_incomparable_types():
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        bubble_sort_iterative([1, "a"])


def test_bubble_sort_iterative_raises_on_none_input():
    """Passing ``None`` raises TypeError because ``len`` is unavailable."""
    with pytest.raises(TypeError):
        bubble_sort_iterative(None)


def test_bubble_sort_iterative_uses_comparison_operator():
    """Comparisons go through ``__gt__`` on the elements (mocked elements)."""
    first = MagicMock()
    second = MagicMock()
    first.__gt__ = MagicMock(return_value=True)
    second.__gt__ = MagicMock(return_value=False)

    result = bubble_sort_iterative([first, second])

    assert result == [second, first]
    first.__gt__.assert_called_once_with(second)


# ---------------------------------------------------------------------------
# bubble_sort_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([-2, -45, -5], [-45, -5, -2]),
        ([-23, 0, 6, -4, 34], [-23, -4, 0, 6, 34]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([3, 3, 3, 3], [3, 3, 3, 3]),
        ([56], [56]),
        (["d", "a", "b", "e"], ["a", "b", "d", "e"]),
        (["z", "a", "y", "b", "x", "c"], ["a", "b", "c", "x", "y", "z"]),
        (["a", "Z", "B", "C", "A", "c"], ["A", "B", "C", "Z", "a", "c"]),
        ([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_bubble_sort_recursive_happy_path(collection, expected):
    """Recursive bubble sort returns the collection in ascending order."""
    assert bubble_sort_recursive(collection) == expected


def test_bubble_sort_recursive_empty_list():
    """An empty list is a valid boundary input and must be returned unchanged."""
    assert bubble_sort_recursive([]) == []


def test_bubble_sort_recursive_single_element():
    """A one-element list is already sorted and must be returned as-is."""
    assert bubble_sort_recursive([7]) == [7]


def test_bubble_sort_recursive_two_elements_unsorted():
    """The smallest non-trivial case: two elements out of order."""
    assert bubble_sort_recursive([2, 1]) == [1, 2]


def test_bubble_sort_recursive_already_sorted():
    """An already sorted list must be returned unchanged (no recursion)."""
    data = list(range(10))
    assert bubble_sort_recursive(data) == list(range(10))


def test_bubble_sort_recursive_reverse_sorted():
    """A reverse-sorted list is the worst case and must be fully reversed."""
    data = list(range(10, 0, -1))
    assert bubble_sort_recursive(data) == list(range(1, 11))


def test_bubble_sort_recursive_sorts_in_place_and_returns_same_object():
    """The function mutates and returns the very same list object."""
    data = [3, 1, 2]
    result = bubble_sort_recursive(data)
    assert result is data
    assert data == [1, 2, 3]


def test_bubble_sort_recursive_matches_builtin_sorted_random():
    """Randomized comparison against the built-in ``sorted``."""
    rng = random.Random(4321)
    data = [rng.randint(-100, 100) for _ in range(100)]
    assert bubble_sort_recursive(data[:]) == sorted(data)


def test_bubble_sort_recursive_mixed_strings_and_digits():
    """Heterogeneous-but-comparable string data sorts like ``sorted``."""
    rng = random.Random(7)
    data = rng.choices(string.ascii_letters + string.digits, k=100)
    assert bubble_sort_recursive(data[:]) == sorted(data)


def test_bubble_sort_recursive_raises_on_incomparable_types():
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        bubble_sort_recursive([1, "a"])


def test_bubble_sort_recursive_raises_on_none_input():
    """Passing ``None`` raises TypeError because ``len`` is unavailable."""
    with pytest.raises(TypeError):
        bubble_sort_recursive(None)


def test_bubble_sort_recursive_recurses_until_sorted():
    """The recursive variant calls itself while swaps keep happening."""
    # Patch the already-imported module object directly instead of resolving
    # the target through a string import path (which requires the `sorts`
    # package to be importable again at patch time).
    module = sys.modules[bubble_sort_recursive.__module__]
    with patch.object(
        module,
        "bubble_sort_recursive",
        wraps=bubble_sort_recursive,
    ) as spy:
        result = spy([3, 2, 1])

    assert result == [1, 2, 3]
    # One initial call plus at least one recursive call.
    assert spy.call_count >= 2


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection",
    [
        [],
        [1],
        [2, 1],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [0, 0, 0],
        [-1, 1, -1, 1],
        [10, -10, 0, 5, -5],
    ],
)
def test_iterative_and_recursive_agree(collection):
    """Both implementations produce identical results for the same input."""
    assert bubble_sort_iterative(collection[:]) == bubble_sort_recursive(collection[:])


def test_both_implementations_handle_large_input():
    """Both implementations sort a larger random list identically to ``sorted``."""
    rng = random.Random(2024)
    data = [rng.randint(-500, 500) for _ in range(300)]
    expected = sorted(data)
    assert bubble_sort_iterative(data[:]) == expected
    assert bubble_sort_recursive(data[:]) == expected
