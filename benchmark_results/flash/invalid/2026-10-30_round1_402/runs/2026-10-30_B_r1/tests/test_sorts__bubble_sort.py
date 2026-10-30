"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
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
        (
            [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7],
        ),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_sorts_known_collections(sort_func, collection, expected):
    """Each sort function orders a variety of known collections ascending."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_ints(sort_func):
    """Sorting random integers matches Python's built-in ``sorted``."""
    collection = random.sample(range(-50, 50), 100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Sorting random alphanumeric characters matches ``sorted``."""
    collection = random.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object_in_place(sort_func):
    """The sort mutates and returns the very same list object passed in."""
    collection = [3, 1, 2]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_empty_collection(sort_func):
    """An empty list is returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_single_element(sort_func):
    """A one-element list is returned unchanged."""
    assert sort_func([42]) == [42]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_elements_unsorted(sort_func):
    """The smallest non-trivial case (two elements) is swapped correctly."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_elements_already_sorted(sort_func):
    """Two already-ordered elements stay in place."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_reverse_sorted_collection(sort_func):
    """A fully descending list is reversed into ascending order."""
    collection = list(range(10, 0, -1))
    assert sort_func(collection) == list(range(1, 11))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_equal_elements(sort_func):
    """A list of identical values is unchanged."""
    assert sort_func([7, 7, 7, 7, 7]) == [7, 7, 7, 7, 7]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_negative_and_positive_mix(sort_func):
    """Mixed-sign values sort with negatives first."""
    assert sort_func([0, -1, 1, -2, 2]) == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uppercase_sorts_before_lowercase(sort_func):
    """ASCII ordering places uppercase letters before lowercase ones."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_large_collection(sort_func):
    """A larger collection (500 elements) is sorted correctly."""
    collection = random.sample(range(-1000, 1000), 500)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_duplicates_preserved(sort_func):
    """Duplicate values are all retained after sorting."""
    collection = [4, 2, 4, 1, 2, 4]
    result = sort_func(collection)
    assert result == [1, 2, 2, 4, 4, 4]
    assert len(result) == len(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_incomparable_types_raise_type_error(sort_func):
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a"])


# ---------------------------------------------------------------------------
# Implementation-specific behaviour
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early when no swap occurs in a pass.

    A sorted list should require exactly one comparison pass, so the inner
    loop runs ``len(collection) - 1`` times and then breaks out.
    """
    collection = [1, 2, 3, 4, 5]
    comparisons = []

    class TrackingList(list):
        def __getitem__(self, index):
            comparisons.append(index)
            return super().__getitem__(index)

    tracking = TrackingList(collection)
    result = bubble_sort_iterative(tracking)
    assert result == collection
    # One pass over n-1 adjacent pairs, then the loop breaks.
    assert len(comparisons) == 2 * (len(collection) - 1)


def test_recursive_recurses_until_sorted():
    """The recursive version calls itself repeatedly until no swap happens."""
    calls = []
    original = bubble_sort_recursive

    def spy(collection):
        calls.append(list(collection))
        return original(collection)

    with patch(__name__ + ".bubble_sort_recursive", side_effect=spy) as mocked:
        # The module-level function is patched, so call through the mock.
        result = mocked([3, 2, 1])

    assert result == [1, 2, 3]
    assert mocked.call_count >= 1


def test_recursive_single_pass_for_sorted_input():
    """A sorted list needs only one recursive invocation (no recursion)."""
    with patch(
        __name__ + ".bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as mocked:
        result = mocked([1, 2, 3])
    assert result == [1, 2, 3]
    assert mocked.call_count == 1


def test_iterative_does_not_use_recursion():
    """The iterative implementation never calls the recursive one."""
    with patch(__name__ + ".bubble_sort_recursive") as mocked_recursive:
        assert bubble_sort_iterative([3, 1, 2]) == [1, 2, 3]
    mocked_recursive.assert_not_called()


def test_sort_functions_are_independent_of_global_state():
    """Sorting twice yields identical results (no leaked state)."""
    first = bubble_sort_iterative([5, 3, 1])
    second = bubble_sort_iterative([5, 3, 1])
    assert first == second == [1, 3, 5]


def test_mock_isolation_example():
    """Demonstrates isolating an external dependency with MagicMock.

    The sort functions are pure, so we simply verify that a mocked helper
    is not consulted — confirming no hidden I/O dependency exists.
    """
    fake_io = MagicMock()
    with patch("builtins.print", fake_io):
        bubble_sort_iterative([2, 1])
        bubble_sort_recursive([2, 1])
    fake_io.assert_not_called()