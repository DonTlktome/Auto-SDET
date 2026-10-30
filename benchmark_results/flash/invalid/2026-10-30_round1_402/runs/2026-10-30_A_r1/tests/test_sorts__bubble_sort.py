"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
from unittest.mock import patch

import pytest

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


@pytest.fixture(params=SORT_FUNCTIONS, ids=["iterative", "recursive"])
def sort_func(request):
    """Parametrized fixture yielding each public sorting function."""
    return request.param


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
        (["z", "a", "y", "b", "x", "c"], ["a", "b", "c", "x", "y", "z"]),
        (
            [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7],
        ),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_bubble_sort_iterative_happy_path(collection, expected):
    """Iterative sort returns the collection in ascending order."""
    assert bubble_sort_iterative(collection) == expected


def test_bubble_sort_iterative_empty_list():
    """An empty list is a valid boundary input and must be returned unchanged."""
    assert bubble_sort_iterative([]) == []


def test_bubble_sort_iterative_single_element():
    """A one-element list is already sorted and must be returned as-is."""
    assert bubble_sort_iterative([42]) == [42]


def test_bubble_sort_iterative_already_sorted():
    """An already sorted list must be returned unchanged."""
    data = list(range(10))
    assert bubble_sort_iterative(data) == list(range(10))


def test_bubble_sort_iterative_reverse_sorted():
    """A reverse-sorted list is the worst case and must be fully reversed."""
    data = list(range(10, 0, -1))
    assert bubble_sort_iterative(data) == list(range(1, 11))


def test_bubble_sort_iterative_mutates_in_place_and_returns_same_object():
    """The function sorts in place and returns the very same list object."""
    data = [3, 1, 2]
    result = bubble_sort_iterative(data)
    assert result is data
    assert data == [1, 2, 3]


def test_bubble_sort_iterative_matches_builtin_sorted_random():
    """Randomized comparison against the builtin ``sorted``."""
    rng = random.Random(1234)
    data = [rng.randint(-100, 100) for _ in range(200)]
    assert bubble_sort_iterative(data[:]) == sorted(data)


def test_bubble_sort_iterative_matches_builtin_sorted_strings():
    """Randomized comparison against ``sorted`` for mixed alphanumeric strings."""
    rng = random.Random(4321)
    alphabet = string.ascii_letters + string.digits
    data = [rng.choice(alphabet) for _ in range(100)]
    assert bubble_sort_iterative(data[:]) == sorted(data)


def test_bubble_sort_iterative_early_exit_on_sorted_input():
    """The early-exit optimisation must stop after a single pass on sorted input."""
    data = [1, 2, 3, 4, 5]

    class CountingList(list):
        """List subclass counting ``__getitem__`` accesses."""

        def __init__(self, *args):
            super().__init__(*args)
            self.getitem_calls = 0

        def __getitem__(self, index):
            self.getitem_calls += 1
            return super().__getitem__(index)

    counting = CountingList(data)
    bubble_sort_iterative(counting)
    # One pass over 5 elements needs at most 2 * 4 = 8 lookups (plus len()).
    assert counting.getitem_calls <= 8


def test_bubble_sort_iterative_raises_on_incomparable_types():
    """Comparing incomparable types must propagate a TypeError."""
    with pytest.raises(TypeError):
        bubble_sort_iterative([1, "a"])


# ---------------------------------------------------------------------------
# bubble_sort_recursive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([-2, -45, -5], [-45, -5, -2]),
        ([-23, 0, 6, -4, 34], [-23, -4, 0, 6, 34]),
        (["z", "a", "y", "b", "x", "c"], ["a", "b", "c", "x", "y", "z"]),
        (["a", "Z", "B", "C", "A", "c"], ["A", "B", "C", "Z", "a", "c"]),
        (
            [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6],
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7],
        ),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_bubble_sort_recursive_happy_path(collection, expected):
    """Recursive sort returns the collection in ascending order."""
    assert bubble_sort_recursive(collection) == expected


def test_bubble_sort_recursive_empty_list():
    """An empty list is a valid boundary input and must be returned unchanged."""
    assert bubble_sort_recursive([]) == []


def test_bubble_sort_recursive_single_element():
    """A one-element list is already sorted and must be returned as-is."""
    assert bubble_sort_recursive([7]) == [7]


def test_bubble_sort_recursive_already_sorted():
    """An already sorted list must be returned unchanged."""
    data = list(range(10))
    assert bubble_sort_recursive(data) == list(range(10))


def test_bubble_sort_recursive_reverse_sorted():
    """A reverse-sorted list is the worst case and must be fully reversed."""
    data = list(range(10, 0, -1))
    assert bubble_sort_recursive(data) == list(range(1, 11))


def test_bubble_sort_recursive_mutates_in_place_and_returns_same_object():
    """The function sorts in place and returns the very same list object."""
    data = [3, 1, 2]
    result = bubble_sort_recursive(data)
    assert result is data
    assert data == [1, 2, 3]


def test_bubble_sort_recursive_matches_builtin_sorted_random():
    """Randomized comparison against the builtin ``sorted``."""
    rng = random.Random(9876)
    data = [rng.randint(-100, 100) for _ in range(100)]
    assert bubble_sort_recursive(data[:]) == sorted(data)


def test_bubble_sort_recursive_matches_builtin_sorted_strings():
    """Randomized comparison against ``sorted`` for mixed alphanumeric strings."""
    rng = random.Random(6789)
    alphabet = string.ascii_letters + string.digits
    data = [rng.choice(alphabet) for _ in range(100)]
    assert bubble_sort_recursive(data[:]) == sorted(data)


def test_bubble_sort_recursive_raises_on_incomparable_types():
    """Comparing incomparable types must propagate a TypeError."""
    with pytest.raises(TypeError):
        bubble_sort_recursive([1, "a"])


def test_bubble_sort_recursive_recursion_depth_is_bounded():
    """Recursion must terminate for a moderately sized input (no RecursionError)."""
    data = list(range(50, 0, -1))
    assert bubble_sort_recursive(data) == list(range(1, 51))


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
        [0, 0, 0, 0],
        [-1, 1, -1, 1],
        ["b", "a", "c", "a"],
    ],
)
def test_both_implementations_agree(collection):
    """Iterative and recursive variants must produce identical results."""
    assert bubble_sort_iterative(collection[:]) == bubble_sort_recursive(collection[:])


def test_both_implementations_agree_on_random_data():
    """Both variants must agree with each other and with ``sorted`` on random data."""
    rng = random.Random(2024)
    data = [rng.randint(-500, 500) for _ in range(150)]
    expected = sorted(data)
    assert bubble_sort_iterative(data[:]) == expected
    assert bubble_sort_recursive(data[:]) == expected


# ---------------------------------------------------------------------------
# Isolation / mocking
# ---------------------------------------------------------------------------


def test_bubble_sort_iterative_uses_no_external_io():
    """Sorting must not perform any file or network I/O."""
    with patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
        assert bubble_sort_iterative([3, 1, 2]) == [1, 2, 3]


def test_bubble_sort_recursive_uses_no_external_io():
    """Sorting must not perform any file or network I/O."""
    with patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
        assert bubble_sort_recursive([3, 1, 2]) == [1, 2, 3]


def test_sort_func_fixture_covers_both_implementations(sort_func):
    """The parametrized fixture must exercise both public sort functions."""
    assert sort_func([3, 1, 2]) == [1, 2, 3]
    assert sort_func([]) == []
    assert sort_func([1]) == [1]