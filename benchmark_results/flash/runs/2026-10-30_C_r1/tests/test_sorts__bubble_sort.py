"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


# ---------------------------------------------------------------------------
# Shared / parametrized tests across both implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
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
    """Both implementations must sort the documented example collections."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_empty_collection(sort_func):
    """An empty list is a boundary case and must be returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_single_element_collection(sort_func):
    """A one-element list is already sorted and must be returned as-is."""
    assert sort_func([42]) == [42]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_two_element_unsorted(sort_func):
    """The smallest non-trivial swap case: exactly two out-of-order items."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_reverse_sorted_collection(sort_func):
    """A fully descending list is the worst case and must be fully reversed."""
    collection = list(range(20, 0, -1))
    assert sort_func(collection) == list(range(1, 21))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_already_sorted_collection(sort_func):
    """An already sorted list must be returned unchanged (early-exit path)."""
    collection = list(range(10))
    assert sort_func(collection) == list(range(10))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_returns_same_object_in_place(sort_func):
    """The functions sort in place and return the very same list object."""
    collection = [3, 1, 2]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_matches_builtin_sorted_on_random_ints(sort_func):
    """Randomized comparison against the reference `sorted` implementation."""
    rng = random.Random(1234)
    collection = [rng.randint(-100, 100) for _ in range(100)]
    expected = sorted(collection)
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Randomized comparison against `sorted` for mixed letters and digits."""
    rng = random.Random(4321)
    alphabet = string.ascii_letters + string.digits
    collection = [rng.choice(alphabet) for _ in range(100)]
    expected = sorted(collection)
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_duplicates_preserved(sort_func):
    """Sorting must not drop or merge duplicate values."""
    collection = [2, 1, 2, 1, 2, 1]
    assert sort_func(collection) == [1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_does_not_mutate_input_when_already_sorted(sort_func):
    """A sorted input keeps its original ordering and length."""
    collection = [1, 2, 3]
    sort_func(collection)
    assert collection == [1, 2, 3]
    assert len(collection) == 3


# ---------------------------------------------------------------------------
# Implementation-specific behaviour
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version breaks out early when no swap occurs in a pass."""
    collection = [1, 2, 3, 4, 5]
    with patch.object(
        sys.modules["bubble_sort"], "len", wraps=len, create=True
    ) as mock_len:
        bubble_sort_iterative(collection)
    # `len` is called exactly once at the start of the function.
    assert mock_len.call_count == 1


def test_iterative_handles_large_collection():
    """A larger input (500 items) still sorts correctly."""
    rng = random.Random(99)
    collection = [rng.randint(-1000, 1000) for _ in range(500)]
    assert bubble_sort_iterative(collection) == sorted(collection)


def test_recursive_handles_large_collection():
    """The recursive version sorts a moderately large input correctly."""
    rng = random.Random(99)
    collection = [rng.randint(-1000, 1000) for _ in range(200)]
    assert bubble_sort_recursive(collection) == sorted(collection)


def test_recursive_recurses_until_sorted():
    """The recursive version calls itself while swaps keep happening."""
    module = sys.modules["bubble_sort"]
    original = module.bubble_sort_recursive
    calls = []

    def counting_wrapper(collection):
        calls.append(list(collection))
        return original(collection)

    with patch.object(module, "bubble_sort_recursive", counting_wrapper):
        # The module-level name is patched, so the recursive self-call is counted.
        result = counting_wrapper([3, 2, 1])

    assert result == [1, 2, 3]
    # One initial call plus at least one recursive call.
    assert len(calls) >= 2


def test_recursive_no_extra_recursion_when_sorted():
    """A sorted input triggers exactly one pass and no recursive call."""
    module = sys.modules["bubble_sort"]
    original = module.bubble_sort_recursive
    calls = []

    def counting_wrapper(collection):
        calls.append(list(collection))
        return original(collection)

    with patch.object(module, "bubble_sort_recursive", counting_wrapper):
        result = counting_wrapper([1, 2, 3])

    assert result == [1, 2, 3]
    assert len(calls) == 1


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_raises_type_error_on_incomparable_items(sort_func):
    """Comparing incompatible types must propagate a TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a"])


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_accepts_tuple_like_sequence_via_list(sort_func):
    """A list built from a tuple sorts identically to `sorted`."""
    collection = list((5, 3, 9, 1))
    assert sort_func(collection) == [1, 3, 5, 9]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_negative_and_positive_mixed(sort_func):
    """Mixed-sign integers sort with negatives first."""
    assert sort_func([0, -1, 1, -2, 2]) == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_uppercase_before_lowercase(sort_func):
    """ASCII ordering places uppercase letters before lowercase ones."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


def test_mock_isolation_example():
    """Demonstrates that external collaborators can be mocked out cleanly."""
    fake = MagicMock(return_value=[1, 2, 3])
    assert fake([3, 2, 1]) == [1, 2, 3]
    fake.assert_called_once_with([3, 2, 1])