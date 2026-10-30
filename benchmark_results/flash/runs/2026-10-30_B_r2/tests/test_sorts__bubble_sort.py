"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


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
    """Both implementations must sort representative collections correctly."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_ints(sort_func):
    """Result must equal Python's built-in sorted() for random integers."""
    rng = random.Random(1234)
    collection = rng.sample(range(-50, 50), 100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Result must equal sorted() for random alphanumeric strings."""
    rng = random.Random(4321)
    collection = rng.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object_in_place(sort_func):
    """The function sorts in place and returns the very same list object."""
    collection = [4, 1, 3, 2]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3, 4]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_does_not_change_length_or_multiset(sort_func):
    """Sorting must preserve length and element multiset."""
    collection = [5, 1, 5, 3, 1, 9, 0]
    original_sorted = sorted(collection)
    result = sort_func(collection)
    assert len(result) == len(original_sorted)
    assert result == original_sorted


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_empty_list(sort_func):
    """An empty list must be returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_single_element_list(sort_func):
    """A one-element list is already sorted and returned as-is."""
    collection = [42]
    assert sort_func(collection) == [42]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_element_unsorted(sort_func):
    """Smallest non-trivial case: two elements out of order."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_element_sorted(sort_func):
    """Two elements already in order stay in order."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_already_sorted_large_list(sort_func):
    """A large already-sorted list must be returned unchanged."""
    collection = list(range(200))
    assert sort_func(collection) == list(range(200))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_reverse_sorted_list(sort_func):
    """Worst case for bubble sort: fully reversed input."""
    collection = list(range(50, 0, -1))
    assert sort_func(collection) == list(range(1, 51))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_equal_elements(sort_func):
    """All-equal elements must not be reordered or lost."""
    collection = [7] * 10
    assert sort_func(collection) == [7] * 10


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_negative_and_zero_values(sort_func):
    """Negative numbers and zero must sort correctly."""
    assert sort_func([0, -1, -100, 5, -50]) == [-100, -50, -1, 0, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uppercase_before_lowercase(sort_func):
    """ASCII ordering: uppercase letters sort before lowercase letters."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_duplicates_preserved(sort_func):
    """Duplicate values must all be retained after sorting."""
    collection = [3, 1, 2, 3, 1, 2, 3]
    assert sort_func(collection) == [1, 1, 2, 2, 3, 3, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_tuples_as_elements(sort_func):
    """Comparable non-scalar elements (tuples) must sort lexicographically."""
    collection = [(2, "b"), (1, "z"), (1, "a")]
    assert sort_func(collection) == [(1, "a"), (1, "z"), (2, "b")]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_incomparable_elements_raise_type_error(sort_func):
    """Mixed incomparable types must raise TypeError, not silently pass."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


# ---------------------------------------------------------------------------
# Implementation-specific behaviour
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version breaks out early when no swap occurs.

    We verify this indirectly by counting comparisons through a custom
    sequence-like object that records ``__getitem__`` accesses.
    """

    class CountingList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.getitem_calls = 0

        def __getitem__(self, index):
            self.getitem_calls += 1
            return super().__getitem__(index)

    collection = CountingList([1, 2, 3, 4, 5])
    bubble_sort_iterative(collection)
    # A single pass over n elements needs at most 2*(n-1) reads; a full
    # O(n^2) run would need far more.
    assert collection.getitem_calls <= 2 * (len(collection) - 1)


def test_recursive_recurses_until_sorted():
    """The recursive version calls itself repeatedly until no swap happens."""
    collection = [3, 2, 1]
    result = bubble_sort_recursive(collection)
    assert result == [1, 2, 3]


def test_recursive_does_not_recurse_for_sorted_input():
    """A sorted input must not trigger a recursive call (single pass only)."""
    collection = [1, 2, 3]
    # Resolve the module object that actually owns the function instead of
    # re-importing it from a dotted string (which may not be importable).
    module = sys.modules[bubble_sort_recursive.__module__]
    original = module.bubble_sort_recursive
    with patch.object(module, "bubble_sort_recursive", wraps=original) as spy:
        module.bubble_sort_recursive(collection)
        # Sorted input: one single pass, no recursive self-call.
        assert spy.call_count == 1
        assert collection == [1, 2, 3]
    # The patched attribute is restored on exit.
    assert module.bubble_sort_recursive is original


def test_recursive_does_recurse_for_unsorted_input():
    """Unsorted input must trigger additional recursive calls."""
    collection = [3, 2, 1]
    module = sys.modules[bubble_sort_recursive.__module__]
    original = module.bubble_sort_recursive
    with patch.object(module, "bubble_sort_recursive", wraps=original) as spy:
        result = module.bubble_sort_recursive(collection)
        assert result == [1, 2, 3]
        assert spy.call_count > 1
    assert module.bubble_sort_recursive is original


def test_iterative_handles_large_random_input():
    """Stress test: iterative sort handles a moderately large random list."""
    rng = random.Random(99)
    collection = [rng.randint(-1000, 1000) for _ in range(300)]
    assert bubble_sort_iterative(collection) == sorted(collection)


def test_recursive_handles_large_random_input():
    """Stress test: recursive sort handles a moderately large random list."""
    rng = random.Random(100)
    collection = [rng.randint(-1000, 1000) for _ in range(300)]
    assert bubble_sort_recursive(collection) == sorted(collection)


def test_both_implementations_agree():
    """Iterative and recursive implementations must produce identical output."""
    rng = random.Random(7)
    for _ in range(20):
        collection = [rng.randint(-20, 20) for _ in range(rng.randint(0, 25))]
        assert bubble_sort_iterative(list(collection)) == bubble_sort_recursive(
            list(collection)
        )


def test_doctest_examples_pass():
    """Run the module's doctests to guarantee documented examples hold."""
    import doctest

    import sorts.bubble_sort as module

    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0
