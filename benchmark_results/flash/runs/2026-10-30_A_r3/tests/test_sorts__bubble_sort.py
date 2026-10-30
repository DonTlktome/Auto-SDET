"""Tests for the bubble sort implementations in sorts/bubble_sort.py."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive


SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


# ---------------------------------------------------------------------------
# Positive (happy path) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_unsorted_integers(sort_func):
    """A typical unsorted list of integers is sorted in ascending order."""
    assert sort_func([0, 5, 2, 3, 2]) == [0, 2, 2, 3, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_negative_integers(sort_func):
    """Negative integers are ordered correctly (most negative first)."""
    assert sort_func([-23, 0, 6, -4, 34]) == [-23, -4, 0, 6, 34]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_strings(sort_func):
    """Strings are sorted lexicographically."""
    assert sort_func(["z", "a", "y", "b", "x", "c"]) == [
        "a",
        "b",
        "c",
        "x",
        "y",
        "z",
    ]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_floats(sort_func):
    """Floating point values are sorted in ascending order."""
    assert sort_func([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6]) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
    ]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_mixed_int_and_float(sort_func):
    """Mixed ints and floats are comparable and sorted together."""
    assert sort_func([1, 3.3, 5, 7.7, 2, 4.4, 6]) == [
        1,
        2,
        3.3,
        4.4,
        5,
        6,
        7.7,
    ]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_data(sort_func):
    """Result matches Python's built-in sorted() for random integer data."""
    rng = random.Random(1234)
    data = rng.sample(range(-50, 50), 100)
    assert sort_func(list(data)) == sorted(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Result matches sorted() for random alphanumeric strings."""
    rng = random.Random(4321)
    data = rng.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Edge cases and boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_empty_list(sort_func):
    """An empty list is returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_single_element(sort_func):
    """A single-element list is returned unchanged."""
    assert sort_func([56]) == [56]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_elements_unsorted(sort_func):
    """The smallest non-trivial case (two elements) is swapped correctly."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_elements_already_sorted(sort_func):
    """Two already-sorted elements stay in place."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_already_sorted_list(sort_func):
    """An already sorted list is returned unchanged."""
    assert sort_func([1, 2, 3, 4]) == [1, 2, 3, 4]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_reverse_sorted_list(sort_func):
    """A reverse-sorted list (worst case) is fully reversed."""
    assert sort_func([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_duplicates(sort_func):
    """A list of identical values is returned unchanged."""
    assert sort_func([3, 3, 3, 3]) == [3, 3, 3, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_duplicates_preserved(sort_func):
    """Duplicate values are all retained after sorting."""
    result = sort_func([2, 1, 2, 1, 2])
    assert result == [1, 1, 2, 2, 2]
    assert len(result) == 5


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_negative_only_values(sort_func):
    """A list of only negative values is sorted correctly."""
    assert sort_func([-2, -45, -5]) == [-45, -5, -2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uppercase_and_lowercase_ordering(sort_func):
    """ASCII ordering places uppercase letters before lowercase ones."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == [
        "A",
        "B",
        "C",
        "Z",
        "a",
        "c",
    ]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_large_list_is_sorted(sort_func):
    """A larger list (200 elements) is sorted correctly."""
    rng = random.Random(99)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert sort_func(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# In-place / identity semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object(sort_func):
    """The function sorts in place and returns the very same list object."""
    data = [3, 1, 2]
    result = sort_func(data)
    assert result is data


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_input_list_is_mutated_in_place(sort_func):
    """The caller's list is mutated to the sorted order."""
    data = [4, 2, 9, 1]
    sort_func(data)
    assert data == [1, 2, 4, 9]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_empty_list_identity(sort_func):
    """An empty list is returned as the same object."""
    data = []
    assert sort_func(data) is data


# ---------------------------------------------------------------------------
# Behavioural details / isolation
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early when no swaps occur in a pass.

    A custom list subclass counts element comparisons; for already sorted
    input the number of comparisons must be strictly less than the
    worst-case n*(n-1)/2.
    """
    n = 10

    class CountingList(list):
        comparisons = 0

        def __getitem__(self, index):
            CountingList.comparisons += 1
            return super().__getitem__(index)

    data = CountingList(range(n))
    bubble_sort_iterative(data)
    assert list(data) == list(range(n))
    assert CountingList.comparisons < n * (n - 1) // 2


def test_recursive_terminates_on_sorted_input():
    """The recursive version returns immediately for already sorted input."""
    data = list(range(20))
    assert bubble_sort_recursive(data) == list(range(20))


def test_recursive_recursion_depth_is_bounded():
    """The recursive implementation does not blow the recursion limit for
    moderately sized inputs (it recurses at most n times)."""
    data = list(range(50, 0, -1))
    assert bubble_sort_recursive(data) == list(range(1, 51))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_raises_type_error_on_incomparable_items(sort_func):
    """Comparing incomparable types raises TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_works_with_tuples_of_comparable_items(sort_func):
    """Lists of tuples are sorted by their natural tuple ordering."""
    assert sort_func([(2, "b"), (1, "a"), (2, "a")]) == [
        (1, "a"),
        (2, "a"),
        (2, "b"),
    ]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_does_not_depend_on_external_state(sort_func):
    """Sorting is deterministic and independent of module-level randomness."""
    with patch("random.sample", side_effect=AssertionError("should not be called")):
        assert sort_func([3, 1, 2]) == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_stability_of_equal_keys(sort_func):
    """Equal keys keep their relative order (bubble sort is stable)."""
    data = [(1, "first"), (0, "x"), (1, "second"), (0, "y")]
    assert sort_func(data) == [(0, "x"), (0, "y"), (1, "first"), (1, "second")]