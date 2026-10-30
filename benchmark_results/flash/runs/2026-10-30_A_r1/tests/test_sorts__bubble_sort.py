"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bubble_sort  # noqa: E402
from bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_unsorted_integers(sort_func):
    """A plain unsorted list of ints is returned in ascending order."""
    assert sort_func([0, 5, 2, 3, 2]) == [0, 2, 2, 3, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_negative_integers(sort_func):
    """Negative integers are ordered correctly (most negative first)."""
    assert sort_func([-23, 0, 6, -4, 34]) == [-23, -4, 0, 6, 34]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_floats(sort_func):
    """Floats are sorted in ascending order."""
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
    """Heterogeneous but comparable numeric types are sorted together."""
    assert sort_func([1, 3.3, 5, 7.7, 2, 4.4, 6]) == [1, 2, 3.3, 4.4, 5, 6, 7.7]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_strings(sort_func):
    """Strings are sorted lexicographically."""
    assert sort_func(["z", "a", "y", "b", "x", "c"]) == ["a", "b", "c", "x", "y", "z"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_strings_with_mixed_case(sort_func):
    """Uppercase letters sort before lowercase ones (ASCII ordering)."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_data(sort_func):
    """Result matches Python's built-in sorted() for random integer data."""
    data = random.sample(range(-50, 50), 100)
    assert sort_func(data) == sorted(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Result matches sorted() for random alphanumeric strings."""
    data = random.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(data) == sorted(data)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_empty_list(sort_func):
    """An empty list is returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_single_element_list(sort_func):
    """A one-element list is returned unchanged."""
    assert sort_func([56]) == [56]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_element_list_unsorted(sort_func):
    """The smallest non-trivial case (two elements) is swapped correctly."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_two_element_list_already_sorted(sort_func):
    """A two-element already-sorted list is left untouched."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_already_sorted_list(sort_func):
    """An already sorted list stays sorted."""
    assert sort_func([1, 2, 3, 4]) == [1, 2, 3, 4]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_reverse_sorted_list(sort_func):
    """A descending list is fully reversed into ascending order."""
    assert sort_func([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_duplicates(sort_func):
    """A list of identical values is returned unchanged."""
    assert sort_func([3, 3, 3, 3]) == [3, 3, 3, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_negative_values(sort_func):
    """A list of only negative values is sorted ascending."""
    assert sort_func([-2, -45, -5]) == [-45, -5, -2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object_identity(sort_func):
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
def test_does_not_change_length(sort_func):
    """Sorting preserves the number of elements."""
    data = [7, 7, 1, 0, -3, 7]
    assert len(sort_func(data)) == len(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_large_list_matches_sorted(sort_func):
    """A larger list (500 elements) still matches sorted()."""
    data = [random.randint(-1000, 1000) for _ in range(500)]
    assert sort_func(data) == sorted(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_tuples_inside_list_are_sorted(sort_func):
    """Comparable non-scalar items (tuples) are sorted correctly."""
    assert sort_func([(2, "b"), (1, "a"), (2, "a")]) == [(1, "a"), (2, "a"), (2, "b")]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_incomparable_items_raise_type_error(sort_func):
    """Mixing incomparable types raises TypeError from the comparison."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


# ---------------------------------------------------------------------------
# Behavioural / implementation-detail tests
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early: a sorted list needs only one pass.

    We verify this indirectly by counting comparisons through a list subclass
    that records how many times ``__gt__`` is invoked.
    """

    class CountingList(list):
        comparisons = 0

        def __getitem__(self, index):
            return super().__getitem__(index)

    class CountingInt(int):
        def __gt__(self, other):
            CountingList.comparisons += 1
            return super().__gt__(other)

    data = CountingList([CountingInt(i) for i in range(10)])
    bubble_sort_iterative(data)
    # One full pass over 9 adjacent pairs, then the loop breaks.
    assert CountingList.comparisons == 9


def test_recursive_returns_immediately_when_already_sorted():
    """The recursive version performs a single pass and returns for sorted input."""
    data = [1, 2, 3, 4, 5]
    with patch.object(
        bubble_sort, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        result = spy(data)
    assert result == [1, 2, 3, 4, 5]
    assert spy.call_count == 1


def test_recursive_recurses_for_unsorted_input():
    """The recursive version calls itself again when a swap occurred."""
    data = [3, 2, 1]
    with patch.object(
        bubble_sort, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        result = spy(data)
    assert result == [1, 2, 3]
    assert spy.call_count > 1


def test_iterative_and_recursive_agree():
    """Both implementations produce identical output for the same input."""
    data = [random.randint(-20, 20) for _ in range(50)]
    assert bubble_sort_iterative(list(data)) == bubble_sort_recursive(list(data))


def test_mock_used_to_isolate_comparison_side_effects():
    """Sanity check that MagicMock can stand in for a collection-like object.

    This documents that the sort functions only rely on ``len``, indexing and
    ``>`` — nothing else — so they can be exercised with a mock-backed object.
    """
    fake = MagicMock()
    fake.__len__.return_value = 0
    assert bubble_sort_iterative(fake) is fake
    assert bubble_sort_recursive(fake) is fake
