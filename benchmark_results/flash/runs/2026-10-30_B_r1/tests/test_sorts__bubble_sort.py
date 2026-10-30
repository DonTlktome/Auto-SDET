"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive


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
        ([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_sort_returns_ascending_order(sort_func, collection, expected):
    """Each sort function orders a variety of collections in ascending order."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_matches_builtin_sorted_on_random_ints(sort_func):
    """Sorting a random sample of ints matches Python's built-in sorted()."""
    random.seed(1234)
    collection = random.sample(range(-50, 50), 100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_matches_builtin_sorted_on_random_strings(sort_func):
    """Sorting random alphanumeric strings matches Python's built-in sorted()."""
    random.seed(4321)
    collection = random.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_returns_same_list_object(sort_func):
    """The functions sort in place and return the very same list object."""
    collection = [3, 1, 2]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_does_not_change_length(sort_func):
    """Sorting preserves the number of elements in the collection."""
    collection = [5, 1, 4, 2, 3]
    assert len(sort_func(collection)) == 5


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_empty_list(sort_func):
    """An empty list is returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_single_element_list(sort_func):
    """A single-element list is returned unchanged."""
    assert sort_func([42]) == [42]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_two_elements_unsorted(sort_func):
    """The smallest non-trivial case (two elements) is sorted correctly."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_two_elements_already_sorted(sort_func):
    """An already sorted two-element list stays sorted."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_reverse_sorted_list(sort_func):
    """A fully descending list is sorted into ascending order."""
    assert sort_func([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_already_sorted_list(sort_func):
    """An already sorted list is returned unchanged."""
    assert sort_func([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_all_equal_elements(sort_func):
    """A list of identical elements is unchanged."""
    assert sort_func([7, 7, 7, 7, 7]) == [7, 7, 7, 7, 7]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_negative_and_positive_mix(sort_func):
    """Mixed negative/positive values are ordered correctly."""
    assert sort_func([0, -1, 1, -2, 2]) == [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_case_sensitive_strings(sort_func):
    """Uppercase letters sort before lowercase letters (ASCII ordering)."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_large_list(sort_func):
    """A larger list (200 elements) is sorted correctly."""
    random.seed(99)
    collection = [random.randint(-1000, 1000) for _ in range(200)]
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_raises_on_incomparable_types(sort_func):
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a"])


# ---------------------------------------------------------------------------
# Behavioural specifics
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early when no swaps occur in a pass."""
    collection = [1, 2, 3, 4, 5]
    with patch.object(
        sys.modules["bubble_sort"], "len", wraps=len, create=True
    ) as mock_len:
        bubble_sort_iterative(collection)
    # len() is called exactly once at the start of the function.
    assert mock_len.call_count == 1


def test_iterative_does_not_recurse():
    """The iterative implementation never calls itself recursively."""
    with patch(
        "bubble_sort.bubble_sort_iterative", wraps=bubble_sort_iterative
    ) as mock_sort:
        result = mock_sort([3, 1, 2])
    assert result == [1, 2, 3]
    assert mock_sort.call_count == 1


def test_recursive_calls_itself_for_unsorted_input():
    """The recursive version recurses at least once for unsorted input."""
    with patch(
        "bubble_sort.bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as mock_sort:
        result = mock_sort([3, 1, 2])
    assert result == [1, 2, 3]
    assert mock_sort.call_count > 1


def test_recursive_does_not_recurse_for_sorted_input():
    """The recursive version returns immediately when no swap is needed."""
    with patch(
        "bubble_sort.bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as mock_sort:
        result = mock_sort([1, 2, 3])
    assert result == [1, 2, 3]
    assert mock_sort.call_count == 1


def test_recursive_does_not_recurse_for_empty_input():
    """The recursive version handles an empty list without recursing."""
    with patch(
        "bubble_sort.bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as mock_sort:
        result = mock_sort([])
    assert result == []
    assert mock_sort.call_count == 1


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_is_idempotent(sort_func):
    """Sorting an already sorted result leaves it unchanged."""
    once = sort_func([4, 2, 9, 1])
    twice = sort_func(once)
    assert once == twice == [1, 2, 4, 9]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_accepts_list_of_tuples(sort_func):
    """Tuples are comparable, so a list of tuples sorts lexicographically."""
    assert sort_func([(2, "b"), (1, "z"), (1, "a")]) == [(1, "a"), (1, "z"), (2, "b")]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_uses_magicmock_comparable_objects(sort_func):
    """Objects exposing __gt__/__lt__ are sorted via their comparison protocol."""
    values = [3, 1, 2]
    mocks = []
    for value in values:
        mock = MagicMock()
        mock.value = value
        mock.__gt__ = lambda self, other: self.value > other.value
        mock.__lt__ = lambda self, other: self.value < other.value
        mocks.append(mock)

    result = sort_func(mocks)
    assert [m.value for m in result] == [1, 2, 3]
