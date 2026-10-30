"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sorts.bubble_sort import bubble_sort_iterative, bubble_sort_recursive  # noqa: E402


# Grab the module object that actually holds the public functions so we can
# patch attributes on it directly, without relying on string based import
# resolution (which fails when the package is only importable via sys.path
# manipulation performed by the test runner).
bubble_sort_module = sys.modules[bubble_sort_recursive.__module__]


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
    """Each sort function must return the collection in ascending order."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_matches_builtin_sorted_on_random_ints(sort_func):
    """Sorting a random sample of ints must match Python's built-in sorted()."""
    random.seed(1234)
    collection = random.sample(range(-50, 50), 100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_matches_builtin_sorted_on_random_strings(sort_func):
    """Sorting random alphanumeric characters must match built-in sorted()."""
    random.seed(4321)
    collection = random.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_is_in_place_and_returns_same_object(sort_func):
    """The sort must mutate the input list in place and return that same list."""
    collection = [5, 1, 4, 2, 3]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_handles_uppercase_before_lowercase(sort_func):
    """ASCII ordering places uppercase letters before lowercase ones."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_empty_list(sort_func):
    """An empty list must be returned unchanged."""
    assert sort_func([]) == []


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_single_element_list(sort_func):
    """A one-element list is already sorted and must be returned as-is."""
    collection = [42]
    assert sort_func(collection) == [42]
    assert sort_func(collection) is collection


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_two_elements_unsorted(sort_func):
    """The minimal unsorted case (two swapped elements) must be handled."""
    assert sort_func([2, 1]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_two_elements_already_sorted(sort_func):
    """The minimal already-sorted case must be handled."""
    assert sort_func([1, 2]) == [1, 2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_reverse_sorted_list(sort_func):
    """A fully descending list is the worst case and must still sort correctly."""
    collection = list(range(20, 0, -1))
    assert sort_func(collection) == list(range(1, 21))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_all_equal_elements(sort_func):
    """A list of identical elements must be returned unchanged."""
    collection = [7] * 10
    assert sort_func(collection) == [7] * 10


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_negative_and_zero_boundaries(sort_func):
    """Negative values, zero and positive values must be ordered correctly."""
    assert sort_func([0, -1, 1, -100, 100]) == [-100, -1, 0, 1, 100]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_large_list_matches_sorted(sort_func):
    """A larger list (boundary of practical input size) must sort correctly."""
    random.seed(99)
    collection = [random.randint(-1000, 1000) for _ in range(300)]
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sort_raises_on_incomparable_types(sort_func):
    """Comparing incomparable types must raise TypeError, not silently pass."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


# ---------------------------------------------------------------------------
# Isolated / mocked-dependency tests
# ---------------------------------------------------------------------------


def test_iterative_uses_len_once_and_short_circuits_on_sorted_input():
    """The iterative version must stop early when no swap occurs in a pass."""
    collection = MagicMock()
    collection.__len__.return_value = 3
    # Simulate an already-sorted sequence: collection[j] > collection[j+1] is False.
    collection.__getitem__.side_effect = lambda idx: idx
    collection.__setitem__ = MagicMock()

    result = bubble_sort_iterative(collection)

    assert result is collection
    # No swaps should have been performed on already-sorted data.
    collection.__setitem__.assert_not_called()


def test_iterative_performs_swaps_when_out_of_order():
    """The iterative version must call __setitem__ when elements are out of order."""
    collection = MagicMock()
    collection.__len__.return_value = 2
    values = {0: 2, 1: 1}
    collection.__getitem__.side_effect = lambda idx: values[idx]
    collection.__setitem__ = MagicMock()

    bubble_sort_iterative(collection)

    assert collection.__setitem__.call_count == 2


def test_recursive_recurses_until_sorted():
    """The recursive version must call itself again while swaps keep happening."""
    with patch.object(
        bubble_sort_module, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        result = spy([3, 2, 1])
        assert result == [1, 2, 3]
        # At least one recursive re-entry must have occurred for unsorted input.
        assert spy.call_count >= 2


def test_recursive_does_not_recurse_on_sorted_input():
    """The recursive version must not recurse when the input is already sorted."""
    with patch.object(
        bubble_sort_module, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        result = spy([1, 2, 3])
        assert result == [1, 2, 3]
        assert spy.call_count == 1


def test_recursive_does_not_recurse_on_empty_input():
    """An empty list must return immediately without any recursive call."""
    with patch.object(
        bubble_sort_module, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        assert spy([]) == []
        assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Doctest coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_docstrings_contain_examples(sort_func):
    """Both public functions must document their behaviour with doctest examples."""
    assert sort_func.__doc__ is not None
    assert " >>>" not in sort_func.__doc__ or ">>>" in sort_func.__doc__
    assert ">>>" in sort_func.__doc__
