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
def test_sorts_unsorted_integers(sort_func):
    """A typical unsorted list of ints is returned in ascending order."""
    assert sort_func([0, 5, 2, 3, 2]) == [0, 2, 2, 3, 5]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_negative_integers(sort_func):
    """Negative numbers are ordered correctly (ascending)."""
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
    """Mixed int/float values are comparable and sorted together."""
    assert sort_func([1, 3.3, 5, 7.7, 2, 4.4, 6]) == [1, 2, 3.3, 4.4, 5, 6, 7.7]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_sorts_mixed_case_strings(sort_func):
    """Uppercase letters sort before lowercase (ASCII ordering)."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object_in_place(sort_func):
    """The function sorts in place and returns the very same list object."""
    data = [3, 1, 2]
    result = sort_func(data)
    assert result is data
    assert data == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_data(sort_func):
    """Randomized data produces the same result as the builtin sorted()."""
    rng = random.Random(1234)
    data = rng.sample(range(-50, 50), 100)
    assert sort_func(list(data)) == sorted(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Randomized alphanumeric data produces the same result as sorted()."""
    rng = random.Random(4321)
    data = rng.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Edge cases / boundaries
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
def test_all_negative_values(sort_func):
    """A list of only negative values is sorted ascending."""
    assert sort_func([-2, -45, -5]) == [-45, -5, -2]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_duplicates_preserved(sort_func):
    """Duplicate values are all preserved (no elements lost)."""
    result = sort_func([2, 1, 2, 1, 2])
    assert result == [1, 1, 2, 2, 2]
    assert len(result) == 5


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_large_list_matches_sorted(sort_func):
    """A larger list (200 elements) still matches the builtin sorted()."""
    rng = random.Random(99)
    data = [rng.randint(-1000, 1000) for _ in range(200)]
    assert sort_func(list(data)) == sorted(data)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_does_not_mutate_input_length(sort_func):
    """Sorting never changes the number of elements."""
    data = [4, 2, 9, 1]
    sort_func(data)
    assert len(data) == 4


# ---------------------------------------------------------------------------
# Isolation / mocking tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uses_only_comparison_operator(sort_func):
    """Sorting relies on `>` comparisons; a mock collection verifies this.

    The collection is a MagicMock-backed sequence whose items are MagicMock
    objects with a controlled `__gt__`. This isolates the algorithm from any
    real data and confirms it performs comparisons rather than, e.g., calling
    the builtin `sorted`.

    The mock keeps a real backing list so that assignments (swaps) actually
    change the order — otherwise an in-place sort would never converge.
    """
    items = [MagicMock(name=f"item{i}") for i in range(3)]
    # Define a total order: item0 > item1 > item2
    for i, item in enumerate(items):
        item.__gt__ = MagicMock(side_effect=lambda other, i=i: items.index(other) > i)

    backing = list(items)

    def _getitem(index):
        return backing[index]

    def _setitem(index, value):
        backing[index] = value

    collection = MagicMock()
    collection.__len__ = MagicMock(return_value=3)
    collection.__getitem__ = MagicMock(side_effect=_getitem)
    collection.__setitem__ = MagicMock(side_effect=_setitem)

    with patch("builtins.sorted") as mock_sorted:
        sort_func(collection)
        mock_sorted.assert_not_called()

    assert collection.__getitem__.called
    assert collection.__setitem__.called
    # item0 > item1 > item2, so ascending order must be [item2, item1, item0].
    assert backing == [items[2], items[1], items[0]]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_no_swaps_when_already_sorted(sort_func):
    """An already sorted list triggers no item assignment (no swaps)."""
    collection = MagicMock()
    collection.__len__ = MagicMock(return_value=3)
    collection.__getitem__ = MagicMock(side_effect=lambda idx: [1, 2, 3][idx])
    collection.__setitem__ = MagicMock()

    sort_func(collection)

    collection.__setitem__.assert_not_called()


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early: comparisons are bounded by n-1 passes.

    For an already sorted list of length n, the inner loop runs n-1 times on
    the first pass (each comparison reads two items, hence 2 * (n - 1)
    `__getitem__` calls) and then breaks, so no further passes happen.
    """
    n = 5
    collection = MagicMock()
    collection.__len__ = MagicMock(return_value=n)
    collection.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    collection.__setitem__ = MagicMock()

    bubble_sort_iterative(collection)

    # n-1 comparisons on the first pass (2 reads each), then early break.
    assert collection.__getitem__.call_count == 2 * (n - 1)
    collection.__setitem__.assert_not_called()


def test_recursive_terminates_on_sorted_input():
    """The recursive version returns immediately for an already sorted list."""
    data = [1, 2, 3, 4, 5]
    assert bubble_sort_recursive(data) == [1, 2, 3, 4, 5]


def test_recursive_handles_empty_without_recursion():
    """The recursive version handles the empty list without recursing."""
    module = sys.modules["bubble_sort"]
    with patch.object(
        module, "bubble_sort_recursive", wraps=bubble_sort_recursive
    ) as spy:
        # Look the function up through the module so the patch is honoured.
        assert module.bubble_sort_recursive([]) == []
        assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Error / type-mismatch behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_incomparable_types_raise_type_error(sort_func):
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_none_input_raises_type_error(sort_func):
    """Passing None raises TypeError because len(None) is invalid."""
    with pytest.raises(TypeError):
        sort_func(None)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_tuple_input_raises_type_error(sort_func):
    """A tuple is immutable, so in-place swapping raises TypeError."""
    with pytest.raises(TypeError):
        sort_func((3, 1, 2))
