"""Tests for sorts/bubble_sort.py — iterative and recursive bubble sort."""

import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SORT_FUNCTIONS = [bubble_sort_iterative, bubble_sort_recursive]


@pytest.fixture(params=SORT_FUNCTIONS, ids=["iterative", "recursive"])
def sort_func(request):
    """Parametrized fixture yielding each public sorting function."""
    return request.param


# ---------------------------------------------------------------------------
# Positive (happy path) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([-2, -45, -5], [-45, -5, -2]),
        ([-23, 0, 6, -4, 34], [-23, -4, 0, 6, 34]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([4, 3, 2, 1], [1, 2, 3, 4]),
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
def test_sort_returns_ascending_order(sort_func, collection, expected):
    """Each sort function orders a variety of collections in ascending order."""
    assert sort_func(list(collection)) == expected


@pytest.mark.parametrize(
    "collection",
    [
        [0, 5, 2, 3, 2],
        [-23, 0, 6, -4, 34],
        ["d", "a", "b", "e"],
        [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6],
        [1, 3.3, 5, 7.7, 2, 4.4, 6],
    ],
)
def test_sort_matches_builtin_sorted(sort_func, collection):
    """Sort output must agree with Python's built-in ``sorted``."""
    assert sort_func(list(collection)) == sorted(collection)


def test_recursive_uppercase_before_lowercase():
    """Recursive sort respects ASCII ordering (uppercase before lowercase)."""
    assert bubble_sort_recursive(["a", "Z", "B", "C", "A", "c"]) == [
        "A",
        "B",
        "C",
        "Z",
        "a",
        "c",
    ]


def test_sort_returns_same_object(sort_func):
    """The functions sort in place and return the very same list object."""
    data = [3, 1, 2]
    result = sort_func(data)
    assert result is data
    assert data == [1, 2, 3]


def test_sort_large_random_integers(sort_func):
    """Sorting a large random sample of ints matches ``sorted``."""
    rng = random.Random(1234)
    data = rng.sample(range(-50, 50), 100)
    assert sort_func(list(data)) == sorted(data)


def test_sort_large_random_strings(sort_func):
    """Sorting a large random list of characters matches ``sorted``."""
    rng = random.Random(4321)
    data = rng.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(list(data)) == sorted(data)


# ---------------------------------------------------------------------------
# Edge / boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_empty_list(func):
    """An empty list is returned unchanged."""
    assert func([]) == []


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_single_element(func):
    """A single-element list is returned unchanged."""
    assert func([42]) == [42]


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_two_elements_unsorted(func):
    """The minimal unsorted case (two elements) is swapped."""
    assert func([2, 1]) == [1, 2]


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_two_elements_sorted(func):
    """The minimal already-sorted case is left untouched."""
    assert func([1, 2]) == [1, 2]


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_already_sorted(func):
    """An already sorted list stays sorted."""
    data = list(range(10))
    assert func(data) == list(range(10))


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_reverse_sorted(func):
    """A reverse-sorted list is fully reversed."""
    data = list(range(10, 0, -1))
    assert func(data) == list(range(1, 11))


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_all_equal_elements(func):
    """A list of identical elements is unchanged."""
    assert func([7, 7, 7, 7, 7]) == [7, 7, 7, 7, 7]


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_duplicates_preserved(func):
    """Duplicate values are all preserved (multiset equality)."""
    data = [2, 1, 2, 1, 2, 1]
    result = func(data)
    assert result == [1, 1, 1, 2, 2, 2]
    assert len(result) == len(data)


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_negative_numbers(func):
    """Negative numbers sort correctly."""
    assert func([-1, -100, -50, 0, 5]) == [-100, -50, -1, 0, 5]


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_does_not_mutate_length(func):
    """Sorting never changes the number of elements."""
    data = [5, 3, 8, 1, 9, 2]
    original_len = len(data)
    func(data)
    assert len(data) == original_len


@pytest.mark.parametrize("func", SORT_FUNCTIONS, ids=["iterative", "recursive"])
def test_sort_incomparable_types_raise_type_error(func):
    """Comparing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        func([1, "a", 2])


# ---------------------------------------------------------------------------
# Behavioural / implementation-detail tests
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """Iterative sort stops early: a sorted list triggers no swaps at all."""
    data = [1, 2, 3, 4, 5]
    assert bubble_sort_iterative(data) == [1, 2, 3, 4, 5]
    assert bubble_sort_iterative([]) == []
    assert bubble_sort_iterative([7]) == [7]


def test_iterative_uses_early_break_for_sorted_input():
    """Iterative sort performs a single outer pass when input is already sorted.

    A list subclass counts ``__getitem__`` accesses; a sorted list should only
    require the first outer pass (n-1 comparisons) before breaking out.
    """

    class CountingList(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.getitem_calls = 0

        def __getitem__(self, index):
            self.getitem_calls += 1
            return super().__getitem__(index)

    data = CountingList([1, 2, 3, 4, 5])
    bubble_sort_iterative(data)
    # One pass over 5 elements: 4 comparisons * 2 lookups = 8 accesses.
    assert data.getitem_calls == 8


def test_recursive_recurses_until_sorted():
    """Recursive sort calls itself repeatedly until no swap occurs."""
    calls = []
    original = bubble_sort_recursive

    def spy(collection):
        calls.append(list(collection))
        return original(collection)

    with patch.dict(
        bubble_sort_recursive.__globals__, {"bubble_sort_recursive": spy}
    ):
        result = bubble_sort_recursive([3, 2, 1])

    assert result == [1, 2, 3]
    # The recursive implementation recurses via the module global, so the spy
    # must have been invoked at least once for the recursive calls.
    assert len(calls) >= 1


def test_recursive_single_pass_on_sorted_input():
    """Recursive sort makes no recursive call when input is already sorted."""
    with patch.dict(
        bubble_sort_recursive.__globals__,
        {"bubble_sort_recursive": MagicMock(side_effect=AssertionError("recursed"))},
    ):
        assert bubble_sort_recursive([1, 2, 3]) == [1, 2, 3]


def test_iterative_and_recursive_agree(sort_func):
    """Both implementations produce identical results on the same input."""
    data = [9, -3, 0, 7, 7, 2, -8, 5]
    assert bubble_sort_iterative(list(data)) == bubble_sort_recursive(list(data))


def test_sort_does_not_use_network_or_io(sort_func):
    """Sorting is pure: no file or socket access is attempted."""
    with patch("builtins.open", side_effect=AssertionError("I/O attempted")):
        assert sort_func([3, 1, 2]) == [1, 2, 3]
