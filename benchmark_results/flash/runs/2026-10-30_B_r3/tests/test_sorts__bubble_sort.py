import random
import string
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the target module importable regardless of the test runner's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        ([1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]),
        ([1, 3.3, 5, 7.7, 2, 4.4, 6], [1, 2, 3.3, 4.4, 5, 6, 7.7]),
    ],
)
def test_sorts_known_collections(sort_func, collection, expected):
    """Each sort function orders a variety of known collections ascending."""
    assert sort_func(collection) == expected


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_ints(sort_func):
    """Sorting a random sample of ints matches Python's built-in sorted()."""
    collection = random.sample(range(-50, 50), 100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_matches_builtin_sorted_on_random_strings(sort_func):
    """Sorting random alphanumeric characters matches Python's sorted()."""
    collection = random.choices(string.ascii_letters + string.digits, k=100)
    assert sort_func(collection) == sorted(collection)


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_returns_same_object_in_place(sort_func):
    """The functions sort in place and return the very same list object."""
    collection = [4, 1, 3, 2]
    result = sort_func(collection)
    assert result is collection
    assert collection == [1, 2, 3, 4]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_does_not_change_length(sort_func):
    """Sorting preserves the number of elements (no duplicates dropped)."""
    collection = [5, 1, 5, 1, 5, 1]
    assert len(sort_func(collection)) == 6


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
def test_reverse_sorted_list(sort_func):
    """A fully descending list is reversed into ascending order."""
    collection = list(range(10, 0, -1))
    assert sort_func(collection) == list(range(1, 11))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_already_sorted_list(sort_func):
    """An already sorted list is left untouched."""
    collection = list(range(20))
    assert sort_func(collection) == list(range(20))


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_all_equal_elements(sort_func):
    """A list of identical values is returned unchanged."""
    assert sort_func([7, 7, 7, 7, 7]) == [7, 7, 7, 7, 7]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_negative_numbers_only(sort_func):
    """Lists containing only negative numbers sort correctly."""
    assert sort_func([-1, -100, -50, -2]) == [-100, -50, -2, -1]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_large_values(sort_func):
    """Very large integers are handled without overflow issues."""
    collection = [10**18, -(10**18), 0, 10**9]
    assert sort_func(collection) == [-(10**18), 0, 10**9, 10**18]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uppercase_before_lowercase(sort_func):
    """ASCII ordering places uppercase letters before lowercase ones."""
    assert sort_func(["a", "Z", "B", "C", "A", "c"]) == ["A", "B", "C", "Z", "a", "c"]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_booleans_sort_like_ints(sort_func):
    """Booleans compare as ints, so False sorts before True."""
    assert sort_func([True, False, True, False]) == [False, False, True, True]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_incomparable_types_raise_type_error(sort_func):
    """Mixing incomparable types (int vs str) raises TypeError."""
    with pytest.raises(TypeError):
        sort_func([1, "a", 2])


# ---------------------------------------------------------------------------
# Behavioural details specific to each implementation
# ---------------------------------------------------------------------------


def test_iterative_early_exit_on_sorted_input():
    """The iterative version stops early when no swap occurs in a pass.

    A custom list subclass counts element comparisons; a sorted input should
    require only a single pass (n - 1 comparisons) thanks to the early break.
    """
    comparisons = {"count": 0}

    class CountingList(list):
        def __getitem__(self, index):
            comparisons["count"] += 1
            return super().__getitem__(index)

    data = CountingList([1, 2, 3, 4, 5])
    bubble_sort_iterative(data)
    # One pass over 5 elements => 4 comparisons (each comparison reads 2 items,
    # but the early break prevents further passes).
    assert comparisons["count"] <= 2 * (len(data) - 1)


def test_recursive_terminates_on_sorted_input():
    """The recursive version returns immediately for already sorted input."""
    data = [1, 2, 3]
    assert bubble_sort_recursive(data) == [1, 2, 3]


def test_recursive_handles_moderately_large_input():
    """The recursive version completes on a moderately large reversed list."""
    collection = list(range(200, 0, -1))
    assert bubble_sort_recursive(collection) == list(range(1, 201))


def test_iterative_handles_moderately_large_input():
    """The iterative version completes on a moderately large reversed list."""
    collection = list(range(200, 0, -1))
    assert bubble_sort_iterative(collection) == list(range(1, 201))


# ---------------------------------------------------------------------------
# Isolation / mocking tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_uses_only_len_and_getitem_contract(sort_func):
    """Sorting works with a MagicMock-wrapped list-like object.

    This verifies the implementations rely only on ``len`` and indexing,
    not on any external I/O or network resources.
    """
    backing = [3, 1, 2]
    proxy = MagicMock()
    proxy.__len__.return_value = len(backing)
    proxy.__getitem__.side_effect = lambda index: backing[index]
    proxy.__setitem__.side_effect = lambda index, value: backing.__setitem__(
        index, value
    )

    result = sort_func(proxy)

    assert result is proxy
    assert backing == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_no_network_or_file_access(sort_func):
    """Sorting performs no socket or file-system access (fully isolated)."""
    with patch("socket.socket", side_effect=AssertionError("network used")), patch(
        "builtins.open", side_effect=AssertionError("file access used")
    ):
        assert sort_func([3, 2, 1]) == [1, 2, 3]


@pytest.mark.parametrize("sort_func", SORT_FUNCTIONS)
def test_deterministic_across_repeated_calls(sort_func):
    """Repeated calls on fresh copies of the same data give identical results."""
    base = [9, 4, 7, 1, 8, 2, 6, 3, 5, 0]
    first = sort_func(base[:])
    second = sort_func(base[:])
    assert first == second == sorted(base)
