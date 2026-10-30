import pytest

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive


# ---------------------------------------------------------------------------
# bubble_sort_iterative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unsorted, expected",
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
def test_bubble_sort_iterative_sorts_various_inputs(unsorted, expected):
    """Positive path: iterative bubble sort returns ascending order for
    integers, floats, strings, and mixed numeric types."""
    assert bubble_sort_iterative(unsorted) == expected


def test_bubble_sort_iterative_empty_list():
    """Edge case: sorting an empty list returns an empty list."""
    assert bubble_sort_iterative([]) == []


def test_bubble_sort_iterative_single_element():
    """Edge case: a single-element list is returned unchanged."""
    assert bubble_sort_iterative([42]) == [42]


def test_bubble_sort_iterative_does_not_mutate_original():
    """The function sorts in place but returns the same list; verify the
    returned list equals the sorted original and is the same object."""
    original = [5, 1, 4, 2, 8]
    result = bubble_sort_iterative(original)
    assert result == sorted(original)
    assert result is original


def test_bubble_sort_iterative_already_sorted_short_circuits():
    """Edge case: an already-sorted list triggers the early-exit optimization
    and still returns the correct sorted order."""
    assert bubble_sort_iterative([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# bubble_sort_recursive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unsorted, expected",
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
        (["a", "Z", "B", "C", "A", "c"], ["A", "B", "C", "Z", "a", "c"]),
    ],
)
def test_bubble_sort_recursive_sorts_various_inputs(unsorted, expected):
    """Positive path: recursive bubble sort returns ascending order for
    integers, floats, strings, and mixed numeric types."""
    assert bubble_sort_recursive(unsorted) == expected


def test_bubble_sort_recursive_empty_list():
    """Edge case: sorting an empty list returns an empty list."""
    assert bubble_sort_recursive([]) == []


def test_bubble_sort_recursive_single_element():
    """Edge case: a single-element list is returned unchanged."""
    assert bubble_sort_recursive([42]) == [42]


def test_bubble_sort_recursive_does_not_mutate_original():
    """The recursive function sorts in place and returns the same list object."""
    original = [5, 1, 4, 2, 8]
    result = bubble_sort_recursive(original)
    assert result == sorted(original)
    assert result is original


def test_bubble_sort_recursive_already_sorted():
    """Edge case: an already-sorted list returns immediately without swaps."""
    assert bubble_sort_recursive([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_bubble_sort_recursive_reverse_sorted():
    """Edge case: a reverse-sorted list exercises the maximum number of
    recursive passes and still produces ascending order."""
    assert bubble_sort_recursive([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]