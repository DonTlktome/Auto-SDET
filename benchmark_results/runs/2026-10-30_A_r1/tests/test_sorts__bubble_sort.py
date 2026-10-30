import pytest

from bubble_sort import bubble_sort_iterative, bubble_sort_recursive


# ---------------------------------------------------------------------------
# bubble_sort_iterative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([], []),
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
def test_bubble_sort_iterative_happy_path(input_list, expected):
    """Test that bubble_sort_iterative sorts various inputs in ascending order."""
    assert bubble_sort_iterative(input_list) == expected


def test_bubble_sort_iterative_does_not_mutate_original_reference():
    """Test that the function returns the same list object (in-place sort)."""
    collection = [3, 1, 2]
    result = bubble_sort_iterative(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_bubble_sort_iterative_already_sorted_short_circuits():
    """Test that an already-sorted list is returned unchanged (early break)."""
    collection = [1, 2, 3, 4, 5]
    assert bubble_sort_iterative(collection) == [1, 2, 3, 4, 5]


def test_bubble_sort_iterative_reverse_sorted():
    """Test worst-case input (fully reversed) is correctly sorted."""
    collection = [5, 4, 3, 2, 1]
    assert bubble_sort_iterative(collection) == [1, 2, 3, 4, 5]


def test_bubble_sort_iterative_single_element():
    """Test boundary case: a single-element list is returned unchanged."""
    assert bubble_sort_iterative([42]) == [42]


def test_bubble_sort_iterative_empty_list():
    """Test boundary case: an empty list is returned unchanged."""
    assert bubble_sort_iterative([]) == []


def test_bubble_sort_iterative_duplicates():
    """Test that duplicate values are preserved and sorted correctly."""
    assert bubble_sort_iterative([2, 2, 1, 1, 3, 3]) == [1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# bubble_sort_recursive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([0, 5, 2, 3, 2], [0, 2, 2, 3, 5]),
        ([], []),
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
def test_bubble_sort_recursive_happy_path(input_list, expected):
    """Test that bubble_sort_recursive sorts various inputs in ascending order."""
    assert bubble_sort_recursive(input_list) == expected


def test_bubble_sort_recursive_does_not_mutate_original_reference():
    """Test that the function returns the same list object (in-place sort)."""
    collection = [3, 1, 2]
    result = bubble_sort_recursive(collection)
    assert result is collection
    assert collection == [1, 2, 3]


def test_bubble_sort_recursive_already_sorted():
    """Test that an already-sorted list is returned unchanged (base case)."""
    collection = [1, 2, 3, 4, 5]
    assert bubble_sort_recursive(collection) == [1, 2, 3, 4, 5]


def test_bubble_sort_recursive_reverse_sorted():
    """Test worst-case input (fully reversed) is correctly sorted."""
    collection = [5, 4, 3, 2, 1]
    assert bubble_sort_recursive(collection) == [1, 2, 3, 4, 5]


def test_bubble_sort_recursive_single_element():
    """Test boundary case: a single-element list is returned unchanged."""
    assert bubble_sort_recursive([42]) == [42]


def test_bubble_sort_recursive_empty_list():
    """Test boundary case: an empty list is returned unchanged."""
    assert bubble_sort_recursive([]) == []


def test_bubble_sort_recursive_duplicates():
    """Test that duplicate values are preserved and sorted correctly."""
    assert bubble_sort_recursive([2, 2, 1, 1, 3, 3]) == [1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# Cross-function consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_list",
    [
        [0, 5, 2, 3, 2],
        [],
        [-2, -45, -5],
        [5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [3, 3, 3, 3],
        [56],
        ["d", "a", "b", "e"],
        [1.1, 3.3, 5.5, 7.7, 2.2, 4.4, 6.6],
    ],
)
def test_iterative_and_recursive_agree(input_list):
    """Test that both implementations produce identical results."""
    assert bubble_sort_iterative(input_list[:]) == bubble_sort_recursive(input_list[:])