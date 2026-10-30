"""Tests for the counting sort implementation in sorts/counting_sort.py."""

import random
import string

import pytest

from sorts.counting_sort import counting_sort, counting_sort_string


# ---------------------------------------------------------------------------
# counting_sort
# ---------------------------------------------------------------------------


def test_counting_sort_doctest_example():
    """The canonical example from the docstring must sort ascending."""
    assert counting_sort([0, 5, 3, 2, 2]) == [0, 2, 2, 3, 5]


def test_counting_sort_empty_list():
    """An empty collection must return an empty list."""
    assert counting_sort([]) == []


def test_counting_sort_single_element():
    """A one-element collection is already sorted."""
    assert counting_sort([42]) == [42]


def test_counting_sort_negative_numbers():
    """Negative values must be handled via the min-offset logic."""
    assert counting_sort([-2, -5, -45]) == [-45, -5, -2]


def test_counting_sort_mixed_signs():
    """A mix of negative, zero and positive values must sort correctly."""
    assert counting_sort([3, -1, 0, -7, 5, 2]) == [-7, -1, 0, 2, 3, 5]


def test_counting_sort_all_equal():
    """A collection of identical values must be returned unchanged."""
    assert counting_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_counting_sort_already_sorted():
    """An already sorted collection must remain sorted."""
    data = [1, 2, 3, 4, 5]
    assert counting_sort(data) == [1, 2, 3, 4, 5]


def test_counting_sort_reverse_sorted():
    """A reverse sorted collection must be reversed into ascending order."""
    assert counting_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_counting_sort_does_not_mutate_input():
    """The input collection must not be modified in place."""
    data = [4, 1, 3, 2]
    original = list(data)
    counting_sort(data)
    assert data == original


def test_counting_sort_returns_new_list():
    """The returned object must be a distinct list instance."""
    data = [3, 1, 2]
    result = counting_sort(data)
    assert result is not data


def test_counting_sort_large_range():
    """A wide value range (sparse counting array) must still sort correctly."""
    data = [-1000, 1000, 0, 500, -500]
    assert counting_sort(data) == [-1000, -500, 0, 500, 1000]


def test_counting_sort_large_input_matches_builtin():
    """A large random input must match Python's built-in sorted()."""
    rng = random.Random(1234)
    data = [rng.randint(-500, 500) for _ in range(1000)]
    assert counting_sort(data) == sorted(data)


def test_counting_sort_stability_with_tuples_of_ints():
    """Sorting must be stable: equal keys keep their relative order.

    The implementation sorts plain integers, so stability is verified by
    checking that the multiset of values is preserved exactly.
    """
    data = [2, 1, 2, 1, 2, 1]
    result = counting_sort(data)
    assert result == [1, 1, 1, 2, 2, 2]
    assert sorted(result) == sorted(data)


@pytest.mark.parametrize(
    "collection, expected",
    [
        ([], []),
        ([0], [0]),
        ([1, 0], [0, 1]),
        ([0, 0, 0], [0, 0, 0]),
        ([-1, -1, -2], [-2, -1, -1]),
        ([10, 9, 8, 7, 6, 5], [5, 6, 7, 8, 9, 10]),
        ([100, -100], [-100, 100]),
        ([3, 3, 1, 1, 2, 2], [1, 1, 2, 2, 3, 3]),
    ],
)
def test_counting_sort_parametrized(collection, expected):
    """Various collections must be sorted into the expected ascending order."""
    assert counting_sort(collection) == expected


@pytest.mark.parametrize("size", [0, 1, 2, 5, 50, 200])
def test_counting_sort_various_sizes(size):
    """Collections of many different sizes must match sorted()."""
    rng = random.Random(size)
    data = [rng.randint(-20, 20) for _ in range(size)]
    assert counting_sort(data) == sorted(data)


def test_counting_sort_raises_on_non_numeric():
    """Non-numeric items must raise a TypeError (arithmetic on strings)."""
    with pytest.raises(TypeError):
        counting_sort(["a", "b"])


# ---------------------------------------------------------------------------
# counting_sort_string
# ---------------------------------------------------------------------------


def test_counting_sort_string_doctest_example():
    """The canonical docstring example must produce the sorted string."""
    assert counting_sort_string("thisisthestring") == "eghhiiinrsssttt"


def test_counting_sort_string_empty():
    """An empty string must return an empty string."""
    assert counting_sort_string("") == ""


def test_counting_sort_string_single_char():
    """A single character string is already sorted."""
    assert counting_sort_string("a") == "a"


def test_counting_sort_string_already_sorted():
    """An already sorted string must be returned unchanged."""
    assert counting_sort_string("abc") == "abc"


def test_counting_sort_string_reverse_sorted():
    """A reverse sorted string must be reversed into ascending order."""
    assert counting_sort_string("zyx") == "xyz"


def test_counting_sort_string_with_spaces_and_punctuation():
    """Characters are ordered by their ordinal value, spaces first."""
    assert counting_sort_string("b a") == " ab"


def test_counting_sort_string_matches_builtin_sorted():
    """A random lowercase string must match ''.join(sorted(...))."""
    rng = random.Random(99)
    text = "".join(rng.choice(string.ascii_lowercase) for _ in range(100))
    assert counting_sort_string(text) == "".join(sorted(text))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("a", "a"),
        ("ba", "ab"),
        ("cba", "abc"),
        ("aaa", "aaa"),
        ("hello", "ehllo"),
        ("dcba", "abcd"),
    ],
)
def test_counting_sort_string_parametrized(text, expected):
    """Various strings must be sorted character by character."""
    assert counting_sort_string(text) == expected


def test_counting_sort_string_preserves_length():
    """Sorting a string must not change its length."""
    text = "mississippi"
    assert len(counting_sort_string(text)) == len(text)