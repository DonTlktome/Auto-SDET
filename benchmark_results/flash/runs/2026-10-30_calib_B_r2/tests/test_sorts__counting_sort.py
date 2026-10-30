"""Tests for the counting sort implementation in sorts/counting_sort.py."""

import random

import pytest

from sorts.counting_sort import counting_sort, counting_sort_string


class TestCountingSort:
    """Tests for the counting_sort function."""

    @pytest.mark.parametrize(
        "collection, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([], []),
            ([-2, -5, -45], [-45, -5, -2]),
            ([1], [1]),
            ([2, 1], [1, 2]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([1, 1, 1, 1], [1, 1, 1, 1]),
            ([-1, 0, 1], [-1, 0, 1]),
            ([100, -100, 0], [-100, 0, 100]),
            ([3, 3, 2, 2, 1, 1], [1, 1, 2, 2, 3, 3]),
        ],
    )
    def test_counting_sort_known_inputs(self, collection, expected):
        """counting_sort returns the collection sorted in ascending order."""
        assert counting_sort(collection) == expected

    def test_counting_sort_empty_list(self):
        """An empty list is returned as an empty list."""
        assert counting_sort([]) == []

    def test_counting_sort_single_element(self):
        """A single-element list is returned unchanged."""
        assert counting_sort([42]) == [42]

    def test_counting_sort_does_not_mutate_input(self):
        """The original collection must not be modified in place."""
        original = [4, 2, 9, 1]
        snapshot = list(original)
        counting_sort(original)
        assert original == snapshot

    def test_counting_sort_returns_new_list(self):
        """The returned object is a distinct list, not the input object."""
        original = [3, 1, 2]
        result = counting_sort(original)
        assert result is not original

    def test_counting_sort_all_negative(self):
        """Lists containing only negative numbers are sorted correctly."""
        assert counting_sort([-3, -1, -2, -10]) == [-10, -3, -2, -1]

    def test_counting_sort_large_range(self):
        """A wide value range (including negatives) is handled correctly."""
        collection = [-1000, 1000, 0, -1, 1]
        assert counting_sort(collection) == [-1000, -1, 0, 1, 1000]

    def test_counting_sort_is_stable(self):
        """Equal values keep their relative order (stability check)."""
        # Use tuples wrapped in a comparable way: sort keys only, so we check
        # stability indirectly by sorting a list of equal values.
        collection = [2, 2, 2, 1, 1]
        assert counting_sort(collection) == [1, 1, 2, 2, 2]

    def test_counting_sort_matches_builtin_sorted_random(self):
        """Randomized comparison against Python's built-in sorted()."""
        rng = random.Random(1234)
        for _ in range(20):
            size = rng.randint(0, 50)
            collection = [rng.randint(-100, 100) for _ in range(size)]
            assert counting_sort(collection) == sorted(collection)

    def test_counting_sort_duplicates_preserved(self):
        """Duplicate values are all preserved in the output."""
        collection = [7, 7, 7, 3, 3, 1]
        result = counting_sort(collection)
        assert result == [1, 3, 3, 7, 7, 7]
        assert len(result) == len(collection)


class TestCountingSortString:
    """Tests for the counting_sort_string function."""

    @pytest.mark.parametrize(
        "string, expected",
        [
            ("thisisthestring", "eghhiiinrsssttt"),
            ("", ""),
            ("a", "a"),
            ("ba", "ab"),
            ("cba", "abc"),
            ("aaa", "aaa"),
            ("zyx", "xyz"),
            ("hello", "ehllo"),
        ],
    )
    def test_counting_sort_string_known_inputs(self, string, expected):
        """counting_sort_string returns the characters sorted ascending."""
        assert counting_sort_string(string) == expected

    def test_counting_sort_string_empty(self):
        """An empty string yields an empty string."""
        assert counting_sort_string("") == ""

    def test_counting_sort_string_preserves_length(self):
        """The sorted string has the same length as the input."""
        text = "mississippi"
        assert len(counting_sort_string(text)) == len(text)

    def test_counting_sort_string_matches_sorted(self):
        """Result matches Python's built-in sorted() joined back together."""
        text = "thequickbrownfox"
        assert counting_sort_string(text) == "".join(sorted(text))

    def test_counting_sort_string_with_spaces(self):
        """Spaces (lowest printable ASCII) are sorted to the front."""
        assert counting_sort_string("b a") == " ab"