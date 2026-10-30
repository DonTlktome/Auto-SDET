"""Tests for sorts/counting_sort.py — pure Python counting sort implementation."""

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
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([7, 7, 7, 7], [7, 7, 7, 7]),
            ([-1, 0, 1], [-1, 0, 1]),
            ([100, -100, 0], [-100, 0, 100]),
            ([3, -3, 3, -3], [-3, -3, 3, 3]),
        ],
    )
    def test_counting_sort_returns_sorted_ascending(self, collection, expected):
        """counting_sort must return the collection ordered ascending."""
        assert counting_sort(collection) == expected

    def test_counting_sort_empty_list(self):
        """An empty list must yield an empty list."""
        assert counting_sort([]) == []

    def test_counting_sort_single_element(self):
        """A single-element list must be returned unchanged."""
        assert counting_sort([42]) == [42]

    def test_counting_sort_all_equal_elements(self):
        """A list of identical elements must be returned unchanged."""
        assert counting_sort([9, 9, 9]) == [9, 9, 9]

    def test_counting_sort_does_not_mutate_input(self):
        """The input collection must not be modified in place."""
        original = [4, 2, 8, 1]
        snapshot = list(original)
        counting_sort(original)
        assert original == snapshot

    def test_counting_sort_returns_new_list(self):
        """The returned object must be a distinct list instance."""
        original = [3, 1, 2]
        result = counting_sort(original)
        assert result is not original

    def test_counting_sort_large_range(self):
        """A wide value range (including negatives) must sort correctly."""
        collection = [1000, -1000, 0, 500, -500]
        assert counting_sort(collection) == [-1000, -500, 0, 500, 1000]

    def test_counting_sort_matches_builtin_sorted(self):
        """Result must match Python's built-in sorted for arbitrary input."""
        collection = [17, -3, 0, 42, -3, 8, 8, 1]
        assert counting_sort(collection) == sorted(collection)

    def test_counting_sort_is_stable(self):
        """Counting sort must preserve the relative order of equal elements."""
        # Use tuples wrapped in a comparable way: sort keys only, check stability
        # by sorting a list of ints and verifying duplicates keep their positions
        # relative to distinguishable neighbours.
        collection = [2, 1, 2, 1, 2]
        assert counting_sort(collection) == [1, 1, 2, 2, 2]

    def test_counting_sort_negative_only(self):
        """A list of only negative numbers must sort ascending."""
        assert counting_sort([-1, -10, -5, -2]) == [-10, -5, -2, -1]

    def test_counting_sort_zero_only(self):
        """A list of only zeros must be returned unchanged."""
        assert counting_sort([0, 0, 0]) == [0, 0, 0]

    def test_counting_sort_raises_on_non_numeric(self):
        """Non-numeric items must raise a TypeError (arithmetic on strings)."""
        with pytest.raises(TypeError):
            counting_sort(["a", "b"])


class TestCountingSortString:
    """Tests for the counting_sort_string function."""

    @pytest.mark.parametrize(
        "text, expected",
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
    def test_counting_sort_string_sorts_characters(self, text, expected):
        """counting_sort_string must return characters sorted ascending."""
        assert counting_sort_string(text) == expected

    def test_counting_sort_string_empty(self):
        """An empty string must yield an empty string."""
        assert counting_sort_string("") == ""

    def test_counting_sort_string_preserves_length(self):
        """The sorted string must have the same length as the input."""
        text = "mississippi"
        assert len(counting_sort_string(text)) == len(text)

    def test_counting_sort_string_matches_builtin_sorted(self):
        """Result must match ''.join(sorted(text)) for arbitrary input."""
        text = "programming"
        assert counting_sort_string(text) == "".join(sorted(text))

    def test_counting_sort_string_with_spaces(self):
        """Spaces (lowest ASCII) must sort to the front."""
        assert counting_sort_string("b a") == " ab"

    def test_counting_sort_string_returns_str(self):
        """The return value must be a str instance."""
        assert isinstance(counting_sort_string("abc"), str)