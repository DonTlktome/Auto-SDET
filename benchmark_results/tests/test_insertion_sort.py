"""
Pytest test suite for insertion_sort.py

Covers the public function `insertion_sort` with:
- Positive (happy-path) cases
- Edge cases (empty, single-element, already-sorted, reverse-sorted, duplicates)
- Type-varied inputs (int, float, str)
- MutableSequence compliance (bytearray, list)
- Parametrized random-like deterministic tests
"""

import math
from collections.abc import MutableSequence
from typing import Any

import pytest

from insertion_sort import insertion_sort


# ---------------------------------------------------------------------------
# Helper: a minimal mutable sequence stub for protocol verification
# ---------------------------------------------------------------------------
class _FakeMutableSequence(MutableSequence):
    """A list-like object that delegates to an internal list, useful to
    verify that insertion_sort works with any MutableSequence, not just list."""

    def __init__(self, iterable=()):
        self._data = list(iterable)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __len__(self):
        return len(self._data)

    def insert(self, index, value):
        self._data.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, _FakeMutableSequence):
            return self._data == other._data
        return self._data == other

    def __repr__(self):
        return f"_FakeMutableSequence({self._data!r})"


# ============================================================================
# Positive / happy-path tests
# ============================================================================

class TestInsertionSortHappyPath:
    """Tests where the function should succeed with typical inputs."""

    @pytest.mark.parametrize(
        "input_data, expected",
        [
            ([0, 5, 3, 2, 2], [0, 2, 2, 3, 5]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
            ([3, 1, 2], [1, 2, 3]),
            ([42], [42]),
            ([], []),
        ],
    )
    def test_sort_ints(self, input_data, expected):
        """Sort lists of integers, including already-sorted, reverse-sorted,
        single-element, and empty."""
        result = insertion_sort(input_data)
        assert result == expected
        # Also verify it returns the same object (mutation, not a copy)
        assert result is input_data

    @pytest.mark.parametrize(
        "input_data, expected",
        [
            ([3.3, 1.1, 2.2], [1.1, 2.2, 3.3]),
            ([0.0, -0.0, 1.0], [-0.0, 0.0, 1.0]),
            ([math.pi, math.e, 0.0], [0.0, math.e, math.pi]),
        ],
    )
    def test_sort_floats(self, input_data, expected):
        """Sort lists of floats."""
        result = insertion_sort(input_data)
        assert result == expected
        assert result is input_data

    @pytest.mark.parametrize(
        "input_data, expected",
        [
            (["d", "a", "b", "e", "c"], ["a", "b", "c", "d", "e"]),
            (["z", "a"], ["a", "z"]),
            (["hello", "world", "abc"], ["abc", "hello", "world"]),
        ],
    )
    def test_sort_strings(self, input_data, expected):
        """Sort lists of strings lexicographically."""
        result = insertion_sort(input_data)
        assert result == expected
        assert result is input_data

    def test_sort_bytearray(self):
        """bytearray is a MutableSequence[int]; ensure it works."""
        data = bytearray([5, 2, 8, 1])
        result = insertion_sort(data)
        assert result == bytearray([1, 2, 5, 8])
        assert result is data

    def test_sort_custom_mutable_sequence(self):
        """Any MutableSequence implementation should be accepted."""
        data = _FakeMutableSequence([9, 3, 6, 1])
        result = insertion_sort(data)
        assert result == [1, 3, 6, 9]
        assert result is data


# ============================================================================
# Edge-case / boundary tests
# ============================================================================

class TestInsertionSortEdgeCases:
    """Tests that probe boundaries and unusual but valid inputs."""

    def test_empty_list_returns_empty(self):
        """Empty collection should be returned unchanged."""
        data = []
        result = insertion_sort(data)
        assert result == []
        assert result is data

    def test_single_element(self):
        """Single-element list is trivially sorted."""
        data = [99]
        result = insertion_sort(data)
        assert result == [99]
        assert result is data

    def test_all_equal_elements(self):
        """List where every element compares equal should remain stable-ish."""
        data = [7, 7, 7, 7]
        result = insertion_sort(data)
        assert result == [7, 7, 7, 7]
        assert result is data

    def test_large_already_sorted(self):
        """Stress-test with a large already-sorted range (O(n) best case)."""
        n = 500
        data = list(range(n))
        result = insertion_sort(data)
        assert result == list(range(n))
        assert result is data

    def test_large_reverse_sorted(self):
        """Stress-test with a large reverse-sorted range (O(n^2) worst case)."""
        n = 200
        data = list(range(n, 0, -1))
        result = insertion_sort(data)
        assert result == list(range(1, n + 1))
        assert result is data

    def test_negative_numbers(self):
        """Negative integers should sort correctly."""
        data = [-2, -5, -45, 0, 3]
        result = insertion_sort(data)
        assert result == [-45, -5, -2, 0, 3]
        assert result is data

    def test_mixed_types_not_supported(self):
        """Insertion sort relies on __lt__; mixed incomparable types raise TypeError."""
        data = [1, "a", 2]
        with pytest.raises(TypeError):
            insertion_sort(data)


# ============================================================================
# Deterministic "random-like" parametrized tests
# ============================================================================

class TestInsertionSortParametrized:
    """Parametrized tests that act like the doctest random examples but are
    deterministic (no actual randomness)."""

    @pytest.mark.parametrize(
        "unsorted",
        [
            list(range(-50, 50)),
            list(range(50, -50, -1)),
            [7, 3, 9, 1, 0, 4, 8, 2, 5, 6],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ],
    )
    def test_against_builtin_sorted(self, unsorted):
        """For various deterministic sequences, insertion_sort must match sorted()."""
        result = insertion_sort(unsorted)
        assert result == sorted(unsorted)
        assert result is unsorted

    @pytest.mark.parametrize(
        "chars",
        [
            list("abcdefghij"),
            list("zyxwvutsrqponmlkjihgfedcba"),
            list("a1b2c3d4e5"),
        ],
    )
    def test_string_characters_against_sorted(self, chars):
        """Character lists sorted via insertion_sort must match sorted()."""
        result = insertion_sort(chars)
        assert result == sorted(chars)
        assert result is chars