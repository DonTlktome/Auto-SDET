"""
Pytest test suite for the linear_search module.

Covers:
- linear_search (iterative)
- rec_linear_search (recursive)

All tests are self-contained and require no network or I/O.
"""

import pytest

from linear_search import linear_search, rec_linear_search


# ---------------------------------------------------------------------------
# linear_search
# ---------------------------------------------------------------------------

class TestLinearSearch:
    """Tests for the iterative linear_search function."""

    # -- Positive / happy-path ------------------------------------------------

    @pytest.mark.parametrize(
        "sequence, target, expected",
        [
            ([0, 5, 7, 10, 15], 0, 0),    # first element
            ([0, 5, 7, 10, 15], 15, 4),   # last element
            ([0, 5, 7, 10, 15], 5, 1),    # middle element
            ([42], 42, 0),                 # single-element list, found
            ([1, 2, 3, 2, 4], 2, 1),      # duplicate — returns first occurrence
        ],
    )
    def test_found(self, sequence, target, expected):
        """Target is present; should return its first index."""
        assert linear_search(sequence, target) == expected

    @pytest.mark.parametrize(
        "sequence, target",
        [
            ([0, 5, 7, 10, 15], 6),   # missing in non-empty list
            ([], 1),                   # empty list
            ([42], 99),                # single-element list, not found
        ],
    )
    def test_not_found(self, sequence, target):
        """Target is absent; should return -1."""
        assert linear_search(sequence, target) == -1

    # -- Edge cases -----------------------------------------------------------

    def test_large_list_found(self):
        """Large list: target at the very end, still found correctly."""
        n = 10_000
        seq = list(range(n))
        assert linear_search(seq, n - 1) == n - 1

    def test_large_list_not_found(self):
        """Large list: target missing, returns -1."""
        seq = list(range(10_000))
        assert linear_search(seq, -100) == -1

    def test_mixed_types(self):
        """Sequence with mixed types; linear search compares using == so
        no TypeError is raised, and if target is present it is found."""
        assert linear_search([1, "two", 3], "two") == 1


# ---------------------------------------------------------------------------
# rec_linear_search
# ---------------------------------------------------------------------------

class TestRecLinearSearch:
    """Tests for the recursive linear_search function."""

    # -- Positive / happy-path ------------------------------------------------

    @pytest.mark.parametrize(
        "sequence, low, high, target, expected",
        [
            ([0, 30, 500, 100, 700], 0, 4, 0, 0),     # first element
            ([0, 30, 500, 100, 700], 0, 4, 700, 4),   # last element
            ([0, 30, 500, 100, 700], 0, 4, 30, 1),    # middle element
            ([42], 0, 0, 42, 0),                       # single element, found
            ([1, 2, 3, 2, 4], 0, 4, 2, 1),            # duplicate — first occurrence
        ],
    )
    def test_found(self, sequence, low, high, target, expected):
        """Target is present within [low, high]; returns correct index."""
        assert rec_linear_search(sequence, low, high, target) == expected

    @pytest.mark.parametrize(
        "sequence, low, high, target",
        [
            ([0, 30, 500, 100, 700], 0, 4, -6),   # missing in non-empty
            ([1, 2, 3], 0, 2, 99),                 # missing
        ],
    )
    def test_not_found(self, sequence, low, high, target):
        """Target is absent; should return -1."""
        assert rec_linear_search(sequence, low, high, target) == -1

    # -- Edge / boundary cases ------------------------------------------------

    def test_empty_subrange(self):
        """When high < low the sub-range is empty; should return -1."""
        assert rec_linear_search([1, 2, 3], 2, 1, 2) == -1

    def test_low_equals_high_found(self):
        """Sub-range of exactly one element that matches target."""
        assert rec_linear_search([10, 20, 30], 1, 1, 20) == 1

    def test_low_equals_high_not_found(self):
        """Sub-range of exactly one element that does NOT match target."""
        assert rec_linear_search([10, 20, 30], 1, 1, 99) == -1

    @pytest.mark.parametrize(
        "sequence, low, high, target",
        [
            ([1, 2, 3], -1, 2, 1),   # low < 0
            ([1, 2, 3], 0, 3, 1),    # high == len(sequence)
            ([1, 2, 3], 0, -1, 1),   # high < 0
            ([1, 2, 3], 3, 0, 1),    # low == len(sequence)
        ],
    )
    def test_invalid_bounds_raise(self, sequence, low, high, target):
        """Out-of-range low or high must raise Exception."""
        with pytest.raises(Exception, match="Invalid upper or lower bound!"):
            rec_linear_search(sequence, low, high, target)

    def test_large_list_recursion(self):
        """Recursive search on a moderately large list to ensure no
        recursion-depth issues for typical inputs."""
        n = 900  # well below default recursion limit of ~1000
        seq = list(range(n))
        # target at the end
        assert rec_linear_search(seq, 0, n - 1, n - 1) == n - 1
        # target missing
        assert rec_linear_search(seq, 0, n - 1, -1) == -1