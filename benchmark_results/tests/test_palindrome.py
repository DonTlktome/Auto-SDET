import pytest
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, r"D:\Documents\Study\Projects\algorithms-bench\strings")
import palindrome


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

PALINDROMES = [
    "MALAYALAM",
    "rotor",
    "level",
    "A",
    "BB",
    "amanaplanacanalpanama",
]

NON_PALINDROMES = [
    "String",
    "ABC",
    "abcdba",
    "AB",
]

EMPTY_STRING = ""


# ---------------------------------------------------------------------------
# is_palindrome
# ---------------------------------------------------------------------------

class TestIsPalindrome:
    """Tests for the two-pointer is_palindrome function."""

    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s: str) -> None:
        """is_palindrome should return True for known palindromes."""
        assert palindrome.is_palindrome(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s: str) -> None:
        """is_palindrome should return False for non-palindromes."""
        assert palindrome.is_palindrome(s) is False

    def test_empty_string_is_palindrome(self) -> None:
        """An empty string is vacuously a palindrome."""
        assert palindrome.is_palindrome(EMPTY_STRING) is True

    def test_single_character_is_palindrome(self) -> None:
        """A single-character string is always a palindrome."""
        assert palindrome.is_palindrome("Z") is True

    def test_two_same_characters(self) -> None:
        """Two identical characters form a palindrome."""
        assert palindrome.is_palindrome("zz") is True

    def test_two_different_characters(self) -> None:
        """Two different characters do not form a palindrome."""
        assert palindrome.is_palindrome("za") is False

    def test_case_sensitivity(self) -> None:
        """is_palindrome is case-sensitive; 'Aa' is not a palindrome."""
        assert palindrome.is_palindrome("Aa") is False

    def test_palindrome_with_spaces(self) -> None:
        """Spaces are treated as characters; 'a a' is a palindrome."""
        assert palindrome.is_palindrome("a a") is True

    def test_long_palindrome(self) -> None:
        """A long even-length palindrome should be detected correctly."""
        s = "ab" * 500  # "ababab..." — not a palindrome
        assert palindrome.is_palindrome(s) is False
        s = "ab" * 500 + "ba" * 500  # not symmetric
        # Build a true long palindrome
        half = "abcdefghij"
        s = half + half[::-1]
        assert palindrome.is_palindrome(s) is True


# ---------------------------------------------------------------------------
# is_palindrome_traversal
# ---------------------------------------------------------------------------

class TestIsPalindromeTraversal:
    """Tests for the half-traversal is_palindrome_traversal function."""

    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s: str) -> None:
        """Should return True for known palindromes."""
        assert palindrome.is_palindrome_traversal(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s: str) -> None:
        """Should return False for non-palindromes."""
        assert palindrome.is_palindrome_traversal(s) is False

    def test_empty_string_is_palindrome(self) -> None:
        """Empty string should be considered a palindrome."""
        assert palindrome.is_palindrome_traversal(EMPTY_STRING) is True

    def test_single_character(self) -> None:
        """Single character is a palindrome."""
        assert palindrome.is_palindrome_traversal("X") is True

    def test_even_length_palindrome(self) -> None:
        """Even-length palindrome 'ABBA' should return True."""
        assert palindrome.is_palindrome_traversal("ABBA") is True

    def test_odd_length_palindrome(self) -> None:
        """Odd-length palindrome 'ABCBA' should return True."""
        assert palindrome.is_palindrome_traversal("ABCBA") is True

    def test_near_palindrome_even(self) -> None:
        """A string that differs in the middle should return False."""
        assert palindrome.is_palindrome_traversal("ABCA") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------

class TestIsPalindromeRecursive:
    """Tests for the recursive is_palindrome_recursive function."""

    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s: str) -> None:
        """Should return True for known palindromes."""
        assert palindrome.is_palindrome_recursive(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s: str) -> None:
        """Should return False for non-palindromes."""
        assert palindrome.is_palindrome_recursive(s) is False

    def test_empty_string_is_palindrome(self) -> None:
        """Empty string base case: len <= 1 → True."""
        assert palindrome.is_palindrome_recursive(EMPTY_STRING) is True

    def test_single_character(self) -> None:
        """Single character base case: len <= 1 → True."""
        assert palindrome.is_palindrome_recursive("Q") is True

    def test_two_same_characters(self) -> None:
        """Two identical chars: first == last, recurse on empty → True."""
        assert palindrome.is_palindrome_recursive("QQ") is True

    def test_two_different_characters(self) -> None:
        """Two different chars: first != last → False."""
        assert palindrome.is_palindrome_recursive("QR") is False

    def test_deep_recursion(self) -> None:
        """Recursive function should handle a moderately long palindrome."""
        s = "a" * 900 + "b" + "a" * 900
        assert palindrome.is_palindrome_recursive(s) is True


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------

class TestIsPalindromeSlice:
    """Tests for the slicing-based is_palindrome_slice function."""

    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s: str) -> None:
        """Should return True for known palindromes."""
        assert palindrome.is_palindrome_slice(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s: str) -> None:
        """Should return False for non-palindromes."""
        assert palindrome.is_palindrome_slice(s) is False

    def test_empty_string_is_palindrome(self) -> None:
        """Empty string reversed is itself → True."""
        assert palindrome.is_palindrome_slice(EMPTY_STRING) is True

    def test_single_character(self) -> None:
        """Single character reversed is itself → True."""
        assert palindrome.is_palindrome_slice("P") is True

    def test_unicode_palindrome(self) -> None:
        """Slice comparison works with Unicode characters."""
        assert palindrome.is_palindrome_slice("∞∞") is True

    def test_unicode_non_palindrome(self) -> None:
        """Slice comparison correctly rejects non-palindrome Unicode."""
        assert palindrome.is_palindrome_slice("∞∆") is False


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------

class TestCrossImplementationConsistency:
    """Ensure all four implementations agree on a broad set of inputs."""

    FUNCTIONS = [
        palindrome.is_palindrome,
        palindrome.is_palindrome_traversal,
        palindrome.is_palindrome_recursive,
        palindrome.is_palindrome_slice,
    ]

    ALL_TEST_CASES = PALINDROMES + NON_PALINDROMES + [EMPTY_STRING]

    @pytest.mark.parametrize("s", ALL_TEST_CASES)
    def test_all_implementations_agree(self, s: str) -> None:
        """Every implementation must return the same result for each input."""
        results = [f(s) for f in self.FUNCTIONS]
        assert all(r == results[0] for r in results), (
            f"Mismatch for input {s!r}: {results}"
        )


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------

class TestBenchmarkFunction:
    """Tests for benchmark_function (mocked to avoid real timing)."""

    @patch("palindrome.timeit")
    def test_calls_timeit_with_correct_stmt_and_setup(self, mock_timeit: MagicMock) -> None:
        """benchmark_function should invoke timeit with the expected stmt and setup."""
        mock_timeit.return_value = 0.0
        palindrome.benchmark_function("is_palindrome_slice")

        mock_timeit.assert_called_once()
        _, kwargs = mock_timeit.call_args
        assert "is_palindrome_slice" in kwargs["stmt"]
        assert "is_palindrome_slice" in kwargs["setup"]
        assert "test_data" in kwargs["setup"]
        assert kwargs["number"] == 500000

    @patch("builtins.print")
    @patch("palindrome.timeit")
    def test_prints_result_line(self, mock_timeit: MagicMock, mock_print: MagicMock) -> None:
        """benchmark_function should print a formatted result line."""
        mock_timeit.return_value = 1.23456

        palindrome.benchmark_function("is_palindrome")

        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        assert "is_palindrome" in output
        assert "500,000" in output
        assert "1.23456" in output

    @patch("builtins.print")
    @patch("palindrome.timeit")
    def test_benchmark_with_recursive_implementation(
        self, mock_timeit: MagicMock, mock_print: MagicMock
    ) -> None:
        """benchmark_function works correctly for the recursive variant."""
        mock_timeit.return_value = 2.0

        palindrome.benchmark_function("is_palindrome_recursive")

        mock_timeit.assert_called_once()
        _, kwargs = mock_timeit.call_args
        assert "is_palindrome_recursive" in kwargs["stmt"]
        assert "is_palindrome_recursive" in kwargs["setup"]
        mock_print.assert_called_once()
