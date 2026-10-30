import pytest

from palindrome import (
    is_palindrome,
    is_palindrome_recursive,
    is_palindrome_slice,
    is_palindrome_traversal,
    test_data,
)


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


# ---------------------------------------------------------------------------
# is_palindrome
# ---------------------------------------------------------------------------

class TestIsPalindrome:
    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s):
        """is_palindrome should return True for valid palindrome strings."""
        assert is_palindrome(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s):
        """is_palindrome should return False for non-palindrome strings."""
        assert is_palindrome(s) is False

    def test_empty_string_is_palindrome(self):
        """An empty string is trivially a palindrome."""
        assert is_palindrome("") is True

    def test_single_character_is_palindrome(self):
        """A single character is always a palindrome."""
        assert is_palindrome("z") is True

    def test_two_identical_characters(self):
        """Two identical characters form a palindrome."""
        assert is_palindrome("aa") is True

    def test_two_different_characters(self):
        """Two different characters do not form a palindrome."""
        assert is_palindrome("ab") is False

    def test_even_length_palindrome(self):
        """Even-length palindromes should be detected correctly."""
        assert is_palindrome("abba") is True

    def test_odd_length_palindrome(self):
        """Odd-length palindromes should be detected correctly."""
        assert is_palindrome("abcba") is True


# ---------------------------------------------------------------------------
# is_palindrome_traversal
# ---------------------------------------------------------------------------

class TestIsPalindromeTraversal:
    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s):
        """is_palindrome_traversal should return True for palindromes."""
        assert is_palindrome_traversal(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s):
        """is_palindrome_traversal should return False for non-palindromes."""
        assert is_palindrome_traversal(s) is False

    def test_empty_string_is_palindrome(self):
        """An empty string is trivially a palindrome."""
        assert is_palindrome_traversal("") is True

    def test_single_character_is_palindrome(self):
        """A single character is always a palindrome."""
        assert is_palindrome_traversal("x") is True

    def test_even_length_palindrome(self):
        """Even-length palindromes should be detected correctly."""
        assert is_palindrome_traversal("abba") is True

    def test_odd_length_palindrome(self):
        """Odd-length palindromes should be detected correctly."""
        assert is_palindrome_traversal("abcba") is True


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------

class TestIsPalindromeRecursive:
    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s):
        """is_palindrome_recursive should return True for palindromes."""
        assert is_palindrome_recursive(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s):
        """is_palindrome_recursive should return False for non-palindromes."""
        assert is_palindrome_recursive(s) is False

    def test_empty_string_is_palindrome(self):
        """An empty string is trivially a palindrome."""
        assert is_palindrome_recursive("") is True

    def test_single_character_is_palindrome(self):
        """A single character is always a palindrome."""
        assert is_palindrome_recursive("y") is True

    def test_even_length_palindrome(self):
        """Even-length palindromes should be detected correctly."""
        assert is_palindrome_recursive("abba") is True

    def test_odd_length_palindrome(self):
        """Odd-length palindromes should be detected correctly."""
        assert is_palindrome_recursive("abcba") is True


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------

class TestIsPalindromeSlice:
    @pytest.mark.parametrize("s", PALINDROMES)
    def test_returns_true_for_palindromes(self, s):
        """is_palindrome_slice should return True for palindromes."""
        assert is_palindrome_slice(s) is True

    @pytest.mark.parametrize("s", NON_PALINDROMES)
    def test_returns_false_for_non_palindromes(self, s):
        """is_palindrome_slice should return False for non-palindromes."""
        assert is_palindrome_slice(s) is False

    def test_empty_string_is_palindrome(self):
        """An empty string is trivially a palindrome."""
        assert is_palindrome_slice("") is True

    def test_single_character_is_palindrome(self):
        """A single character is always a palindrome."""
        assert is_palindrome_slice("w") is True

    def test_even_length_palindrome(self):
        """Even-length palindromes should be detected correctly."""
        assert is_palindrome_slice("abba") is True

    def test_odd_length_palindrome(self):
        """Odd-length palindromes should be detected correctly."""
        assert is_palindrome_slice("abcba") is True


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    @pytest.mark.parametrize(
        "s",
        PALINDROMES + NON_PALINDROMES + ["", "a", "ab", "abba", "abcba"],
    )
    def test_all_implementations_agree(self, s):
        """All four implementations should agree on every input string."""
        results = {
            is_palindrome(s),
            is_palindrome_traversal(s),
            is_palindrome_recursive(s),
            is_palindrome_slice(s),
        }
        assert len(results) == 1


# ---------------------------------------------------------------------------
# test_data integrity
# ---------------------------------------------------------------------------

class TestTestData:
    def test_test_data_matches_slice_implementation(self):
        """The module-level test_data should be consistent with is_palindrome_slice."""
        for key, value in test_data.items():
            assert is_palindrome_slice(key) == value

    def test_test_data_is_non_empty(self):
        """test_data should contain at least one entry."""
        assert len(test_data) > 0