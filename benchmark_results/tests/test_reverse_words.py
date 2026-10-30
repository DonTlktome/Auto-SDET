"""Pytest test suite for the `reverse_words` function."""

import pytest
from reverse_words import reverse_words


class TestReverseWords:
    """Tests for reverse_words(sentence: str) -> str."""

    # ------------------------------------------------------------------
    # Positive / happy-path tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I love Python", "Python love I"),
            ("hello world", "world hello"),
            ("a b c", "c b a"),
            ("single", "single"),
            ("  leading and trailing  ", "trailing and leading"),
            ("one   two  three", "three two one"),
            ("", ""),  # empty string edge case
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reverse the order of words for various normal inputs."""
        assert reverse_words(sentence) == expected

    # ------------------------------------------------------------------
    # Edge-case / boundary tests
    # ------------------------------------------------------------------

    def test_only_whitespace(self):
        """A string containing only whitespace should return an empty string."""
        assert reverse_words("     ") == ""

    def test_single_word_with_surrounding_spaces(self):
        """A single word with leading/trailing spaces returns just the word."""
        assert reverse_words("   hello   ") == "hello"

    def test_tabs_and_newlines_as_whitespace(self):
        """str.split() handles all whitespace, so tabs/newlines are collapsed."""
        assert reverse_words("one\ttwo\nthree") == "three two one"

    def test_punctuation_is_preserved(self):
        """Punctuation attached to words stays attached after reversal."""
        assert reverse_words("Hello, world!") == "world! Hello,"

    def test_numbers_as_words(self):
        """Numeric tokens are treated as words."""
        assert reverse_words("123 456 789") == "789 456 123"

    def test_mixed_case_preserved(self):
        """Word casing is preserved; only order changes."""
        assert reverse_words("UPPER lower Mixed") == "Mixed lower UPPER"

    def test_unicode_characters(self):
        """Unicode word characters are handled correctly."""
        assert reverse_words("café naïve résumé") == "résumé naïve café"

    def test_doctest_examples(self):
        """Verify the examples from the function's own docstring."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"