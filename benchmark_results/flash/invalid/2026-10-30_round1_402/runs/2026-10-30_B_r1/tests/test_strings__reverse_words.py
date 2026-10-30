"""Tests for strings/reverse_words.py."""

import pytest

from strings.reverse_words import reverse_words


class TestReverseWords:
    """Tests for the reverse_words function."""

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I love Python", "Python love I"),
            ("I     Love          Python", "Python Love I"),
            ("hello", "hello"),
            ("a b", "b a"),
            ("one two three four", "four three two one"),
            ("  leading and trailing  ", "trailing and leading"),
            ("\t tabs\tand\nnewlines ", "newlines and tabs"),
        ],
    )
    def test_reverses_word_order(self, sentence, expected):
        """Happy path: words are reversed and whitespace is normalized."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize("sentence", ["", " ", "   ", "\t\n\r ", "\n\n"])
    def test_empty_or_whitespace_only_returns_empty_string(self, sentence):
        """Edge case: empty or whitespace-only input yields an empty string."""
        assert reverse_words(sentence) == ""

    def test_single_word_with_surrounding_whitespace(self):
        """Edge case: a single word surrounded by whitespace is returned stripped."""
        assert reverse_words("   word   ") == "word"

    def test_returns_new_string_and_does_not_mutate_input(self):
        """The input string is immutable; the result is a distinct object."""
        sentence = "alpha beta"
        result = reverse_words(sentence)
        assert result == "beta alpha"
        assert sentence == "alpha beta"
        assert result is not sentence

    def test_punctuation_is_preserved_with_words(self):
        """Edge case: punctuation stays attached to its word."""
        assert reverse_words("Hello, world!") == "world! Hello,"

    def test_unicode_words_are_supported(self):
        """Edge case: non-ASCII characters are handled correctly."""
        assert reverse_words("héllo wörld ünïcode") == "ünïcode wörld héllo"

    def test_reversing_twice_restores_normalized_sentence(self):
        """Property: applying the function twice restores the normalized input."""
        sentence = "  the   quick  brown   fox "
        normalized = " ".join(sentence.split())
        assert reverse_words(reverse_words(sentence)) == normalized

    def test_very_long_input(self):
        """Boundary: a large number of words is reversed correctly."""
        words = [f"w{i}" for i in range(1000)]
        sentence = " ".join(words)
        expected = " ".join(reversed(words))
        assert reverse_words(sentence) == expected

    def test_whitespace_only_variants_are_all_empty(self):
        """Boundary: mixed whitespace characters produce an empty result."""
        assert reverse_words(" \t\n\r\v\f ") == ""
