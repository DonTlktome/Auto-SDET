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
            ("Python", "Python"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reverses word order and collapses extra whitespace."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("", ""),
            ("   ", ""),
            ("\t\n", ""),
            ("  leading", "leading"),
            ("trailing   ", "trailing"),
            ("  both  sides  ", "sides both"),
            ("\n\tmixed \t whitespace\n", "whitespace mixed"),
        ],
    )
    def test_reverse_words_whitespace_edge_cases(self, sentence, expected):
        """Handles empty strings and strings made only of whitespace."""
        assert reverse_words(sentence) == expected

    def test_reverse_words_single_word_with_padding(self):
        """A single word surrounded by whitespace is returned stripped."""
        assert reverse_words("   solo   ") == "solo"

    def test_reverse_words_preserves_internal_punctuation(self):
        """Punctuation attached to words is preserved as part of the word."""
        assert reverse_words("Hello, world!") == "world! Hello,"

    def test_reverse_words_is_involution_for_normalized_input(self):
        """Applying the function twice restores the normalized original."""
        sentence = "the quick brown fox"
        assert reverse_words(reverse_words(sentence)) == sentence

    def test_reverse_words_does_not_mutate_input(self):
        """The input string is immutable and remains unchanged."""
        sentence = "alpha beta gamma"
        original = str(sentence)
        reverse_words(sentence)
        assert sentence == original

    def test_reverse_words_returns_str(self):
        """The return type is always a str."""
        assert isinstance(reverse_words("a b c"), str)
        assert isinstance(reverse_words(""), str)

    def test_reverse_words_unicode(self):
        """Unicode words are reversed correctly."""
        assert reverse_words("héllo wörld") == "wörld héllo"

    def test_reverse_words_doctest_examples(self):
        """The documented doctest examples produce the documented output."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"
