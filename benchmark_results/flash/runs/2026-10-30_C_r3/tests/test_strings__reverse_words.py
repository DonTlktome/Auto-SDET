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
            ("a b c d e", "e d c b a"),
            ("one two", "two one"),
            ("  leading and trailing  ", "trailing and leading"),
            ("\t tabs\tand\nnewlines ", "newlines and tabs"),
            ("Python", "Python"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reversing words should produce the expected word order."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n", " \t\n  "],
    )
    def test_reverse_words_empty_or_whitespace_only(self, sentence):
        """Empty or whitespace-only input should return an empty string."""
        assert reverse_words(sentence) == ""

    def test_reverse_words_single_word_with_whitespace(self):
        """A single word surrounded by whitespace should be returned stripped."""
        assert reverse_words("   word   ") == "word"

    def test_reverse_words_preserves_word_content(self):
        """Word characters (case, punctuation) must not be altered."""
        assert reverse_words("Hello, World!") == "World! Hello,"

    def test_reverse_words_is_involution(self):
        """Applying reverse_words twice should restore the normalized sentence."""
        sentence = "  the   quick brown   fox  "
        once = reverse_words(sentence)
        assert reverse_words(once) == "the quick brown fox"

    def test_reverse_words_returns_str(self):
        """The return type must always be a str."""
        assert isinstance(reverse_words("a b"), str)
        assert isinstance(reverse_words(""), str)

    def test_reverse_words_does_not_mutate_input(self):
        """The original input string must remain unchanged (strings are immutable)."""
        sentence = "I love Python"
        reverse_words(sentence)
        assert sentence == "I love Python"

    def test_reverse_words_unicode(self):
        """Unicode words should be reversed correctly."""
        assert reverse_words("héllo wörld") == "wörld héllo"

    def test_reverse_words_doctest_examples(self):
        """The documented doctest examples must hold."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"
