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
            ("Python", "Python"),
            ("a  b   c    d", "d c b a"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reversing words returns words in reverse order with single spaces."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n", " \t\n  "],
    )
    def test_reverse_words_empty_or_whitespace_only(self, sentence):
        """Empty or whitespace-only input yields an empty string."""
        assert reverse_words(sentence) == ""

    def test_reverse_words_single_word_unchanged(self):
        """A single word with no surrounding whitespace is returned unchanged."""
        assert reverse_words("word") == "word"

    def test_reverse_words_is_involution(self):
        """Applying reverse_words twice restores the original word order."""
        sentence = "the quick brown fox jumps"
        assert reverse_words(reverse_words(sentence)) == sentence

    def test_reverse_words_preserves_word_content(self):
        """Word characters are preserved; only order and spacing change."""
        sentence = "Hello, World! 123"
        result = reverse_words(sentence)
        assert result == "123 World! Hello,"
        assert sorted(result.split()) == sorted(sentence.split())

    def test_reverse_words_does_not_mutate_input(self):
        """The input string is immutable and remains unchanged."""
        sentence = "keep me intact"
        original = str(sentence)
        reverse_words(sentence)
        assert sentence == original

    def test_reverse_words_returns_str_type(self):
        """The return value is always a str instance."""
        assert isinstance(reverse_words("a b c"), str)
        assert isinstance(reverse_words(""), str)

    def test_reverse_words_unicode(self):
        """Unicode words are reversed correctly."""
        assert reverse_words("héllo wörld ünïcode") == "ünïcode wörld héllo"

    def test_reverse_words_doctest_examples(self):
        """The documented doctest examples produce the documented output."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"
