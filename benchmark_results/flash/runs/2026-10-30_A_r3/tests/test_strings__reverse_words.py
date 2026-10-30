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
            ("a", "a"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reversing words returns words in reverse order with single spaces."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n", " \t\n "],
    )
    def test_reverse_words_empty_or_whitespace_only(self, sentence):
        """Empty or whitespace-only input yields an empty string."""
        assert reverse_words(sentence) == ""

    def test_reverse_words_single_word_with_whitespace(self):
        """A single word surrounded by whitespace is returned stripped."""
        assert reverse_words("   word   ") == "word"

    def test_reverse_words_preserves_word_content(self):
        """Word content (case, punctuation) is preserved, only order changes."""
        assert reverse_words("Hello, World!") == "World! Hello,"

    def test_reverse_words_is_involution(self):
        """Applying reverse_words twice restores the normalized sentence."""
        sentence = "  the   quick  brown   fox "
        once = reverse_words(sentence)
        assert reverse_words(once) == "the quick brown fox"

    def test_reverse_words_does_not_mutate_input(self):
        """The input string is immutable and remains unchanged."""
        sentence = "I love Python"
        reverse_words(sentence)
        assert sentence == "I love Python"

    def test_reverse_words_returns_str(self):
        """The return type is always a str."""
        assert isinstance(reverse_words("a b c"), str)

    def test_reverse_words_unicode(self):
        """Unicode words are handled correctly."""
        assert reverse_words("héllo wörld 日本語") == "日本語 wörld héllo"

    def test_reverse_words_long_input(self):
        """A long input is reversed correctly."""
        words = [f"w{i}" for i in range(1000)]
        sentence = " ".join(words)
        assert reverse_words(sentence) == " ".join(reversed(words))

    def test_reverse_words_doctest_examples(self):
        """The documented doctest examples produce the documented output."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"
