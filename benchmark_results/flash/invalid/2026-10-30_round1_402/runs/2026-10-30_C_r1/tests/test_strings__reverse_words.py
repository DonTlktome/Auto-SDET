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

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n\n", " \t \n "],
    )
    def test_empty_or_whitespace_only_returns_empty_string(self, sentence):
        """Edge case: empty or whitespace-only input yields an empty string."""
        assert reverse_words(sentence) == ""

    def test_single_word_with_surrounding_whitespace(self):
        """Edge case: a single word surrounded by whitespace is returned stripped."""
        assert reverse_words("   word   ") == "word"

    def test_does_not_mutate_characters_within_words(self):
        """Edge case: characters inside words keep their original order."""
        assert reverse_words("abc def") == "def abc"

    def test_punctuation_is_preserved(self):
        """Edge case: punctuation attached to words is preserved."""
        assert reverse_words("Hello, world!") == "world! Hello,"

    def test_unicode_words(self):
        """Edge case: non-ASCII words are handled correctly."""
        assert reverse_words("héllo wörld") == "wörld héllo"

    def test_returns_new_string(self):
        """Edge case: the function returns a new str object, not the input."""
        sentence = "a b c"
        result = reverse_words(sentence)
        assert result == "c b a"
        assert result is not sentence
        assert sentence == "a b c"

    def test_idempotent_after_two_reversals(self):
        """Edge case: reversing twice restores the original word order."""
        sentence = "alpha beta gamma"
        assert reverse_words(reverse_words(sentence)) == sentence

    def test_whitespace_normalization_is_stable(self):
        """Edge case: output contains single spaces only, no leading/trailing space."""
        result = reverse_words("  a   b    c  ")
        assert result == "c b a"
        assert "  " not in result
        assert result == result.strip()

    def test_non_string_input_raises_attribute_error(self):
        """Edge case: non-string input raises AttributeError (no .split)."""
        with pytest.raises(AttributeError):
            reverse_words(123)
