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

    @pytest.mark.parametrize("sentence", ["", " ", "   ", "\t\n\r ", "     \t  \n "])
    def test_empty_or_whitespace_only_returns_empty_string(self, sentence):
        """Edge case: empty or whitespace-only input yields an empty string."""
        assert reverse_words(sentence) == ""

    def test_single_word_unchanged(self):
        """Edge case: a single word is returned unchanged."""
        assert reverse_words("Python") == "Python"

    def test_palindromic_words(self):
        """Edge case: words themselves are not reversed, only their order."""
        assert reverse_words("ab ba") == "ba ab"

    def test_duplicate_words_preserved(self):
        """Edge case: duplicate words are all preserved in reversed order."""
        assert reverse_words("a a b a") == "a b a a"

    def test_returns_new_string_and_does_not_mutate_input(self):
        """The input string is immutable; result is a distinct string object."""
        sentence = "I love Python"
        result = reverse_words(sentence)
        assert result == "Python love I"
        assert sentence == "I love Python"
        assert result is not sentence

    def test_output_has_single_spaces_only(self):
        """Invariant: output never contains consecutive or edge whitespace."""
        result = reverse_words("  a   b\t\tc\n d  ")
        assert result == "d c b a"
        assert "  " not in result
        assert result == result.strip()

    def test_unicode_words(self):
        """Edge case: non-ASCII words are handled correctly."""
        assert reverse_words("héllo wörld ünïcode") == "ünïcode wörld héllo"

    def test_long_input(self):
        """Boundary: a large number of words is reversed correctly."""
        words = [f"w{i}" for i in range(1000)]
        assert reverse_words(" ".join(words)) == " ".join(reversed(words))

    def test_non_string_raises_attribute_error(self):
        """Edge case: non-string input raises AttributeError (no .split)."""
        with pytest.raises(AttributeError):
            reverse_words(None)

    def test_doctest_examples(self):
        """The documented doctest examples hold."""
        assert reverse_words("I love Python") == "Python love I"
        assert reverse_words("I     Love          Python") == "Python Love I"
