"""Tests for the ``reverse_words`` function in ``strings/reverse_words.py``."""

import pytest

from strings.reverse_words import reverse_words


class TestReverseWordsHappyPath:
    """Positive test cases for reverse_words."""

    def test_reverses_two_words(self):
        """A simple two-word sentence should have its words swapped."""
        assert reverse_words("hello world") == "world hello"

    def test_reverses_three_words(self):
        """A three-word sentence should be reversed in order."""
        assert reverse_words("I love Python") == "Python love I"

    def test_single_word_is_unchanged(self):
        """A single word with no whitespace should be returned as-is."""
        assert reverse_words("Python") == "Python"

    def test_doctest_example_with_extra_spaces(self):
        """The documented example with irregular spacing must match the docstring."""
        assert reverse_words("I     Love          Python") == "Python Love I"


class TestReverseWordsEdgeCases:
    """Boundary and edge cases for reverse_words."""

    def test_empty_string_returns_empty_string(self):
        """An empty string should produce an empty string."""
        assert reverse_words("") == ""

    def test_whitespace_only_returns_empty_string(self):
        """A string of only whitespace should collapse to an empty string."""
        assert reverse_words("     ") == ""

    def test_leading_and_trailing_whitespace_is_stripped(self):
        """Surrounding whitespace must not produce leading/trailing spaces."""
        assert reverse_words("   alpha beta   ") == "beta alpha"

    def test_tabs_and_newlines_are_treated_as_separators(self):
        """Any whitespace (tabs, newlines) should act as a word separator."""
        assert reverse_words("alpha\tbeta\ngamma") == "gamma beta alpha"

    def test_mixed_whitespace_types(self):
        """Mixed whitespace runs should be normalized to single spaces."""
        assert reverse_words("  one \t two \n three  ") == "three two one"

    def test_punctuation_is_preserved(self):
        """Punctuation attached to words should be preserved verbatim."""
        assert reverse_words("hello, world!") == "world! hello,"

    def test_unicode_words_are_supported(self):
        """Non-ASCII words should be reversed correctly."""
        assert reverse_words("héllo wörld") == "wörld héllo"

    def test_repeated_words_are_preserved(self):
        """Duplicate words should all be kept, just reordered."""
        assert reverse_words("a a b") == "b a a"

    def test_result_has_no_double_spaces(self):
        """The output must never contain consecutive spaces."""
        result = reverse_words("  a    b     c  ")
        assert "  " not in result
        assert result == "c b a"


class TestReverseWordsProperties:
    """Invariant-based tests for reverse_words."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "",
            " ",
            "one",
            "one two",
            "  one   two   three  ",
            "a\tb\nc",
            "Hello, World!",
            "héllo wörld",
        ],
    )
    def test_reversing_twice_restores_normalized_sentence(self, sentence):
        """Applying reverse_words twice yields the whitespace-normalized input."""
        normalized = " ".join(sentence.split())
        assert reverse_words(reverse_words(sentence)) == normalized

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("a b c", "c b a"),
            ("a", "a"),
            ("", ""),
            ("   ", ""),
            ("x  y", "y x"),
            ("1 2 3 4", "4 3 2 1"),
        ],
    )
    def test_parametrized_reversals(self, sentence, expected):
        """Multiple input/output pairs should reverse word order as expected."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        ["", "   ", "single", "two words", "  many   spaced   words  "],
    )
    def test_word_count_is_preserved(self, sentence):
        """The number of words in the output equals the number in the input."""
        assert len(reverse_words(sentence).split()) == len(sentence.split())

    @pytest.mark.parametrize(
        "sentence",
        ["", "   ", "single", "two words", "  many   spaced   words  "],
    )
    def test_output_is_whitespace_normalized(self, sentence):
        """Output must be stripped and contain only single-space separators."""
        result = reverse_words(sentence)
        assert result == result.strip()
        assert "  " not in result
        assert "\t" not in result
        assert "\n" not in result

    def test_returns_string_type(self):
        """The function must always return a str instance."""
        assert isinstance(reverse_words("a b"), str)
        assert isinstance(reverse_words(""), str)