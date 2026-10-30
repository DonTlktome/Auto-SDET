"""Tests for the ``reverse_words`` function in ``strings/reverse_words.py``."""

import pytest

from strings.reverse_words import reverse_words


class TestReverseWordsHappyPath:
    """Positive test cases for the happy path of ``reverse_words``."""

    def test_reverses_two_words(self):
        """A simple two-word sentence should have its words swapped."""
        assert reverse_words("hello world") == "world hello"

    def test_reverses_three_words(self):
        """A three-word sentence should be reversed in order."""
        assert reverse_words("I love Python") == "Python love I"

    def test_single_word_is_unchanged(self):
        """A single word with no surrounding whitespace stays the same."""
        assert reverse_words("Python") == "Python"

    def test_doctest_example_with_extra_spaces(self):
        """The documented example with irregular spacing must match the docstring."""
        assert reverse_words("I     Love          Python") == "Python Love I"


class TestReverseWordsEdgeCases:
    """Boundary and edge cases for ``reverse_words``."""

    def test_empty_string_returns_empty_string(self):
        """An empty string should produce an empty string."""
        assert reverse_words("") == ""

    def test_whitespace_only_returns_empty_string(self):
        """A string of only whitespace has no words, so the result is empty."""
        assert reverse_words("     ") == ""

    def test_leading_and_trailing_whitespace_is_stripped(self):
        """Leading/trailing whitespace must not create empty words."""
        assert reverse_words("   hello world   ") == "world hello"

    def test_tabs_and_newlines_are_treated_as_separators(self):
        """Any whitespace (tabs, newlines) should split words like spaces do."""
        assert reverse_words("alpha\tbeta\ngamma") == "gamma beta alpha"

    def test_single_character_words(self):
        """Single-character words should be reversed correctly."""
        assert reverse_words("a b c") == "c b a"

    def test_punctuation_is_preserved_with_its_word(self):
        """Punctuation attached to words must travel with the word."""
        assert reverse_words("Hello, world!") == "world! Hello,"

    def test_unicode_words_are_supported(self):
        """Non-ASCII words should be handled without corruption."""
        assert reverse_words("héllo wörld") == "wörld héllo"

    def test_original_string_is_not_mutated(self):
        """Strings are immutable, but verify the input value is unchanged."""
        original = "one two three"
        reverse_words(original)
        assert original == "one two three"

    def test_reversing_twice_restores_normalized_sentence(self):
        """Applying the function twice yields the whitespace-normalized original."""
        sentence = "  the   quick  brown   fox "
        assert reverse_words(reverse_words(sentence)) == "the quick brown fox"


class TestReverseWordsParametrized:
    """Parametrized coverage of many input/output pairs."""

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("", ""),
            (" ", ""),
            ("\t\n", ""),
            ("word", "word"),
            ("a b", "b a"),
            ("one two three four", "four three two one"),
            ("  spaced   out  ", "out spaced"),
            ("x  y", "y x"),
            ("1 2 3", "3 2 1"),
            ("CamelCase snake_case", "snake_case CamelCase"),
        ],
    )
    def test_reverse_words(self, sentence, expected):
        """Verify ``reverse_words`` output for a variety of inputs."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        ["", "   ", "single", "two words", "  many   spaces   here  "],
    )
    def test_output_has_no_extra_whitespace(self, sentence):
        """The result must never contain leading, trailing, or double spaces."""
        result = reverse_words(sentence)
        assert result == result.strip()
        assert "  " not in result

    @pytest.mark.parametrize(
        "sentence",
        ["", "one", "one two", "  a   b   c  ", "tab\tseparated\twords"],
    )
    def test_word_multiset_is_preserved(self, sentence):
        """Reversing must preserve the exact set of words, only reordered."""
        assert sorted(reverse_words(sentence).split()) == sorted(sentence.split())


class TestReverseWordsTypeContract:
    """Type-related contract checks for ``reverse_words``."""

    def test_returns_str_instance(self):
        """The function must always return a ``str``."""
        assert isinstance(reverse_words("hello world"), str)

    def test_returns_str_for_empty_input(self):
        """Even for empty input the return type must be ``str``."""
        assert isinstance(reverse_words(""), str)

    @pytest.mark.parametrize("bad_input", [None, 123, ["a", "b"]])
    def test_non_string_input_raises_attribute_error(self, bad_input):
        """Non-string inputs lacking ``.split`` should raise ``AttributeError``."""
        with pytest.raises(AttributeError):
            reverse_words(bad_input)

    def test_bytes_input_raises_type_error(self):
        """``bytes`` has ``.split``, so the failure surfaces as a ``TypeError``."""
        with pytest.raises(TypeError):
            reverse_words(b"bytes")
