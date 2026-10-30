"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


class TestWordOccurrence:
    """Tests for the word_occurrence function."""

    def test_happy_path_basic_sentence(self):
        """A simple sentence should map each word to its number of occurrences."""
        result = word_occurrence("the quick brown fox jumps over the lazy dog the")
        assert result["the"] == 3
        assert result["quick"] == 1
        assert result["brown"] == 1
        assert result["fox"] == 1
        assert result["jumps"] == 1
        assert result["over"] == 1
        assert result["lazy"] == 1
        assert result["dog"] == 1

    def test_returns_defaultdict_of_int(self):
        """The return value should be a defaultdict(int) so missing keys yield 0."""
        result = word_occurrence("a b c")
        assert isinstance(result, defaultdict)
        assert result.default_factory is int
        assert result["not_present"] == 0

    def test_matches_counter_for_mixed_case_sentence(self):
        """Counts must match collections.Counter over the same split tokens."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        result = word_occurrence(sentence)
        expected = Counter(sentence.split())
        assert all(result[word] == count for word, count in expected.items())
        assert len(result) == len(expected)

    def test_case_sensitivity(self):
        """Words differing only in case are counted as distinct tokens."""
        result = word_occurrence("Word word WORD")
        assert result["Word"] == 1
        assert result["word"] == 1
        assert result["WORD"] == 1
        assert len(result) == 3

    def test_multiple_spaces_are_collapsed(self):
        """Extra whitespace between words must not create empty tokens."""
        result = word_occurrence("Two  spaces")
        assert dict(result) == {"Two": 1, "spaces": 1}

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n", " \t\n  "],
    )
    def test_empty_or_whitespace_only_returns_empty_dict(self, sentence):
        """Empty or whitespace-only input should produce an empty mapping."""
        result = word_occurrence(sentence)
        assert dict(result) == {}
        assert len(result) == 0

    @pytest.mark.parametrize(
        "sentence,expected",
        [
            ("single", {"single": 1}),
            ("a a a a a", {"a": 5}),
            ("x y x y", {"x": 2, "y": 2}),
            ("one\ttwo\nthree", {"one": 1, "two": 1, "three": 1}),
            ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
            ("punct! punct! punct?", {"punct!": 2, "punct?": 1}),
            ("123 123 abc", {"123": 2, "abc": 1}),
        ],
    )
    def test_parametrized_counts(self, sentence, expected):
        """Various inputs should yield the expected word-to-count mapping."""
        assert dict(word_occurrence(sentence)) == expected

    def test_punctuation_is_not_stripped(self):
        """Tokens are split on whitespace only, so punctuation stays attached."""
        result = word_occurrence("hello, hello world.")
        assert result["hello,"] == 1
        assert result["hello"] == 1
        assert result["world."] == 1

    def test_large_repeated_input(self):
        """A large input with a single repeated word should count correctly."""
        sentence = " ".join(["repeat"] * 1000)
        result = word_occurrence(sentence)
        assert result["repeat"] == 1000
        assert len(result) == 1

    def test_does_not_mutate_input(self):
        """The input string must remain unchanged after the call."""
        sentence = "keep this intact"
        original = str(sentence)
        word_occurrence(sentence)
        assert sentence == original

    def test_result_is_independent_between_calls(self):
        """Separate calls must not share state."""
        first = word_occurrence("a b")
        second = word_occurrence("c d")
        assert dict(first) == {"a": 1, "b": 1}
        assert dict(second) == {"c": 1, "d": 1}
        assert first is not second

    def test_unicode_words(self):
        """Non-ASCII words should be counted like any other token."""
        result = word_occurrence("café café naïve")
        assert result["café"] == 2
        assert result["naïve"] == 1

    def test_non_string_input_raises_attribute_error(self):
        """Passing a non-string (no .split) should raise AttributeError."""
        with pytest.raises(AttributeError):
            word_occurrence(12345)