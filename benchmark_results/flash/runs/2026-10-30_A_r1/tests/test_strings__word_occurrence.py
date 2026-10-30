"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


class TestWordOccurrence:
    """Tests for the word_occurrence function."""

    def test_returns_defaultdict(self):
        """The function should return a defaultdict(int) instance."""
        result = word_occurrence("hello world")
        assert isinstance(result, defaultdict)
        assert result.default_factory is int

    def test_simple_sentence_counts(self):
        """Each word in a simple sentence should be counted once."""
        result = word_occurrence("hello world")
        assert dict(result) == {"hello": 1, "world": 1}

    def test_repeated_words_are_counted(self):
        """Repeated words should accumulate their counts."""
        result = word_occurrence("a b a b a")
        assert dict(result) == {"a": 3, "b": 2}

    def test_case_sensitivity(self):
        """Words differing only in case are treated as distinct keys."""
        result = word_occurrence("a A a A")
        assert dict(result) == {"a": 2, "A": 2}

    def test_matches_counter_for_docstring_sentence(self):
        """Counts must match collections.Counter for the docstring example."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        result = word_occurrence(sentence)
        expected = Counter(sentence.split())
        assert all(result[word] == count for word, count in expected.items())
        assert len(result) == len(expected)

    def test_multiple_spaces_are_ignored(self):
        """Extra whitespace should not produce empty-string keys."""
        result = word_occurrence("Two  spaces")
        assert dict(result) == {"Two": 1, "spaces": 1}
        assert "" not in result

    @pytest.mark.parametrize(
        "sentence",
        ["", "   ", "\t", "\n", " \t\n  "],
    )
    def test_empty_or_whitespace_only_returns_empty_dict(self, sentence):
        """Empty or whitespace-only input should yield an empty mapping."""
        result = word_occurrence(sentence)
        assert dict(result) == {}
        assert len(result) == 0

    @pytest.mark.parametrize(
        "sentence,expected",
        [
            ("single", {"single": 1}),
            ("one two three", {"one": 1, "two": 1, "three": 1}),
            ("x x x x", {"x": 4}),
            ("a\nb\tc", {"a": 1, "b": 1, "c": 1}),
            ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
            ("word, word. word!", {"word,": 1, "word.": 1, "word!": 1}),
        ],
    )
    def test_parametrized_inputs(self, sentence, expected):
        """Various inputs should produce the expected word counts."""
        assert dict(word_occurrence(sentence)) == expected

    def test_punctuation_is_not_stripped(self):
        """Punctuation attached to words is kept as part of the token."""
        result = word_occurrence("hi hi, hi.")
        assert dict(result) == {"hi": 1, "hi,": 1, "hi.": 1}

    def test_missing_key_returns_zero(self):
        """Accessing an absent key on the defaultdict should return 0."""
        result = word_occurrence("alpha beta")
        assert result["gamma"] == 0
        # Accessing a missing key inserts it with the default value.
        assert "gamma" in result

    def test_large_input_counts(self):
        """A large repeated input should be counted correctly."""
        sentence = " ".join(["word"] * 1000 + ["other"] * 500)
        result = word_occurrence(sentence)
        assert result["word"] == 1000
        assert result["other"] == 500
        assert len(result) == 2

    def test_unicode_words(self):
        """Non-ASCII words should be counted like any other token."""
        result = word_occurrence("héllo wörld héllo 日本語 日本語")
        assert dict(result) == {"héllo": 2, "wörld": 1, "日本語": 2}

    def test_input_not_mutated(self):
        """The input string must remain unchanged (strings are immutable)."""
        sentence = "keep me intact"
        original = sentence
        word_occurrence(sentence)
        assert sentence == original

    def test_result_is_independent_between_calls(self):
        """Separate calls must not share state."""
        first = word_occurrence("a a b")
        second = word_occurrence("a b b")
        assert dict(first) == {"a": 2, "b": 1}
        assert dict(second) == {"a": 1, "b": 2}
        assert first is not second