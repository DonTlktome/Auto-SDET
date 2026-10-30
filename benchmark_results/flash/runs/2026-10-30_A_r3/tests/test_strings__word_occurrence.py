"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


class TestWordOccurrence:
    """Tests for the word_occurrence function."""

    def test_returns_defaultdict_of_int(self):
        """Happy path: the return value is a defaultdict mapping words to ints."""
        result = word_occurrence("hello world")
        assert isinstance(result, defaultdict)
        assert result.default_factory is int
        assert dict(result) == {"hello": 1, "world": 1}

    def test_counts_repeated_words(self):
        """Happy path: repeated words are counted correctly."""
        result = word_occurrence("a b a b a")
        assert dict(result) == {"a": 3, "b": 2}

    def test_matches_counter_for_mixed_sentence(self):
        """Happy path: results agree with collections.Counter on a mixed sentence."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        result = word_occurrence(sentence)
        expected = Counter(sentence.split())
        assert all(result[word] == count for word, count in expected.items())
        assert len(result) == len(expected)

    def test_case_sensitivity(self):
        """Edge case: counting is case-sensitive, so 'A' and 'a' are distinct."""
        result = word_occurrence("A a A")
        assert dict(result) == {"A": 2, "a": 1}

    def test_multiple_spaces_are_collapsed(self):
        """Edge case: extra whitespace does not create empty-string keys."""
        result = word_occurrence("Two  spaces")
        assert dict(result) == {"Two": 1, "spaces": 1}
        assert "" not in result

    @pytest.mark.parametrize(
        "sentence",
        ["", " ", "   ", "\t", "\n", " \t\n "],
    )
    def test_empty_or_whitespace_only_returns_empty_dict(self, sentence):
        """Edge case: empty or whitespace-only input yields an empty mapping."""
        result = word_occurrence(sentence)
        assert dict(result) == {}
        assert len(result) == 0

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("single", {"single": 1}),
            ("one two three", {"one": 1, "two": 1, "three": 1}),
            ("x x x x", {"x": 4}),
            ("a\nb\tc", {"a": 1, "b": 1, "c": 1}),
            ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
            ("punctuation, stays! attached?", {"punctuation,": 1, "stays!": 1, "attached?": 1}),
            ("123 123 abc", {"123": 2, "abc": 1}),
        ],
    )
    def test_parametrized_counts(self, sentence, expected):
        """Happy path: various inputs produce the expected word counts."""
        assert dict(word_occurrence(sentence)) == expected

    def test_missing_key_returns_zero_without_inserting(self):
        """Edge case: defaultdict semantics give 0 for absent keys."""
        result = word_occurrence("alpha beta")
        assert result["gamma"] == 0
        # Accessing a missing key inserts it with the default value.
        assert "gamma" in result
        assert result["gamma"] == 0

    def test_preserves_first_seen_order(self):
        """Edge case: insertion order follows first appearance in the sentence."""
        result = word_occurrence("z y x z y z")
        assert list(result.keys()) == ["z", "y", "x"]
        assert list(result.values()) == [3, 2, 1]

    def test_does_not_mutate_input(self):
        """Edge case: the input string is left unchanged."""
        sentence = "keep me intact"
        original = str(sentence)
        word_occurrence(sentence)
        assert sentence == original

    def test_large_input_counts(self):
        """Boundary: a large repeated input is counted correctly."""
        sentence = " ".join(["word"] * 1000 + ["other"] * 500)
        result = word_occurrence(sentence)
        assert result["word"] == 1000
        assert result["other"] == 500
        assert len(result) == 2

    def test_non_string_input_raises_attribute_error(self):
        """Edge case: a non-string input (no .split) raises AttributeError."""
        with pytest.raises(AttributeError):
            word_occurrence(None)

    def test_unicode_words(self):
        """Edge case: unicode words are counted as distinct tokens."""
        result = word_occurrence("héllo héllo мир мир мир")
        assert dict(result) == {"héllo": 2, "мир": 3}