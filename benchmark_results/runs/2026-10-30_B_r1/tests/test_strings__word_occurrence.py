"""Tests for the word_occurrence function."""

from collections import Counter

import pytest

from word_occurrence import word_occurrence


class TestWordOccurrence:
    """Test suite for word_occurrence."""

    def test_happy_path_counts_each_word(self):
        """Test that a normal sentence returns correct per-word counts."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        result = word_occurrence(sentence)

        expected = Counter(sentence.split())
        assert dict(result) == dict(expected)

    def test_single_word(self):
        """Test that a single-word sentence returns a count of one."""
        result = word_occurrence("hello")
        assert dict(result) == {"hello": 1}

    def test_repeated_single_word(self):
        """Test that a sentence with one repeated word counts correctly."""
        result = word_occurrence("word word word")
        assert dict(result) == {"word": 3}

    def test_multiple_spaces_are_collapsed(self):
        """Test that consecutive spaces do not produce empty-string keys."""
        result = word_occurrence("Two  spaces")
        assert dict(result) == {"Two": 1, "spaces": 1}

    def test_leading_and_trailing_whitespace(self):
        """Test that leading/trailing whitespace is ignored by split()."""
        result = word_occurrence("  leading trailing  ")
        assert dict(result) == {"leading": 1, "trailing": 1}

    def test_case_sensitive_counting(self):
        """Test that words differing only by case are counted separately."""
        result = word_occurrence("Hello hello HELLO")
        assert dict(result) == {"Hello": 1, "hello": 1, "HELLO": 1}

    def test_empty_string_returns_empty_dict(self):
        """Test that an empty string yields an empty dictionary."""
        result = word_occurrence("")
        assert dict(result) == {}

    def test_whitespace_only_returns_empty_dict(self):
        """Test that a whitespace-only string yields an empty dictionary."""
        result = word_occurrence("   \t  \n  ")
        assert dict(result) == {}

    def test_return_type_is_defaultdict(self):
        """Test that the return value is a defaultdict of int."""
        result = word_occurrence("a b a")
        assert isinstance(result, dict)
        # Accessing a missing key should return 0, not raise KeyError.
        assert result["missing"] == 0

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("", {}),
            ("one", {"one": 1}),
            ("one two", {"one": 1, "two": 1}),
            ("one one", {"one": 2}),
            ("a b a b a", {"a": 3, "b": 2}),
            ("x  y   x", {"x": 2, "y": 1}),
        ],
    )
    def test_parametrized_counts(self, sentence, expected):
        """Test various inputs against their expected word counts."""
        assert dict(word_occurrence(sentence)) == expected

    def test_matches_counter_for_random_sentence(self):
        """Test that output matches collections.Counter for a mixed sentence."""
        sentence = "the quick brown fox jumps over the lazy dog the fox"
        result = word_occurrence(sentence)
        assert dict(result) == dict(Counter(sentence.split()))