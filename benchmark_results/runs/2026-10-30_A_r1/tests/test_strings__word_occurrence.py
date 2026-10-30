from collections import Counter

import pytest

from word_occurrence import word_occurrence


class TestWordOccurrence:
    """Test suite for word_occurrence."""

    @pytest.mark.parametrize(
        ("sentence", "expected"),
        [
            ("a b A b c b d b d e f e g e h e i e j e 0",
             Counter("a b A b c b d b d e f e g e h e i e j e 0".split())),
            ("Two  spaces", {"Two": 1, "spaces": 1}),
            ("hello", {"hello": 1}),
            ("one two three", {"one": 1, "two": 1, "three": 1}),
        ],
    )
    def test_word_occurrence_happy_path(self, sentence, expected):
        """Test that word counts match expected values for normal input."""
        result = word_occurrence(sentence)
        assert dict(result) == dict(expected)

    def test_word_occurrence_repeated_words(self):
        """Test that repeated words are counted correctly."""
        result = word_occurrence("foo foo foo bar bar baz")
        assert dict(result) == {"foo": 3, "bar": 2, "baz": 1}

    def test_word_occurrence_empty_string(self):
        """Test that an empty string returns an empty dict."""
        assert dict(word_occurrence("")) == {}

    def test_word_occurrence_whitespace_only(self):
        """Test that whitespace-only input returns an empty dict."""
        assert dict(word_occurrence("   \t\n  ")) == {}

    def test_word_occurrence_leading_trailing_spaces(self):
        """Test that leading/trailing whitespace is ignored by split()."""
        result = word_occurrence("  hello   world  ")
        assert dict(result) == {"hello": 1, "world": 1}

    def test_word_occurrence_case_sensitive(self):
        """Test that words differing only by case are treated as distinct."""
        result = word_occurrence("Hello hello HELLO")
        assert dict(result) == {"Hello": 1, "hello": 1, "HELLO": 1}

    def test_word_occurrence_returns_defaultdict(self):
        """Test that the return value is a defaultdict of int."""
        result = word_occurrence("a b")
        assert isinstance(result, dict)
        assert result["missing"] == 0

    def test_word_occurrence_matches_counter(self):
        """Test that results match collections.Counter for a mixed sentence."""
        sentence = "the quick brown fox jumps over the lazy dog the"
        result = word_occurrence(sentence)
        assert dict(result) == dict(Counter(sentence.split()))