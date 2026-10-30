"""
Pytest test suite for word_occurrence function.

Tests cover:
- Normal sentences with repeated words
- Single word
- Empty string
- Multiple consecutive spaces
- Case sensitivity
- Numeric tokens
- Special characters / punctuation
"""

from collections import Counter

import pytest

from strings.word_occurrence import word_occurrence


class TestWordOccurrence:
    """Test suite for the word_occurrence function."""

    # ------------------------------------------------------------------
    # Positive / happy-path tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            (
                "a b A b c b d b d e f e g e h e i e j e 0",
                {
                    "a": 1, "b": 4, "A": 1, "c": 1, "d": 2,
                    "e": 6, "f": 1, "g": 1, "h": 1, "i": 1,
                    "j": 1, "0": 1,
                },
            ),
            (
                "hello world hello",
                {"hello": 2, "world": 1},
            ),
            (
                "one",
                {"one": 1},
            ),
            (
                "Two  spaces",
                {"Two": 1, "spaces": 1},
            ),
        ],
    )
    def test_word_occurrence_happy_path(
        self, sentence: str, expected: dict
    ) -> None:
        """word_occurrence should return correct counts for normal input."""
        result = word_occurrence(sentence)
        assert dict(result) == expected

    # ------------------------------------------------------------------
    # Edge-case tests
    # ------------------------------------------------------------------

    def test_empty_string_returns_empty_dict(self) -> None:
        """An empty string should produce an empty dictionary."""
        result = word_occurrence("")
        assert result == {}

    def test_only_spaces_returns_empty_dict(self) -> None:
        """A string containing only whitespace should produce an empty dict."""
        result = word_occurrence("     ")
        assert result == {}

    def test_leading_trailing_spaces_are_handled(self) -> None:
        """Leading and trailing spaces should not create empty-string keys."""
        result = word_occurrence("  hello   world  ")
        assert dict(result) == {"hello": 1, "world": 1}

    def test_case_sensitivity(self) -> None:
        """The function is case-sensitive: 'Word' and 'word' are distinct."""
        result = word_occurrence("Word word WORD")
        assert dict(result) == {"Word": 1, "word": 1, "WORD": 1}

    def test_numeric_tokens(self) -> None:
        """Numeric tokens are treated as regular words."""
        result = word_occurrence("123 456 123")
        assert dict(result) == {"123": 2, "456": 1}

    def test_punctuation_attached_to_words(self) -> None:
        """Punctuation attached to words is kept as part of the token."""
        result = word_occurrence("hello, world! hello,")
        assert dict(result) == {"hello,": 2, "world!": 1}

    def test_tabs_and_newlines_as_separators(self) -> None:
        """str.split() handles tabs and newlines as whitespace separators."""
        result = word_occurrence("one\ttwo\none")
        assert dict(result) == {"one": 2, "two": 1}

    # ------------------------------------------------------------------
    # Consistency / property-based sanity checks
    # ------------------------------------------------------------------

    def test_consistency_with_counter(self) -> None:
        """word_occurrence should match collections.Counter on the same input."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        result = word_occurrence(sentence)
        expected = Counter(sentence.split())
        assert dict(result) == dict(expected)

    def test_total_count_equals_number_of_words(self) -> None:
        """Sum of all counts should equal the number of tokens after split."""
        sentence = "the quick brown fox jumps over the lazy dog"
        result = word_occurrence(sentence)
        assert sum(result.values()) == len(sentence.split())

    def test_return_type_is_defaultdict(self) -> None:
        """The return value should be a defaultdict(int)."""
        from collections import defaultdict

        result = word_occurrence("a b c")
        assert isinstance(result, defaultdict)
        # Accessing a missing key should return 0, not raise KeyError
        assert result["nonexistent"] == 0
