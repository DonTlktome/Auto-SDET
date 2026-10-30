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

    def test_empty_string_returns_empty_dict(self):
        """An empty string should produce an empty mapping."""
        result = word_occurrence("")
        assert dict(result) == {}
        assert len(result) == 0

    def test_whitespace_only_string_returns_empty_dict(self):
        """A whitespace-only string should produce an empty mapping."""
        result = word_occurrence("   \t\n  ")
        assert dict(result) == {}

    def test_multiple_spaces_are_collapsed(self):
        """Extra whitespace between words should not create empty keys."""
        result = word_occurrence("Two  spaces")
        assert dict(result) == {"Two": 1, "spaces": 1}

    def test_leading_and_trailing_whitespace(self):
        """Leading/trailing whitespace should be ignored."""
        result = word_occurrence("  alpha beta  ")
        assert dict(result) == {"alpha": 1, "beta": 1}

    def test_newlines_and_tabs_are_separators(self):
        """Any whitespace character should act as a word separator."""
        result = word_occurrence("one\ttwo\nthree one")
        assert dict(result) == {"one": 2, "two": 1, "three": 1}

    def test_missing_key_returns_zero(self):
        """Accessing an absent key on the defaultdict should yield 0."""
        result = word_occurrence("only")
        assert result["absent"] == 0
        # Accessing a missing key inserts it with value 0.
        assert "absent" in result

    def test_punctuation_is_part_of_word(self):
        """Punctuation is not stripped; it stays attached to the token."""
        result = word_occurrence("hi, hi. hi")
        assert dict(result) == {"hi,": 1, "hi.": 1, "hi": 1}

    def test_single_word(self):
        """A single word should be counted exactly once."""
        result = word_occurrence("word")
        assert dict(result) == {"word": 1}

    def test_numeric_tokens(self):
        """Numeric tokens are treated as ordinary words."""
        result = word_occurrence("0 1 0 2 0")
        assert dict(result) == {"0": 3, "1": 1, "2": 1}

    def test_matches_counter_for_docstring_sentence(self):
        """Counts must match collections.Counter for the docstring example."""
        sentence = "a b A b c b d b d e f e g e h e i e j e 0"
        occurrence_dict = word_occurrence(sentence)
        expected = Counter(sentence.split())
        assert all(
            occurrence_dict[word] == count for word, count in expected.items()
        )
        assert dict(occurrence_dict) == dict(expected)

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("", {}),
            ("   ", {}),
            ("x", {"x": 1}),
            ("x x", {"x": 2}),
            ("x y x y x", {"x": 3, "y": 2}),
            ("a\tb\nc", {"a": 1, "b": 1, "c": 1}),
            ("A a", {"A": 1, "a": 1}),
            ("one  two   three", {"one": 1, "two": 1, "three": 1}),
        ],
    )
    def test_parametrized_counts(self, sentence, expected):
        """Verify counts across a variety of input shapes."""
        assert dict(word_occurrence(sentence)) == expected

    @pytest.mark.parametrize(
        "sentence",
        [
            "",
            "single",
            "a b c",
            "repeat repeat repeat",
            "  padded  words  ",
            "mixed CASE Mixed case",
        ],
    )
    def test_total_counts_equal_token_count(self, sentence):
        """The sum of all counts must equal the number of whitespace tokens."""
        result = word_occurrence(sentence)
        assert sum(result.values()) == len(sentence.split())

    def test_does_not_mutate_input(self):
        """The input string must remain unchanged after the call."""
        sentence = "keep me intact"
        original = str(sentence)
        word_occurrence(sentence)
        assert sentence == original

    def test_independent_results_between_calls(self):
        """Separate calls must return independent dictionaries."""
        first = word_occurrence("a a")
        second = word_occurrence("b")
        assert dict(first) == {"a": 2}
        assert dict(second) == {"b": 1}
        assert first is not second

    def test_long_repeated_input(self):
        """A large repeated input should be counted correctly."""
        sentence = " ".join(["word"] * 1000)
        result = word_occurrence(sentence)
        assert dict(result) == {"word": 1000}