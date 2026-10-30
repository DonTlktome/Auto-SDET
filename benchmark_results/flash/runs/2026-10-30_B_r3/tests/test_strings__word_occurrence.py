"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


def test_word_occurrence_returns_defaultdict():
    """The function should return a defaultdict(int) instance."""
    result = word_occurrence("hello world")
    assert isinstance(result, defaultdict)
    assert result.default_factory is int


def test_word_occurrence_simple_sentence():
    """Counts for a simple sentence with unique words should all be 1."""
    result = word_occurrence("the quick brown fox")
    assert dict(result) == {"the": 1, "quick": 1, "brown": 1, "fox": 1}


def test_word_occurrence_repeated_words():
    """Repeated words should be counted correctly."""
    result = word_occurrence("a b a c a b")
    assert dict(result) == {"a": 3, "b": 2, "c": 1}


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case must be counted separately."""
    result = word_occurrence("A a A")
    assert dict(result) == {"A": 2, "a": 1}


def test_word_occurrence_matches_counter():
    """Result must agree with collections.Counter on the same input."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert all(result[word] == count for word, count in expected.items())
    assert len(result) == len(expected)


def test_word_occurrence_multiple_spaces():
    """Extra whitespace between words must not create empty entries."""
    result = word_occurrence("Two  spaces")
    assert dict(result) == {"Two": 1, "spaces": 1}
    assert "" not in result


def test_word_occurrence_leading_and_trailing_whitespace():
    """Leading/trailing whitespace should be ignored."""
    result = word_occurrence("   alpha beta   ")
    assert dict(result) == {"alpha": 1, "beta": 1}


def test_word_occurrence_tabs_and_newlines():
    """Any whitespace (tabs, newlines) should act as a separator."""
    result = word_occurrence("one\ttwo\nthree  one")
    assert dict(result) == {"one": 2, "two": 1, "three": 1}


@pytest.mark.parametrize(
    "sentence",
    ["", " ", "   ", "\t", "\n", " \t\n "],
)
def test_word_occurrence_empty_or_whitespace_only(sentence):
    """Empty or whitespace-only input should yield an empty mapping."""
    result = word_occurrence(sentence)
    assert dict(result) == {}
    assert len(result) == 0


def test_word_occurrence_missing_key_returns_zero():
    """Accessing an absent word on the defaultdict should return 0."""
    result = word_occurrence("only one")
    assert result["absent"] == 0
    # Accessing a missing key inserts it with value 0 (defaultdict semantics).
    assert "absent" in result


def test_word_occurrence_single_word():
    """A single word should be counted once."""
    result = word_occurrence("word")
    assert dict(result) == {"word": 1}


def test_word_occurrence_punctuation_kept_as_part_of_word():
    """Punctuation is not stripped; it stays attached to the token."""
    result = word_occurrence("hi, hi. hi")
    assert dict(result) == {"hi,": 1, "hi.": 1, "hi": 1}


def test_word_occurrence_numeric_tokens():
    """Numeric tokens are treated as ordinary words."""
    result = word_occurrence("1 2 2 3 3 3")
    assert dict(result) == {"1": 1, "2": 2, "3": 3}


def test_word_occurrence_does_not_mutate_input():
    """The input string must remain unchanged after the call."""
    sentence = "keep me intact"
    original = str(sentence)
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_large_input():
    """A large repeated input should be counted correctly."""
    sentence = " ".join(["word"] * 1000 + ["other"] * 500)
    result = word_occurrence(sentence)
    assert result["word"] == 1000
    assert result["other"] == 500
    assert len(result) == 2


def test_word_occurrence_unicode_words():
    """Unicode words should be handled without error."""
    result = word_occurrence("café café naïve")
    assert dict(result) == {"café": 2, "naïve": 1}