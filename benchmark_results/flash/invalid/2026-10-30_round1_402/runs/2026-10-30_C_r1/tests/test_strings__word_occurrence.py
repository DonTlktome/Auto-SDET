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
    """Words differing only in case are treated as distinct keys."""
    result = word_occurrence("a A a A b")
    assert dict(result) == {"a": 2, "A": 2, "b": 1}


def test_word_occurrence_matches_counter():
    """Results must agree with collections.Counter over split()."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert all(result[word] == count for word, count in expected.items())
    assert dict(result) == dict(expected)


def test_word_occurrence_multiple_spaces():
    """Extra whitespace between words must not create empty keys."""
    result = word_occurrence("Two  spaces")
    assert dict(result) == {"Two": 1, "spaces": 1}
    assert "" not in result


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n  \r\n", {}),
        ("single", {"single": 1}),
        ("  padded  ", {"padded": 1}),
        ("a\tb\nc", {"a": 1, "b": 1, "c": 1}),
        ("x x x", {"x": 3}),
        ("1 2 3 1", {"1": 2, "2": 1, "3": 1}),
        ("hello, hello, world!", {"hello,": 2, "world!": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Various inputs, including empty/whitespace-only strings, are handled."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_empty_string_returns_empty_dict():
    """An empty string yields an empty mapping (no keys at all)."""
    result = word_occurrence("")
    assert len(result) == 0
    assert dict(result) == {}


def test_word_occurrence_missing_key_defaults_to_zero():
    """Accessing an absent key on the returned defaultdict yields 0."""
    result = word_occurrence("alpha beta")
    assert result["gamma"] == 0
    # Accessing a missing key inserts it with the default value.
    assert "gamma" in result


def test_word_occurrence_does_not_mutate_input():
    """The input string is immutable, but verify the call is repeatable."""
    sentence = "repeat repeat once"
    first = dict(word_occurrence(sentence))
    second = dict(word_occurrence(sentence))
    assert first == second == {"repeat": 2, "once": 1}


def test_word_occurrence_large_input():
    """A large input is counted correctly (boundary/scale check)."""
    sentence = " ".join(["word"] * 1000 + ["other"] * 500)
    result = word_occurrence(sentence)
    assert result["word"] == 1000
    assert result["other"] == 500
    assert len(result) == 2


def test_word_occurrence_punctuation_preserved():
    """Punctuation is not stripped; tokens are split on whitespace only."""
    result = word_occurrence("end. end. end!")
    assert dict(result) == {"end.": 2, "end!": 1}


def test_word_occurrence_unicode_words():
    """Non-ASCII words are counted as ordinary tokens."""
    result = word_occurrence("café café naïve")
    assert dict(result) == {"café": 2, "naïve": 1}