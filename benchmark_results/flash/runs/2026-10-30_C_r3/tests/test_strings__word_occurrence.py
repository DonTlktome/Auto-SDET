"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


def test_word_occurrence_happy_path():
    """Counts each word in a simple sentence correctly."""
    result = word_occurrence("the quick brown fox jumps over the lazy dog the")
    assert result["the"] == 3
    assert result["quick"] == 1
    assert result["fox"] == 1
    assert len(result) == 8


def test_word_occurrence_returns_defaultdict():
    """The returned object is a defaultdict(int) so missing keys yield 0."""
    result = word_occurrence("a b c")
    assert isinstance(result, defaultdict)
    assert result["missing"] == 0
    assert "missing" in result  # defaultdict inserts the default on access


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case are counted separately."""
    result = word_occurrence("a b A b c b d b d e f e g e h e i e j e 0")
    assert result["b"] == 4
    assert result["e"] == 6
    assert result["A"] == 1
    assert result["a"] == 1


def test_word_occurrence_matches_counter():
    """Result agrees with collections.Counter over the same split tokens."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert all(result[word] == count for word, count in expected.items())
    assert len(result) == len(expected)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n  \r\n", {}),
        ("single", {"single": 1}),
        ("Two  spaces", {"Two": 1, "spaces": 1}),
        ("a a a", {"a": 3}),
        ("x y z", {"x": 1, "y": 1, "z": 1}),
        ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
        ("line1\nline2\tline3", {"line1": 1, "line2": 1, "line3": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Various inputs (empty, whitespace-only, mixed separators) produce expected counts."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_empty_string_returns_empty_dict():
    """An empty string yields an empty mapping."""
    result = word_occurrence("")
    assert dict(result) == {}
    assert len(result) == 0


def test_word_occurrence_whitespace_only_returns_empty_dict():
    """A whitespace-only string yields an empty mapping (split() drops it)."""
    result = word_occurrence(" \t\n\r ")
    assert dict(result) == {}


def test_word_occurrence_punctuation_is_part_of_word():
    """Punctuation is not stripped; tokens are split on whitespace only."""
    result = word_occurrence("hello, hello. hello!")
    assert dict(result) == {"hello,": 1, "hello.": 1, "hello!": 1}


def test_word_occurrence_large_repetition():
    """A word repeated many times is counted exactly."""
    sentence = " ".join(["word"] * 1000)
    result = word_occurrence(sentence)
    assert result["word"] == 1000
    assert len(result) == 1


def test_word_occurrence_does_not_mutate_input():
    """The input string is unchanged after the call."""
    sentence = "alpha beta alpha"
    original = sentence
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_unicode_words():
    """Non-ASCII words are counted correctly."""
    result = word_occurrence("café café naïve 日本語 日本語 日本語")
    assert dict(result) == {"café": 2, "naïve": 1, "日本語": 3}


def test_word_occurrence_single_character_words():
    """Single-character tokens are handled."""
    result = word_occurrence("a b a c a")
    assert dict(result) == {"a": 3, "b": 1, "c": 1}