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
    """Happy path: counts each word in a simple sentence."""
    result = word_occurrence("the quick brown fox jumps over the lazy dog the")
    assert result["the"] == 3
    assert result["quick"] == 1
    assert result["dog"] == 1
    assert len(result) == 8


def test_word_occurrence_matches_counter():
    """Counts must agree with collections.Counter over the same split."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert all(result[word] == count for word, count in expected.items())
    assert len(result) == len(expected)


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case are counted separately."""
    result = word_occurrence("a A a A A")
    assert result["a"] == 2
    assert result["A"] == 3


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n  \r\n", {}),
        ("single", {"single": 1}),
        ("Two  spaces", {"Two": 1, "spaces": 1}),
        ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
        ("x x x x", {"x": 4}),
        ("one\ntwo\tthree", {"one": 1, "two": 1, "three": 1}),
        ("0 0 1", {"0": 2, "1": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Various whitespace layouts and inputs produce the expected counts."""
    result = word_occurrence(sentence)
    assert dict(result) == expected


def test_word_occurrence_empty_string_returns_empty_dict():
    """Edge case: an empty string yields an empty mapping."""
    result = word_occurrence("")
    assert len(result) == 0
    assert dict(result) == {}


def test_word_occurrence_missing_key_returns_zero():
    """Accessing an unseen word on the defaultdict returns 0 without KeyError."""
    result = word_occurrence("alpha beta")
    assert result["gamma"] == 0
    # Accessing a missing key inserts it with the default value.
    assert "gamma" in result


def test_word_occurrence_does_not_mutate_input():
    """The input string is not modified (strings are immutable, sanity check)."""
    sentence = "keep me intact"
    original = sentence
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_punctuation_attached_to_words():
    """Punctuation is not stripped; it stays part of the token."""
    result = word_occurrence("hi, hi. hi")
    assert result["hi,"] == 1
    assert result["hi."] == 1
    assert result["hi"] == 1


def test_word_occurrence_large_repetition():
    """Boundary: a large number of repetitions is counted correctly."""
    n = 10_000
    result = word_occurrence(" ".join(["word"] * n))
    assert result["word"] == n
    assert len(result) == 1


def test_word_occurrence_unicode_words():
    """Unicode words are counted as distinct tokens."""
    result = word_occurrence("café café naïve 日本語 日本語 日本語")
    assert result["café"] == 2
    assert result["naïve"] == 1
    assert result["日本語"] == 3