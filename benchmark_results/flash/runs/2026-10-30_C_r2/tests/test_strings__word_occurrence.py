"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


def test_word_occurrence_happy_path():
    """Basic sentence: each word counted once, order preserved."""
    result = word_occurrence("the quick brown fox")
    assert result == {"the": 1, "quick": 1, "brown": 1, "fox": 1}


def test_word_occurrence_returns_defaultdict():
    """The function returns a defaultdict(int), not a plain dict."""
    result = word_occurrence("a b a")
    assert isinstance(result, defaultdict)
    assert result.default_factory is int


def test_word_occurrence_missing_key_returns_zero():
    """Accessing an absent key on the returned defaultdict yields 0."""
    result = word_occurrence("a b a")
    assert result["nonexistent"] == 0


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case are counted separately."""
    result = word_occurrence("a b A b c b d b d e f e g e h e i e j e 0")
    assert result["b"] == 4
    assert result["e"] == 6
    assert result["A"] == 1
    assert result["a"] == 1


def test_word_occurrence_matches_counter():
    """Counts agree with collections.Counter over the same split."""
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
        ("x y x y x", {"x": 3, "y": 2}),
        ("one\ntwo\tthree", {"one": 1, "two": 1, "three": 1}),
        ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
        ("0 1 2 0", {"0": 2, "1": 1, "2": 1}),
        ("!!! ??? !!!", {"!!!": 2, "???": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Various inputs, including whitespace-only and punctuation-only cases."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_empty_string():
    """An empty string produces an empty mapping."""
    result = word_occurrence("")
    assert len(result) == 0
    assert dict(result) == {}


def test_word_occurrence_whitespace_only():
    """Whitespace-only input produces an empty mapping."""
    result = word_occurrence(" \t\n\r ")
    assert dict(result) == {}


def test_word_occurrence_single_word():
    """A single word is counted exactly once."""
    assert dict(word_occurrence("hello")) == {"hello": 1}


def test_word_occurrence_repeated_single_word():
    """A repeated single word is counted correctly."""
    assert dict(word_occurrence("spam " * 100)) == {"spam": 100}


def test_word_occurrence_does_not_mutate_input():
    """The input string is unchanged after the call."""
    sentence = "keep me intact"
    original = str(sentence)
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_total_counts_equal_word_count():
    """Sum of all counts equals the number of whitespace-separated tokens."""
    sentence = "a b c a b a"
    result = word_occurrence(sentence)
    assert sum(result.values()) == len(sentence.split())


def test_word_occurrence_large_input():
    """A large input is counted correctly (boundary/performance sanity check)."""
    n = 5000
    sentence = " ".join(["word"] * n + ["other"] * (n // 2))
    result = word_occurrence(sentence)
    assert result["word"] == n
    assert result["other"] == n // 2
    assert len(result) == 2


def test_word_occurrence_unicode_words():
    """Non-ASCII words are handled as ordinary tokens."""
    result = word_occurrence("héllo wörld héllo")
    assert dict(result) == {"héllo": 2, "wörld": 1}


def test_word_occurrence_punctuation_attached_to_words():
    """Punctuation is not stripped; tokens are split on whitespace only."""
    result = word_occurrence("hello, hello. hello")
    assert dict(result) == {"hello,": 1, "hello.": 1, "hello": 1}


def test_word_occurrence_non_string_raises_attribute_error():
    """Passing a non-string (no .split) raises AttributeError."""
    with pytest.raises(AttributeError):
        word_occurrence(None)


def test_word_occurrence_none_raises():
    """Explicit None input raises AttributeError rather than silently passing."""
    with pytest.raises(AttributeError):
        word_occurrence(None)  # type: ignore[arg-type]