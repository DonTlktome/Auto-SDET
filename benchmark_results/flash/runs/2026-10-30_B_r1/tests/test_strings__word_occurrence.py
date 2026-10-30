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
    """Happy path: counts each distinct word in a simple sentence."""
    result = word_occurrence("the quick brown fox")
    assert dict(result) == {"the": 1, "quick": 1, "brown": 1, "fox": 1}


def test_word_occurrence_repeated_words():
    """Happy path: repeated words are counted correctly."""
    result = word_occurrence("a b a b a")
    assert dict(result) == {"a": 3, "b": 2}


def test_word_occurrence_matches_counter():
    """Result must agree with collections.Counter over the same split."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert all(result[word] == count for word, count in expected.items())
    assert len(result) == len(expected)


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case are treated as distinct keys."""
    result = word_occurrence("Word word WORD")
    assert dict(result) == {"Word": 1, "word": 1, "WORD": 1}


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n  \r\n", {}),
        ("single", {"single": 1}),
        ("Two  spaces", {"Two": 1, "spaces": 1}),
        ("  leading and trailing  ", {"leading": 1, "and": 1, "trailing": 1}),
        ("tabs\tand\nnewlines", {"tabs": 1, "and": 1, "newlines": 1}),
        ("0 0 0", {"0": 3}),
        ("!!! ??? !!!", {"!!!": 2, "???": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Edge cases: empty/whitespace-only input, odd spacing, punctuation, digits."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_empty_string_returns_empty_dict():
    """Boundary: an empty string yields an empty mapping."""
    result = word_occurrence("")
    assert len(result) == 0
    assert dict(result) == {}


def test_word_occurrence_missing_key_returns_zero():
    """Accessing an absent word on the defaultdict returns 0 without raising."""
    result = word_occurrence("only these words")
    assert result["absent"] == 0
    # Accessing a missing key inserts it with the default value.
    assert "absent" in result


def test_word_occurrence_does_not_mutate_input():
    """The input string is immutable, but verify the same value is reusable."""
    sentence = "repeat repeat"
    first = word_occurrence(sentence)
    second = word_occurrence(sentence)
    assert dict(first) == dict(second) == {"repeat": 2}
    assert first is not second


def test_word_occurrence_large_input():
    """Scale check: a long repeated input is counted correctly."""
    sentence = " ".join(["word"] * 1000 + ["other"] * 500)
    result = word_occurrence(sentence)
    assert result["word"] == 1000
    assert result["other"] == 500
    assert len(result) == 2


def test_word_occurrence_preserves_first_seen_order():
    """Keys are inserted in first-appearance order."""
    result = word_occurrence("c a b a c")
    assert list(result.keys()) == ["c", "a", "b"]
    assert list(result.values()) == [2, 2, 1]


def test_word_occurrence_non_string_raises_attribute_error():
    """Passing a non-string (no .split) raises AttributeError."""
    with pytest.raises(AttributeError):
        word_occurrence(None)