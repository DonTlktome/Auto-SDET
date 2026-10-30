"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


def test_word_occurrence_returns_defaultdict():
    """The function must return a defaultdict(int) instance."""
    result = word_occurrence("hello world")
    assert isinstance(result, defaultdict)
    assert result.default_factory is int


def test_word_occurrence_happy_path_counts():
    """Counts each whitespace-separated token, case-sensitively."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    result = word_occurrence(sentence)
    expected = Counter(sentence.split())
    assert dict(result) == dict(expected)
    assert result["b"] == 4
    assert result["e"] == 6
    assert result["A"] == 1
    assert result["a"] == 1


def test_word_occurrence_missing_key_returns_zero():
    """Accessing an absent word on the returned defaultdict yields 0."""
    result = word_occurrence("one two two")
    assert result["three"] == 0
    # The lookup must not raise and must not corrupt existing counts.
    assert result["two"] == 2


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n  \r\n", {}),
        ("single", {"single": 1}),
        ("Two  spaces", {"Two": 1, "spaces": 1}),
        ("a a a", {"a": 3}),
        ("x y x y", {"x": 2, "y": 2}),
        ("Hello hello HELLO", {"Hello": 1, "hello": 1, "HELLO": 1}),
        ("line1\nline2\tline3", {"line1": 1, "line2": 1, "line3": 1}),
        ("0 0 1", {"0": 2, "1": 1}),
        ("punct , . !", {"punct": 1, ",": 1, ".": 1, "!": 1}),
    ],
)
def test_word_occurrence_parametrized(sentence, expected):
    """Various inputs, including empty/whitespace-only and punctuation cases."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_empty_string():
    """An empty string produces an empty mapping."""
    result = word_occurrence("")
    assert len(result) == 0
    assert dict(result) == {}


def test_word_occurrence_whitespace_only():
    """Whitespace-only input produces an empty mapping (split yields nothing)."""
    result = word_occurrence(" \t\n\r\v\f ")
    assert dict(result) == {}


def test_word_occurrence_preserves_insertion_order():
    """Keys appear in first-occurrence order."""
    result = word_occurrence("c a b a c")
    assert list(result.keys()) == ["c", "a", "b"]
    assert list(result.values()) == [2, 2, 1]


def test_word_occurrence_does_not_mutate_input():
    """The input string is immutable and must be returned unchanged."""
    sentence = "keep me intact"
    original = str(sentence)
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_large_input():
    """A large repeated input is counted correctly."""
    sentence = " ".join(["word"] * 1000 + ["other"] * 500)
    result = word_occurrence(sentence)
    assert result["word"] == 1000
    assert result["other"] == 500
    assert len(result) == 2


def test_word_occurrence_matches_counter_property():
    """Counts always match collections.Counter over the same split tokens."""
    sentence = "the quick brown fox jumps over the lazy dog the end"
    result = word_occurrence(sentence)
    counter = Counter(sentence.split())
    assert all(result[word] == count for word, count in counter.items())
    assert sum(result.values()) == len(sentence.split())


def test_word_occurrence_unicode_words():
    """Non-ASCII words are handled as ordinary tokens."""
    result = word_occurrence("café café naïve")
    assert dict(result) == {"café": 2, "naïve": 1}


def test_word_occurrence_non_string_raises():
    """Passing a non-string (no .split) raises AttributeError."""
    with pytest.raises(AttributeError):
        word_occurrence(None)


def test_word_occurrence_none_raises_type_error_or_attribute_error():
    """Passing an int raises AttributeError since ints have no .split."""
    with pytest.raises(AttributeError):
        word_occurrence(123)