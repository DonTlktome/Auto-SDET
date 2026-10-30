"""Tests for strings/word_occurrence.py."""

from collections import Counter, defaultdict

import pytest

from strings.word_occurrence import word_occurrence


# ---------------------------------------------------------------------------
# Positive / happy-path tests
# ---------------------------------------------------------------------------


def test_word_occurrence_basic_counts():
    """A simple sentence yields the expected per-word counts."""
    result = word_occurrence("the quick brown fox jumps over the lazy dog the")
    assert result["the"] == 3
    assert result["quick"] == 1
    assert result["brown"] == 1
    assert result["fox"] == 1
    assert result["jumps"] == 1
    assert result["over"] == 1
    assert result["lazy"] == 1
    assert result["dog"] == 1


def test_word_occurrence_matches_counter():
    """Counts must agree with collections.Counter for the docstring example."""
    sentence = "a b A b c b d b d e f e g e h e i e j e 0"
    occurrence_dict = word_occurrence(sentence)
    assert all(
        occurrence_dict[word] == count
        for word, count in Counter(sentence.split()).items()
    )


def test_word_occurrence_returns_defaultdict():
    """The returned object is a defaultdict(int) so missing keys default to 0."""
    result = word_occurrence("hello world")
    assert isinstance(result, defaultdict)
    assert result.default_factory is int
    # Accessing an absent key must not raise and must yield 0.
    assert result["not-present"] == 0


def test_word_occurrence_is_case_sensitive():
    """Words differing only in case are counted separately."""
    result = word_occurrence("Word word WORD wOrD")
    assert result["Word"] == 1
    assert result["word"] == 1
    assert result["WORD"] == 1
    assert result["wOrD"] == 1
    assert len(result) == 4


def test_word_occurrence_preserves_insertion_order():
    """Keys appear in first-seen order (dict semantics of defaultdict)."""
    result = word_occurrence("b a c a b")
    assert list(result.keys()) == ["b", "a", "c"]
    assert list(result.values()) == [2, 2, 1]


# ---------------------------------------------------------------------------
# Edge cases / boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("", {}),
        ("   ", {}),
        ("\t\n\r\n", {}),
        ("\n", {}),
    ],
)
def test_word_occurrence_empty_or_whitespace_only(sentence, expected):
    """Empty and whitespace-only inputs produce an empty mapping."""
    result = word_occurrence(sentence)
    assert dict(result) == expected
    assert len(result) == 0


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Two  spaces", {"Two": 1, "spaces": 1}),
        ("  leading", {"leading": 1}),
        ("trailing  ", {"trailing": 1}),
        ("a\tb\nc\rd", {"a": 1, "b": 1, "c": 1, "d": 1}),
        ("a   b   a", {"a": 2, "b": 1}),
    ],
)
def test_word_occurrence_whitespace_handling(sentence, expected):
    """Any run of whitespace acts as a single separator (str.split semantics)."""
    assert dict(word_occurrence(sentence)) == expected


def test_word_occurrence_single_word():
    """A single word with no separators is counted once."""
    assert dict(word_occurrence("hello")) == {"hello": 1}


def test_word_occurrence_single_repeated_word():
    """A single word repeated many times is counted correctly."""
    assert dict(word_occurrence("x x x x x")) == {"x": 5}


def test_word_occurrence_punctuation_is_part_of_word():
    """Punctuation is not stripped; it stays attached to the token."""
    result = word_occurrence("hello, hello. hello!")
    assert result["hello,"] == 1
    assert result["hello."] == 1
    assert result["hello!"] == 1
    assert "hello" not in result


def test_word_occurrence_unicode_words():
    """Non-ASCII words are handled as ordinary tokens."""
    result = word_occurrence("café café naïve 日本語 日本語 日本語")
    assert result["café"] == 2
    assert result["naïve"] == 1
    assert result["日本語"] == 3


def test_word_occurrence_large_input():
    """A large input is counted correctly (boundary/scale check)."""
    n = 1000
    sentence = " ".join(["word"] * n + ["other"] * (n // 2))
    result = word_occurrence(sentence)
    assert result["word"] == n
    assert result["other"] == n // 2
    assert len(result) == 2


def test_word_occurrence_does_not_mutate_input():
    """The input string is immutable and must be returned unchanged."""
    sentence = "alpha beta alpha"
    original = str(sentence)
    word_occurrence(sentence)
    assert sentence == original


def test_word_occurrence_returns_new_object_each_call():
    """Each call returns an independent mapping (no shared state)."""
    first = word_occurrence("a b")
    second = word_occurrence("a b")
    assert first is not second
    first["a"] = 99
    assert second["a"] == 1


def test_word_occurrence_non_string_raises_attribute_error():
    """Passing a non-string (no .split) raises AttributeError."""
    with pytest.raises(AttributeError):
        word_occurrence(12345)


def test_word_occurrence_none_raises_attribute_error():
    """Passing None raises AttributeError since None has no .split."""
    with pytest.raises(AttributeError):
        word_occurrence(None)