import pytest

from reverse_words import reverse_words


class TestReverseWords:
    """Tests for the reverse_words function."""

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I love Python", "Python love I"),
            ("Hello World", "World Hello"),
            ("one two three four", "four three two one"),
            ("single", "single"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Reverse the order of words for normal, well-formed sentences."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I     Love          Python", "Python Love I"),
            ("  leading and trailing  ", "trailing and leading"),
            ("a\tb\nc", "c b a"),
            ("   ", ""),
            ("", ""),
        ],
    )
    def test_reverse_words_whitespace_handling(self, sentence, expected):
        """Collapse extra whitespace and handle empty/whitespace-only input."""
        assert reverse_words(sentence) == expected

    def test_reverse_words_preserves_word_characters(self):
        """Ensure punctuation and case within words are preserved."""
        assert reverse_words("Hello, World!") == "World! Hello,"

    def test_reverse_words_returns_string(self):
        """The function must always return a str instance."""
        result = reverse_words("some words here")
        assert isinstance(result, str)