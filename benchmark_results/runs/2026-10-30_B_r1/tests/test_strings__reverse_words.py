import pytest

from reverse_words import reverse_words


class TestReverseWords:
    """Tests for the reverse_words function."""

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I love Python", "Python love I"),
            ("Hello World", "World Hello"),
            ("one", "one"),
            ("a b c d e", "e d c b a"),
            ("The quick brown fox", "fox brown quick The"),
        ],
    )
    def test_reverse_words_happy_path(self, sentence, expected):
        """Test reversing the order of words for normal inputs."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("I     Love          Python", "Python Love I"),
            ("  leading and trailing  ", "trailing and leading"),
            ("\tTab\tSeparated\tWords", "Words Separated Tab"),
            ("\nNew\nLine\nWords", "Words Line New"),
            ("multiple   spaces   between   words", "words between spaces multiple"),
        ],
    )
    def test_reverse_words_extra_whitespace(self, sentence, expected):
        """Test that extra whitespace (spaces, tabs, newlines) is collapsed."""
        assert reverse_words(sentence) == expected

    @pytest.mark.parametrize(
        "sentence",
        [
            "",
            "   ",
            "\t\t\t",
            "\n\n\n",
            " \t \n ",
        ],
    )
    def test_reverse_words_empty_or_whitespace_only(self, sentence):
        """Test that empty or whitespace-only strings return an empty string."""
        assert reverse_words(sentence) == ""

    def test_reverse_words_single_word_with_surrounding_whitespace(self):
        """Test a single word surrounded by whitespace returns just the word."""
        assert reverse_words("   hello   ") == "hello"

    def test_reverse_words_preserves_word_characters(self):
        """Test that punctuation and case within words are preserved."""
        assert reverse_words("Hello, World!") == "World! Hello,"
        assert reverse_words("UPPER lower MiXeD") == "MiXeD lower UPPER"

    def test_reverse_words_unicode(self):
        """Test that unicode words are handled correctly."""
        assert reverse_words("héllo wörld") == "wörld héllo"
        assert reverse_words("日本語 の テスト") == "テスト の 日本語"