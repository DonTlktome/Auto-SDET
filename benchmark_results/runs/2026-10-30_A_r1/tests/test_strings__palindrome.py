import pytest

from palindrome import (
    is_palindrome,
    is_palindrome_recursive,
    is_palindrome_slice,
    is_palindrome_traversal,
    test_data,
)


# Shared test inputs for all four implementations.
PALINDROMES = [
    "MALAYALAM",
    "rotor",
    "level",
    "A",
    "BB",
    "amanaplanacanalpanama",
]

NON_PALINDROMES = [
    "String",
    "ABC",
    "abcdba",
    "AB",
]


@pytest.mark.parametrize("s", PALINDROMES)
def test_is_palindrome_positive(s):
    """is_palindrome returns True for known palindromes."""
    assert is_palindrome(s) is True


@pytest.mark.parametrize("s", NON_PALINDROMES)
def test_is_palindrome_negative(s):
    """is_palindrome returns False for known non-palindromes."""
    assert is_palindrome(s) is False


def test_is_palindrome_empty_string():
    """is_palindrome treats the empty string as a palindrome (edge case)."""
    assert is_palindrome("") is True


def test_is_palindrome_single_char():
    """is_palindrome handles a single-character string (boundary case)."""
    assert is_palindrome("z") is True


def test_is_palindrome_two_char_equal():
    """is_palindrome handles a two-character palindrome (boundary case)."""
    assert is_palindrome("aa") is True


def test_is_palindrome_two_char_unequal():
    """is_palindrome handles a two-character non-palindrome (boundary case)."""
    assert is_palindrome("ab") is False


@pytest.mark.parametrize("s", PALINDROMES)
def test_is_palindrome_traversal_positive(s):
    """is_palindrome_traversal returns True for known palindromes."""
    assert is_palindrome_traversal(s) is True


@pytest.mark.parametrize("s", NON_PALINDROMES)
def test_is_palindrome_traversal_negative(s):
    """is_palindrome_traversal returns False for known non-palindromes."""
    assert is_palindrome_traversal(s) is False


def test_is_palindrome_traversal_empty_string():
    """is_palindrome_traversal treats the empty string as a palindrome."""
    assert is_palindrome_traversal("") is True


def test_is_palindrome_traversal_single_char():
    """is_palindrome_traversal handles a single-character string."""
    assert is_palindrome_traversal("z") is True


@pytest.mark.parametrize("s", PALINDROMES)
def test_is_palindrome_recursive_positive(s):
    """is_palindrome_recursive returns True for known palindromes."""
    assert is_palindrome_recursive(s) is True


@pytest.mark.parametrize("s", NON_PALINDROMES)
def test_is_palindrome_recursive_negative(s):
    """is_palindrome_recursive returns False for known non-palindromes."""
    assert is_palindrome_recursive(s) is False


def test_is_palindrome_recursive_empty_string():
    """is_palindrome_recursive treats the empty string as a palindrome."""
    assert is_palindrome_recursive("") is True


def test_is_palindrome_recursive_single_char():
    """is_palindrome_recursive handles a single-character string."""
    assert is_palindrome_recursive("z") is True


@pytest.mark.parametrize("s", PALINDROMES)
def test_is_palindrome_slice_positive(s):
    """is_palindrome_slice returns True for known palindromes."""
    assert is_palindrome_slice(s) is True


@pytest.mark.parametrize("s", NON_PALINDROMES)
def test_is_palindrome_slice_negative(s):
    """is_palindrome_slice returns False for known non-palindromes."""
    assert is_palindrome_slice(s) is False


def test_is_palindrome_slice_empty_string():
    """is_palindrome_slice treats the empty string as a palindrome."""
    assert is_palindrome_slice("") is True


def test_is_palindrome_slice_single_char():
    """is_palindrome_slice handles a single-character string."""
    assert is_palindrome_slice("z") is True


@pytest.mark.parametrize("key, expected", test_data.items())
def test_all_implementations_agree_with_test_data(key, expected):
    """All four implementations agree with the module's test_data mapping."""
    assert is_palindrome(key) == expected
    assert is_palindrome_traversal(key) == expected
    assert is_palindrome_recursive(key) == expected
    assert is_palindrome_slice(key) == expected


@pytest.mark.parametrize("s", PALINDROMES + NON_PALINDROMES + ["", "z", "aa", "ab"])
def test_implementations_are_consistent(s):
    """All four implementations return the same result for a given input."""
    results = {
        is_palindrome(s),
        is_palindrome_traversal(s),
        is_palindrome_recursive(s),
        is_palindrome_slice(s),
    }
    assert len(results) == 1