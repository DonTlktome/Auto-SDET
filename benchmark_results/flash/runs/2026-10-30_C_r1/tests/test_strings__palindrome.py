"""Tests for strings/palindrome.py — palindrome detection algorithms."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Locate the module under test on disk.  The test file may live either at the
# repository root or inside a 'tests' sub-directory, so search upwards from
# this file instead of hard-coding a parent level.
# ---------------------------------------------------------------------------
def _find_source() -> Path:
    here = Path(__file__).resolve().parent
    search_roots = [here, *list(here.parents)[:4]]

    # Fast path: the usual layouts (<root>/strings/palindrome.py).
    for root in search_roots:
        for candidate in (root / "strings" / "palindrome.py", root / "palindrome.py"):
            if candidate.is_file():
                return candidate

    # Fallback: walk the tree (never walk the filesystem root itself).
    for root in search_roots:
        if root == Path(root.anchor):
            continue
        for candidate in sorted(root.glob("**/palindrome.py")):
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"could not locate strings/palindrome.py starting from {here}"
    )


MODULE_PATH = _find_source()


def _load_module():
    spec = importlib.util.spec_from_file_location("palindrome_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


palindrome = _load_module()

is_palindrome = palindrome.is_palindrome
is_palindrome_traversal = palindrome.is_palindrome_traversal
is_palindrome_recursive = palindrome.is_palindrome_recursive
is_palindrome_slice = palindrome.is_palindrome_slice
benchmark_function = palindrome.benchmark_function
test_data = palindrome.test_data

# benchmark_function builds its timeit setup string as
# "from __main__ import test_data, <name>".  Expose the names on __main__ so
# that this import resolves when the real timeit path is exercised.
_MAIN_MODULE = sys.modules.get("__main__")
if _MAIN_MODULE is not None:
    for _name in (
        "test_data",
        "is_palindrome",
        "is_palindrome_traversal",
        "is_palindrome_recursive",
        "is_palindrome_slice",
    ):
        setattr(_MAIN_MODULE, _name, getattr(palindrome, _name))

ALL_IMPLS = [
    is_palindrome,
    is_palindrome_traversal,
    is_palindrome_recursive,
    is_palindrome_slice,
]


# ---------------------------------------------------------------------------
# test_data sanity
# ---------------------------------------------------------------------------
def test_test_data_is_consistent_with_python_slicing():
    """The module's fixture data must agree with the canonical s == s[::-1] check."""
    assert all((key == key[::-1]) == value for key, value in test_data.items())


def test_test_data_contains_both_true_and_false_cases():
    """The fixture data must exercise both outcomes, otherwise tests are weak."""
    assert any(test_data.values())
    assert not all(test_data.values())


# ---------------------------------------------------------------------------
# is_palindrome (two-pointer)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_matches_fixture_data(s, expected):
    """Two-pointer implementation agrees with the module's fixture expectations."""
    assert is_palindrome(s) is expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", True),          # empty string boundary
        ("a", True),         # single character boundary
        ("aa", True),        # even length, minimal
        ("ab", False),       # even length, minimal mismatch
        ("aba", True),       # odd length, minimal
        ("abc", False),      # odd length, minimal mismatch
        ("abca", False),     # mismatch only at the outer pair
        ("abccba", True),    # even palindrome
        ("abccbx", False),   # even length, mismatch at last pair
        ("  ", True),        # whitespace is not stripped
        ("a a", True),       # inner space mirrored
        ("a b", False),      # inner space not mirrored
        ("12321", True),     # digits
        ("12345", False),    # digits, non-palindrome
        ("!@#$%$#@!", True), # punctuation
    ],
)
def test_is_palindrome_edge_cases(s, expected):
    """Two-pointer implementation handles empty, tiny, and non-alphanumeric inputs."""
    assert is_palindrome(s) is expected


def test_is_palindrome_returns_bool_type():
    """The return value must be a real bool, not a truthy/falsy object."""
    assert isinstance(is_palindrome("rotor"), bool)
    assert isinstance(is_palindrome("rotorx"), bool)


def test_is_palindrome_long_string():
    """A long palindrome (10k chars) is detected without recursion limits."""
    half = "abcdefghij" * 500
    # half + half[::-1] is symmetric and therefore a palindrome.
    assert is_palindrome(half + half[::-1]) is True
    # Trailing an extra character onto a palindrome breaks symmetry.
    assert is_palindrome(half + half[::-1] + "X") is False


# ---------------------------------------------------------------------------
# is_palindrome_traversal (half traversal)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_traversal_matches_fixture_data(s, expected):
    """Half-traversal implementation agrees with the module's fixture expectations."""
    assert is_palindrome_traversal(s) is expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("ab", False),
        ("aba", True),
        ("abc", False),
        ("abca", False),
        ("abccba", True),
        ("abccbx", False),
        ("  ", True),
        ("a a", True),
        ("a b", False),
        ("12321", True),
        ("12345", False),
        ("!@#$%$#@!", True),
    ],
)
def test_is_palindrome_traversal_edge_cases(s, expected):
    """Half-traversal implementation handles empty, tiny, and non-alphanumeric inputs."""
    assert is_palindrome_traversal(s) is expected


def test_is_palindrome_traversal_returns_bool_type():
    """`all(...)` returns a bool for both outcomes."""
    assert isinstance(is_palindrome_traversal("level"), bool)
    assert isinstance(is_palindrome_traversal("levels"), bool)


def test_is_palindrome_traversal_odd_length_ignores_middle_char():
    """For odd lengths the middle character is never compared (it is always equal)."""
    assert is_palindrome_traversal("abcba") is True
    assert is_palindrome_traversal("abcxa") is False


# ---------------------------------------------------------------------------
# is_palindrome_recursive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_recursive_matches_fixture_data(s, expected):
    """Recursive implementation agrees with the module's fixture expectations."""
    assert is_palindrome_recursive(s) is expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", True),          # base case: len <= 1
        ("a", True),         # base case: len == 1
        ("aa", True),        # one recursion step
        ("ab", False),       # one recursion step, mismatch
        ("aba", True),
        ("abc", False),
        ("abca", False),
        ("abccba", True),
        ("abccbx", False),
        ("  ", True),
        ("a a", True),
        ("a b", False),
        ("12321", True),
        ("12345", False),
        ("!@#$%$#@!", True),
    ],
)
def test_is_palindrome_recursive_edge_cases(s, expected):
    """Recursive implementation handles base cases and non-alphanumeric inputs."""
    assert is_palindrome_recursive(s) is expected


def test_is_palindrome_recursive_returns_bool_type():
    """The recursive implementation returns a real bool."""
    assert isinstance(is_palindrome_recursive("rotor"), bool)
    assert isinstance(is_palindrome_recursive("rotors"), bool)


def test_is_palindrome_recursive_short_circuits_on_first_mismatch():
    """A mismatch at the outermost pair returns False without deeper recursion."""
    with patch.object(
        palindrome, "is_palindrome_recursive", wraps=palindrome.is_palindrome_recursive
    ) as spy:
        assert palindrome.is_palindrome_recursive("ab" + "c" * 50) is False
    # Only the outermost call should have happened.
    assert spy.call_count == 1


# ---------------------------------------------------------------------------
# is_palindrome_slice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s,expected", list(test_data.items()))
def test_is_palindrome_slice_matches_fixture_data(s, expected):
    """Slice-reversal implementation agrees with the module's fixture expectations."""
    assert is_palindrome_slice(s) is expected


@pytest.mark.parametrize(
    "s,expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("ab", False),
        ("aba", True),
        ("abc", False),
        ("abca", False),
        ("abccba", True),
        ("abccbx", False),
        ("  ", True),
        ("a a", True),
        ("a b", False),
        ("12321", True),
        ("12345", False),
        ("!@#$%$#@!", True),
    ],
)
def test_is_palindrome_slice_edge_cases(s, expected):
    """Slice implementation handles empty, tiny, and non-alphanumeric inputs."""
    assert is_palindrome_slice(s) is expected


def test_is_palindrome_slice_returns_bool_type():
    """`s == s[::-1]` yields a real bool."""
    assert isinstance(is_palindrome_slice("level"), bool)
    assert isinstance(is_palindrome_slice("levels"), bool)


# ---------------------------------------------------------------------------
# Cross-implementation consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "s",
    [
        "",
        "a",
        "ab",
        "aba",
        "abc",
        "abca",
        "abccba",
        "abccbx",
        "MALAYALAM",
        "String",
        "amanaplanacanalpanama",
        "abcdba",
        "  ",
        "a a",
        "a b",
        "12321",
        "12345",
        "!@#$%$#@!",
        "x" * 101,
        "x" * 100 + "y",
    ],
)
def test_all_implementations_agree(s):
    """Every implementation must produce identical results for the same input."""
    results = {impl.__name__: impl(s) for impl in ALL_IMPLS}
    assert len(set(results.values())) == 1, results


# ---------------------------------------------------------------------------
# benchmark_function
# ---------------------------------------------------------------------------
def test_benchmark_function_prints_expected_format(capsys):
    """benchmark_function prints the function name and run count in the expected format."""
    fake_timeit = MagicMock(return_value=0.12345)
    with patch.object(palindrome, "timeit", fake_timeit):
        benchmark_function("is_palindrome_slice")

    captured = capsys.readouterr()
    assert "is_palindrome_slice" in captured.out
    assert "500,000" in captured.out
    assert "0.12345" in captured.out
    assert captured.out.endswith("seconds\n")


def test_benchmark_function_builds_timeit_arguments():
    """benchmark_function passes the correct stmt/setup/number to timeit."""
    fake_timeit = MagicMock(return_value=0.0)
    with patch.object(palindrome, "timeit", fake_timeit):
        benchmark_function("is_palindrome")

    fake_timeit.assert_called_once()
    kwargs = fake_timeit.call_args.kwargs
    assert kwargs["number"] == 500000
    assert "is_palindrome(key) == value" in kwargs["stmt"]
    assert "test_data" in kwargs["stmt"]
    assert kwargs["setup"] == "from __main__ import test_data, is_palindrome"


def test_benchmark_function_returns_none(capsys):
    """benchmark_function is a side-effect-only helper and returns None."""
    with patch.object(palindrome, "timeit", MagicMock(return_value=0.0)):
        assert benchmark_function("is_palindrome_slice") is None
    capsys.readouterr()


def test_benchmark_function_uses_real_timeit_without_network(capsys):
    """The real timeit path works offline and prints a numeric duration."""
    with patch.object(palindrome, "timeit", wraps=palindrome.timeit) as spy:
        # Patch the module-level name so the setup string import still resolves.
        benchmark_function("is_palindrome_slice")
    capsys.readouterr()
    assert spy.call_count == 1
