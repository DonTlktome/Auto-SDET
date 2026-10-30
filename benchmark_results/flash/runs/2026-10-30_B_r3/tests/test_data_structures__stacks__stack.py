"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test in a way that works both as a package module
# and as a standalone file (the source uses PEP 695 generics syntax).
# ---------------------------------------------------------------------------
_SOURCE_PATH = Path(__file__).resolve().parent / "stack.py"


def _load_module():
    """Load the target module from its file path, falling back to import."""
    try:  # Preferred: normal package import.
        from data_structures.stacks import stack as module  # type: ignore

        return module
    except Exception:  # pragma: no cover - fallback for isolated layouts
        spec = importlib.util.spec_from_file_location("stack_under_test", _SOURCE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["stack_under_test"] = module
        spec.loader.exec_module(module)
        return module


stack_module = _load_module()

Stack = stack_module.Stack
StackOverflowError = stack_module.StackOverflowError
StackUnderflowError = stack_module.StackUnderflowError


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------
def test_default_limit_is_ten():
    """A freshly constructed Stack should default to a limit of 10."""
    s = Stack()
    assert s.limit == 10
    assert s.stack == []
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_custom_limit_is_stored(limit):
    """The limit passed to the constructor must be stored verbatim."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.size() == 0


# ---------------------------------------------------------------------------
# __bool__ / __str__
# ---------------------------------------------------------------------------
def test_bool_false_when_empty_true_when_populated():
    """__bool__ reflects whether the stack holds any elements."""
    s = Stack(3)
    assert bool(s) is False
    s.push(1)
    assert bool(s) is True
    s.pop()
    assert bool(s) is False


@pytest.mark.parametrize(
    "items, expected",
    [
        ([], "[]"),
        ([1], "[1]"),
        ([1, 2, 3], "[1, 2, 3]"),
        (["a", "b"], "['a', 'b']"),
    ],
)
def test_str_matches_list_representation(items, expected):
    """__str__ should mirror the string form of the internal list."""
    s = Stack(10)
    for item in items:
        s.push(item)
    assert str(s) == expected


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_adds_to_top():
    """push() appends the element to the end (top) of the stack."""
    s = Stack(3)
    s.push(10)
    s.push(20)
    assert s.stack == [10, 20]
    assert s.peek() == 20
    assert s.size() == 2


def test_push_returns_none():
    """push() is documented to return None."""
    s = Stack(1)
    assert s.push(42) is None


def test_push_raises_overflow_when_full():
    """Pushing onto a full stack raises StackOverflowError."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(20)
    # The failed push must not mutate the stack.
    assert s.stack == [10]


def test_push_raises_overflow_on_zero_limit():
    """A stack with limit 0 can never accept an element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)
    assert s.size() == 0


def test_push_accepts_none_and_mixed_types():
    """push() should accept any object, including None and mixed types."""
    s = Stack(3)
    s.push(None)
    s.push("x")
    s.push(3.5)
    assert s.stack == [None, "x", 3.5]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_in_first_out():
    """pop() removes and returns the most recently pushed element (LIFO)."""
    s = Stack(5)
    for value in (-5, 10, 7):
        s.push(value)
    assert s.pop() == 7
    assert s.pop() == 10
    assert s.pop() == -5
    assert s.is_empty() is True


def test_pop_raises_underflow_when_empty():
    """Popping an empty stack raises StackUnderflowError."""
    s = Stack()
    with pytest.raises(StackUnderflowError):
        s.pop()


def test_pop_after_emptying_raises_underflow():
    """A stack emptied by pop() must raise underflow on the next pop."""
    s = Stack(2)
    s.push(1)
    assert s.pop() == 1
    with pytest.raises(StackUnderflowError):
        s.pop()


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing():
    """peek() returns the top element and leaves the stack unchanged."""
    s = Stack(3)
    s.push(-5)
    s.push(10)
    assert s.peek() == 10
    assert s.size() == 2
    assert s.stack == [-5, 10]


def test_peek_raises_underflow_when_empty():
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        Stack().peek()


def test_peek_returns_none_when_none_is_on_top():
    """peek() must return None (not raise) when None is the stored element."""
    s = Stack(1)
    s.push(None)
    assert s.peek() is None
    assert s.size() == 1


# ---------------------------------------------------------------------------
# is_empty / is_full / size
# ---------------------------------------------------------------------------
def test_is_empty_transitions():
    """is_empty() flips correctly as elements are pushed and popped."""
    s = Stack(2)
    assert s.is_empty() is True
    s.push(1)
    assert s.is_empty() is False
    s.pop()
    assert s.is_empty() is True


@pytest.mark.parametrize(
    "limit, pushes, expected_full",
    [
        (10, 0, False),
        (1, 0, False),
        (1, 1, True),
        (3, 2, False),
        (3, 3, True),
        (0, 0, True),
    ],
)
def test_is_full_boundaries(limit, pushes, expected_full):
    """is_full() is True exactly when size equals the limit."""
    s = Stack(limit)
    for i in range(pushes):
        s.push(i)
    assert s.is_full() is expected_full


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_size_tracks_element_count(count):
    """size() equals the number of elements currently stored."""
    s = Stack(10)
    for i in range(count):
        s.push(i)
        assert s.size() == i + 1
    assert s.size() == count


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "items, item, expected",
    [
        ([10], 10, True),
        ([10], 20, False),
        ([], 1, False),
        ([1, 2, 3], 2, True),
        ([1, 2, 3], 4, False),
        ([None], None, True),
        (["a", "b"], "a", True),
    ],
)
def test_contains(items, item, expected):
    """The `in` operator reports membership in the stack."""
    s = Stack(10)
    for value in items:
        s.push(value)
    assert (item in s) is expected


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
def test_exception_types_are_distinct_and_base_exception_derived():
    """Overflow/Underflow errors are distinct and derive from BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# Module-level integration test (mirrors the module's own test_stack)
# ---------------------------------------------------------------------------
def test_module_level_test_stack_runs():
    """The module's own `test_stack` helper should execute without error."""
    stack_module.test_stack()


def test_full_lifecycle_sequence():
    """End-to-end scenario covering every public operation in sequence."""
    s = Stack(3)
    assert bool(s) is False
    assert s.is_empty() is True
    assert s.is_full() is False
    assert str(s) == "[]"

    with pytest.raises(StackUnderflowError):
        s.pop()
    with pytest.raises(StackUnderflowError):
        s.peek()

    for i in range(3):
        assert s.size() == i
        s.push(i)

    assert bool(s) is True
    assert s.is_empty() is False
    assert s.is_full() is True
    assert str(s) == "[0, 1, 2]"
    assert s.pop() == 2
    assert s.peek() == 1

    s.push(100)
    assert str(s) == "[0, 1, 100]"

    with pytest.raises(StackOverflowError):
        s.push(200)

    assert s.size() == 3
    assert 1 in s
    assert 55 not in s