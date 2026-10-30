"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The source file uses PEP 695 generic syntax (``class Stack[T]``), so it can
# only be imported on Python >= 3.12.  We load it directly from its file path
# to avoid depending on package layout / __init__.py files.
#
# The module MUST be registered under its fully-qualified dotted name: the
# doctests embedded in the source expect tracebacks such as
# ``data_structures.stacks.stack.StackOverflowError``, and doctest compares
# that exception string verbatim (it is taken from the exception class's
# __module__).  Loading the file under a different name (e.g.
# "stack_under_test") rewrites the exception line and makes those doctests
# fail.
# ---------------------------------------------------------------------------

_MODULE_NAME = "data_structures.stacks.stack"

_SOURCE_PATH = Path(
    r"D:\Documents\Study\Projects\algorithms-bench\data_structures\stacks\stack.py"
)

if not _SOURCE_PATH.exists():  # pragma: no cover - fallback for relocated files
    _SOURCE_PATH = Path(__file__).resolve().parent / "stack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # module_from_spec already sets __name__ from the spec name, so every
    # class defined by the module gets __module__ == _MODULE_NAME.
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


stack_module = _load_module()

Stack = stack_module.Stack
StackOverflowError = stack_module.StackOverflowError
StackUnderflowError = stack_module.StackUnderflowError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_stack() -> "Stack[int]":
    """Return a fresh, empty stack with the default limit."""
    return Stack()


@pytest.fixture()
def small_stack() -> "Stack[int]":
    """Return a fresh, empty stack with a limit of 3."""
    return Stack(3)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_exception_classes_are_defined() -> None:
    """Both custom exceptions exist and derive from BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)


def test_exception_classes_are_distinct() -> None:
    """The two error types must not be related to each other."""
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_default_limit_is_ten() -> None:
    """A stack created without arguments has a limit of 10 and is empty."""
    s = Stack()
    assert s.limit == 10
    assert s.stack == []
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit: int) -> None:
    """A custom limit is stored verbatim and the internal list starts empty."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.stack == []


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------


def test_bool_empty_stack_is_false(empty_stack: "Stack[int]") -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_non_empty_stack_is_true(empty_stack: "Stack[int]") -> None:
    """A stack holding at least one element is truthy."""
    empty_stack.push(1)
    assert bool(empty_stack) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


def test_str_empty_stack(empty_stack: "Stack[int]") -> None:
    """str() of an empty stack is the empty list representation."""
    assert str(empty_stack) == "[]"


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([1], "[1]"),
        ([1, 2, 3], "[1, 2, 3]"),
        (["a", "b"], "['a', 'b']"),
        ([None], "[None]"),
    ],
)
def test_str_with_items(items: list, expected: str) -> None:
    """str() mirrors the underlying list, bottom element first."""
    s = Stack(10)
    for item in items:
        s.push(item)
    assert str(s) == expected


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


def test_push_single_element(empty_stack: "Stack[int]") -> None:
    """Pushing one element increases size and makes it the top element."""
    empty_stack.push(42)
    assert empty_stack.size() == 1
    assert empty_stack.peek() == 42


def test_push_returns_none(empty_stack: "Stack[int]") -> None:
    """push() is a mutator and returns None."""
    assert empty_stack.push(1) is None


def test_push_preserves_order(empty_stack: "Stack[int]") -> None:
    """Elements are stored in insertion order (bottom to top)."""
    for value in (10, 20, 30):
        empty_stack.push(value)
    assert empty_stack.stack == [10, 20, 30]


def test_push_up_to_limit(small_stack: "Stack[int]") -> None:
    """Pushing exactly `limit` elements is allowed."""
    for value in (1, 2, 3):
        small_stack.push(value)
    assert small_stack.size() == 3
    assert small_stack.is_full() is True


def test_push_beyond_limit_raises_overflow(small_stack: "Stack[int]") -> None:
    """Pushing past the limit raises StackOverflowError."""
    for value in (1, 2, 3):
        small_stack.push(value)
    with pytest.raises(StackOverflowError):
        small_stack.push(4)
    # The failed push must not mutate the stack.
    assert small_stack.size() == 3
    assert small_stack.peek() == 3


def test_push_on_zero_limit_stack_raises_overflow() -> None:
    """A stack with limit 0 can never accept an element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)
    assert s.size() == 0


def test_push_after_pop_succeeds(small_stack: "Stack[int]") -> None:
    """After popping, there is room again so push must succeed."""
    for value in (1, 2, 3):
        small_stack.push(value)
    small_stack.pop()
    small_stack.push(99)
    assert small_stack.stack == [1, 2, 99]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------


def test_pop_returns_last_pushed(empty_stack: "Stack[int]") -> None:
    """pop() returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10


def test_pop_decreases_size(empty_stack: "Stack[int]") -> None:
    """pop() removes exactly one element."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.size() == 1


def test_pop_empty_stack_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Popping an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_until_empty_then_underflow(empty_stack: "Stack[int]") -> None:
    """Popping all elements works; one more pop raises underflow."""
    empty_stack.push(1)
    empty_stack.push(2)
    assert empty_stack.pop() == 2
    assert empty_stack.pop() == 1
    assert empty_stack.is_empty() is True
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_lifo_order(empty_stack: "Stack[int]") -> None:
    """Elements come off in reverse insertion order."""
    values = [1, 2, 3, 4, 5]
    for value in values:
        empty_stack.push(value)
    popped = [empty_stack.pop() for _ in values]
    assert popped == list(reversed(values))


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------


def test_peek_returns_top_without_removing(empty_stack: "Stack[int]") -> None:
    """peek() returns the top element and leaves the stack unchanged."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2
    assert empty_stack.peek() == 10


def test_peek_empty_stack_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_single_element(empty_stack: "Stack[int]") -> None:
    """With one element, peek returns that element."""
    empty_stack.push("only")
    assert empty_stack.peek() == "only"


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_is_empty_true_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A newly created stack is empty."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack: "Stack[int]") -> None:
    """A stack with an element is not empty."""
    empty_stack.push(10)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_push_and_pop(empty_stack: "Stack[int]") -> None:
    """Emptying the stack again makes is_empty() true."""
    empty_stack.push(10)
    empty_stack.pop()
    assert empty_stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------


def test_is_full_false_for_new_default_stack(empty_stack: "Stack[int]") -> None:
    """A fresh default stack (limit 10) is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_when_limit_reached() -> None:
    """A stack with limit 1 becomes full after one push."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_when_one_below_limit(small_stack: "Stack[int]") -> None:
    """A stack one element below its limit is not full."""
    small_stack.push(1)
    small_stack.push(2)
    assert small_stack.is_full() is False


def test_is_full_true_for_zero_limit_stack() -> None:
    """A stack with limit 0 is full from the start."""
    assert Stack(0).is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """size() equals the number of elements currently on the stack."""
    s = Stack(3)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_after_pop(empty_stack: "Stack[int]") -> None:
    """size() decreases by one after a pop."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


def test_contains_present_item(small_stack: "Stack[int]") -> None:
    """An element that was pushed is found by `in`."""
    small_stack.push(10)
    assert 10 in small_stack


def test_contains_absent_item(small_stack: "Stack[int]") -> None:
    """An element that was never pushed is not found by `in`."""
    small_stack.push(10)
    assert 20 not in small_stack


def test_contains_on_empty_stack(empty_stack: "Stack[int]") -> None:
    """Nothing is contained in an empty stack."""
    assert 1 not in empty_stack


@pytest.mark.parametrize(
    ("items", "probe", "expected"),
    [
        ([1, 2, 3], 2, True),
        ([1, 2, 3], 4, False),
        (["a"], "a", True),
        (["a"], "b", False),
        ([None], None, True),
        ([0], False, True),  # 0 == False in Python
    ],
)
def test_contains_parametrized(items: list, probe: object, expected: bool) -> None:
    """`in` behaves like membership testing on the underlying list."""
    s = Stack(10)
    for item in items:
        s.push(item)
    assert (probe in s) is expected


# ---------------------------------------------------------------------------
# Integration / module-level test function
# ---------------------------------------------------------------------------


def test_module_level_test_stack_runs() -> None:
    """The module's own `test_stack()` helper must execute without error."""
    stack_module.test_stack()


def test_full_lifecycle() -> None:
    """End-to-end scenario covering every public operation."""
    stack: "Stack[int]" = Stack(10)
    assert bool(stack) is False
    assert stack.is_empty() is True
    assert stack.is_full() is False
    assert str(stack) == "[]"

    with pytest.raises(StackUnderflowError):
        stack.pop()
    with pytest.raises(StackUnderflowError):
        stack.peek()

    for i in range(10):
        assert stack.size() == i
        stack.push(i)

    assert bool(stack)
    assert not stack.is_empty()
    assert stack.is_full()
    assert str(stack) == str(list(range(10)))
    assert stack.pop() == 9
    assert stack.peek() == 8

    stack.push(100)
    assert str(stack) == str([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])

    with pytest.raises(StackOverflowError):
        stack.push(200)

    assert not stack.is_empty()
    assert stack.size() == 10
    assert 5 in stack
    assert 55 not in stack


def test_doctests_pass() -> None:
    """All doctests embedded in the source module must pass."""
    import doctest

    results = doctest.testmod(stack_module, verbose=False)
    assert results.failed == 0
