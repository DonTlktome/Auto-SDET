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
    try:
        from data_structures.stacks import stack as module  # type: ignore

        return module
    except Exception:
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
    """A stack created without arguments must default to a limit of 10."""
    s = Stack()
    assert s.limit == 10
    assert s.size() == 0
    assert s.stack == []


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_custom_limit_is_stored(limit):
    """The limit passed to the constructor must be stored verbatim."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.size() == 0


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_adds_element_to_top():
    """push() must append the element to the end (top) of the stack."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert s.stack == [1, 2]
    assert s.peek() == 2


def test_push_returns_none():
    """push() is documented to return None."""
    s = Stack(2)
    assert s.push("x") is None


def test_push_raises_overflow_when_full():
    """Pushing onto a full stack must raise StackOverflowError."""
    s = Stack(1)
    s.push(10)
    with pytest.raises(StackOverflowError):
        s.push(20)


def test_push_raises_overflow_on_zero_limit():
    """A stack with limit 0 is immediately full, so any push overflows."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)


def test_push_overflow_does_not_mutate_stack():
    """A failed push must leave the stack contents unchanged."""
    s = Stack(2)
    s.push(1)
    s.push(2)
    with pytest.raises(StackOverflowError):
        s.push(3)
    assert s.stack == [1, 2]
    assert s.size() == 2


def test_push_accepts_heterogeneous_values():
    """push() is generic and must accept any object type."""
    s = Stack(4)
    values = [1, "two", 3.0, None]
    for v in values:
        s.push(v)
    assert s.stack == values


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed_element():
    """pop() must return elements in LIFO order."""
    s = Stack()
    s.push(-5)
    s.push(10)
    assert s.pop() == 10
    assert s.pop() == -5


def test_pop_raises_underflow_on_empty_stack():
    """Popping an empty stack must raise StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        Stack().pop()


def test_pop_after_underflow_still_works():
    """After an underflow error the stack must remain usable."""
    s = Stack(2)
    with pytest.raises(StackUnderflowError):
        s.pop()
    s.push(7)
    assert s.pop() == 7
    assert s.is_empty() is True


def test_pop_reduces_size():
    """Each successful pop must decrement the size by one."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert s.size() == 2
    s.pop()
    assert s.size() == 1


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing():
    """peek() must return the top element and leave the stack unchanged."""
    s = Stack()
    s.push(-5)
    s.push(10)
    assert s.peek() == 10
    assert s.size() == 2
    assert s.peek() == 10


def test_peek_raises_underflow_on_empty_stack():
    """Peeking an empty stack must raise StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        Stack().peek()


def test_peek_single_element():
    """peek() on a one-element stack returns that element."""
    s = Stack(1)
    s.push("only")
    assert s.peek() == "only"


# ---------------------------------------------------------------------------
# is_empty / is_full / size
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack():
    """A freshly created stack must report itself as empty."""
    assert Stack().is_empty() is True


def test_is_empty_false_after_push():
    """A stack with at least one element must not be empty."""
    s = Stack()
    s.push(10)
    assert s.is_empty() is False


def test_is_empty_true_again_after_all_pops():
    """Emptying a stack via pop() must make is_empty() true again."""
    s = Stack(2)
    s.push(1)
    s.pop()
    assert s.is_empty() is True


def test_is_full_false_for_new_stack():
    """A new stack with a positive limit must not be full."""
    assert Stack().is_full() is False


def test_is_full_true_when_size_equals_limit():
    """is_full() must be true exactly when size == limit."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_true_for_zero_limit_stack():
    """A zero-limit stack is full from the start."""
    assert Stack(0).is_full() is True


def test_is_full_false_after_pop():
    """Popping from a full stack must make is_full() false."""
    s = Stack(1)
    s.push(1)
    s.pop()
    assert s.is_full() is False


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count):
    """size() must equal the number of elements currently on the stack."""
    s = Stack(5)
    for i in range(count):
        s.push(i)
    assert s.size() == count


# ---------------------------------------------------------------------------
# dunder methods: __bool__, __str__, __contains__
# ---------------------------------------------------------------------------
def test_bool_false_when_empty():
    """bool(stack) must be False for an empty stack."""
    assert bool(Stack()) is False


def test_bool_true_when_non_empty():
    """bool(stack) must be True once an element is pushed."""
    s = Stack()
    s.push(0)  # falsy element, but the stack itself is non-empty
    assert bool(s) is True


def test_str_of_empty_stack():
    """str() of an empty stack must be the empty list representation."""
    assert str(Stack()) == "[]"


def test_str_matches_underlying_list():
    """str() must mirror the underlying list representation."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert str(s) == str([1, 2])


@pytest.mark.parametrize(
    ("contents", "item", "expected"),
    [
        ([10], 10, True),
        ([10], 20, False),
        ([1, 2, 3], 2, True),
        ([1, 2, 3], 4, False),
        ([], 1, False),
        (["a", "b"], "b", True),
    ],
)
def test_contains(contents, item, expected):
    """`in` must reflect membership in the underlying stack list."""
    s = Stack(10)
    for value in contents:
        s.push(value)
    assert (item in s) is expected


# ---------------------------------------------------------------------------
# Error class hierarchy
# ---------------------------------------------------------------------------
def test_error_classes_are_distinct_and_raisable():
    """Overflow and underflow errors must be distinct, catchable classes."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    assert StackOverflowError is not StackUnderflowError
    with pytest.raises(StackOverflowError):
        raise StackOverflowError
    with pytest.raises(StackUnderflowError):
        raise StackUnderflowError


# ---------------------------------------------------------------------------
# Integration: the module's own self-test
# ---------------------------------------------------------------------------
def test_module_self_test_passes():
    """The module-provided test_stack() helper must run without assertions."""
    stack_module.test_stack()


def test_full_lifecycle():
    """End-to-end LIFO behaviour across push/pop/peek/size/is_full."""
    s = Stack(3)
    assert s.is_empty() and not s.is_full()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.is_full()
    with pytest.raises(StackOverflowError):
        s.push(4)
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]
    assert s.is_empty()
    with pytest.raises(StackUnderflowError):
        s.pop()