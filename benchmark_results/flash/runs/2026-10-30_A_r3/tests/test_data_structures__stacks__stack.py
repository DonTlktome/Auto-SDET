"""Pytest test suite for data_structures.stacks.stack.

The module under test defines a generic LIFO ``Stack`` with a fixed capacity,
plus two custom exception types (``StackOverflowError`` and
``StackUnderflowError``) that derive from ``BaseException``.

Because the exceptions inherit from ``BaseException`` (not ``Exception``),
``pytest.raises`` is used explicitly for them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The source file lives at .../data_structures/stacks/stack.py and uses PEP 695
# generic syntax (``class Stack[T]``), so it requires Python >= 3.12.  We try a
# normal package import first and fall back to loading the file directly by
# path so the tests work regardless of how the project is laid out.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on environment layout
    from data_structures.stacks.stack import (
        Stack,
        StackOverflowError,
        StackUnderflowError,
    )
except Exception:  # pragma: no cover - fallback path loading
    import importlib.util

    _SOURCE = Path(__file__).resolve().parents[1] / "stack.py"
    if not _SOURCE.exists():
        _SOURCE = Path(
            r"D:\Documents\Study\Projects\algorithms-bench"
            r"\data_structures\stacks\stack.py"
        )
    _spec = importlib.util.spec_from_file_location("stack_under_test", _SOURCE)
    assert _spec is not None and _spec.loader is not None
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["stack_under_test"] = _module
    _spec.loader.exec_module(_module)

    Stack = _module.Stack
    StackOverflowError = _module.StackOverflowError
    StackUnderflowError = _module.StackUnderflowError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def empty_stack() -> Stack[int]:
    """Return a fresh, empty stack with the default limit of 10."""
    return Stack()


@pytest.fixture()
def small_stack() -> Stack[int]:
    """Return a fresh, empty stack with a limit of 3."""
    return Stack(3)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
def test_exceptions_are_base_exception_subclasses() -> None:
    """Both custom errors must derive from BaseException (not Exception)."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    assert not issubclass(StackOverflowError, Exception)
    assert not issubclass(StackUnderflowError, Exception)


def test_exceptions_are_distinct_types() -> None:
    """The two error types must not be related to each other."""
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_defaults() -> None:
    """A default stack starts empty with a limit of 10."""
    stack: Stack[int] = Stack()
    assert stack.stack == []
    assert stack.limit == 10
    assert stack.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit: int) -> None:
    """A custom limit is stored verbatim and the stack starts empty."""
    stack: Stack[int] = Stack(limit)
    assert stack.limit == limit
    assert stack.stack == []
    assert stack.size() == 0


def test_init_negative_limit_is_stored() -> None:
    """A negative limit is accepted at construction time (no validation)."""
    stack: Stack[int] = Stack(-1)
    assert stack.limit == -1
    assert stack.is_empty() is True


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------
def test_bool_empty_stack_is_false(empty_stack: Stack[int]) -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_non_empty_stack_is_true(empty_stack: Stack[int]) -> None:
    """A stack holding at least one element is truthy."""
    empty_stack.push(1)
    assert bool(empty_stack) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], "[]"),
        ([1], "[1]"),
        ([1, 2, 3], "[1, 2, 3]"),
        (["a", "b"], "['a', 'b']"),
    ],
)
def test_str_representation(items: list, expected: str) -> None:
    """``str(stack)`` mirrors the string form of the underlying list."""
    stack: Stack = Stack(10)
    for item in items:
        stack.push(item)
    assert str(stack) == expected


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_single_element(empty_stack: Stack[int]) -> None:
    """Pushing one element appends it to the top of the stack."""
    empty_stack.push(42)
    assert empty_stack.stack == [42]
    assert empty_stack.size() == 1
    assert empty_stack.peek() == 42


def test_push_returns_none(empty_stack: Stack[int]) -> None:
    """``push`` is a mutator and returns ``None``."""
    assert empty_stack.push(1) is None


def test_push_multiple_preserves_order(empty_stack: Stack[int]) -> None:
    """Elements are stored in insertion order (bottom -> top)."""
    for value in (1, 2, 3):
        empty_stack.push(value)
    assert empty_stack.stack == [1, 2, 3]


def test_push_up_to_limit(small_stack: Stack[int]) -> None:
    """Pushing exactly ``limit`` elements succeeds and fills the stack."""
    small_stack.push(1)
    small_stack.push(2)
    small_stack.push(3)
    assert small_stack.is_full() is True
    assert small_stack.size() == 3


def test_push_beyond_limit_raises_overflow(small_stack: Stack[int]) -> None:
    """Pushing past the limit raises ``StackOverflowError``."""
    for value in (1, 2, 3):
        small_stack.push(value)
    with pytest.raises(StackOverflowError):
        small_stack.push(4)
    # The failed push must not mutate the stack.
    assert small_stack.stack == [1, 2, 3]


def test_push_on_zero_limit_stack_raises_overflow() -> None:
    """A stack with limit 0 can never accept an element."""
    stack: Stack[int] = Stack(0)
    assert stack.is_full() is True
    with pytest.raises(StackOverflowError):
        stack.push(1)
    assert stack.is_empty() is True


def test_push_after_pop_succeeds(small_stack: Stack[int]) -> None:
    """Popping frees a slot so a subsequent push is allowed."""
    for value in (1, 2, 3):
        small_stack.push(value)
    assert small_stack.pop() == 3
    small_stack.push(99)
    assert small_stack.stack == [1, 2, 99]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(empty_stack: Stack[int]) -> None:
    """``pop`` removes and returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10
    assert empty_stack.size() == 1


def test_pop_empties_stack(empty_stack: Stack[int]) -> None:
    """Popping the only element leaves the stack empty."""
    empty_stack.push(7)
    assert empty_stack.pop() == 7
    assert empty_stack.is_empty() is True
    assert empty_stack.stack == []


def test_pop_on_empty_raises_underflow(empty_stack: Stack[int]) -> None:
    """Popping an empty stack raises ``StackUnderflowError``."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_drains_in_reverse_order() -> None:
    """Draining the stack yields elements in reverse insertion order."""
    stack: Stack[int] = Stack(5)
    for value in (1, 2, 3, 4, 5):
        stack.push(value)
    popped = [stack.pop() for _ in range(5)]
    assert popped == [5, 4, 3, 2, 1]
    assert stack.is_empty() is True


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(empty_stack: Stack[int]) -> None:
    """``peek`` returns the top element and leaves the stack unchanged."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2
    assert empty_stack.stack == [-5, 10]


def test_peek_on_empty_raises_underflow(empty_stack: Stack[int]) -> None:
    """Peeking an empty stack raises ``StackUnderflowError``."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_after_pop(empty_stack: Stack[int]) -> None:
    """After a pop, ``peek`` exposes the new top element."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.peek() == 1


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack(empty_stack: Stack[int]) -> None:
    """A newly created stack reports itself as empty."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack: Stack[int]) -> None:
    """A stack with an element is not empty."""
    empty_stack.push(10)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_draining(empty_stack: Stack[int]) -> None:
    """A stack becomes empty again once all elements are popped."""
    empty_stack.push(1)
    empty_stack.pop()
    assert empty_stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------
def test_is_full_false_for_new_stack(empty_stack: Stack[int]) -> None:
    """A new stack with a positive limit is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_when_at_limit() -> None:
    """A stack holding exactly ``limit`` elements is full."""
    stack: Stack[int] = Stack(1)
    stack.push(10)
    assert stack.is_full() is True


def test_is_full_false_when_below_limit(small_stack: Stack[int]) -> None:
    """A stack below its limit is not full."""
    small_stack.push(1)
    small_stack.push(2)
    assert small_stack.is_full() is False


def test_is_full_true_for_zero_limit() -> None:
    """A zero-limit stack is full from the start."""
    assert Stack(0).is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """``size`` equals the number of elements currently on the stack."""
    stack: Stack[int] = Stack(3)
    for value in range(count):
        stack.push(value)
    assert stack.size() == count


def test_size_decreases_after_pop(empty_stack: Stack[int]) -> None:
    """``size`` shrinks by one for each pop."""
    empty_stack.push(1)
    empty_stack.push(2)
    assert empty_stack.size() == 2
    empty_stack.pop()
    assert empty_stack.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
def test_contains_present_item(small_stack: Stack[int]) -> None:
    """``in`` returns True for an element on the stack."""
    small_stack.push(10)
    assert 10 in small_stack


def test_contains_absent_item(small_stack: Stack[int]) -> None:
    """``in`` returns False for an element not on the stack."""
    small_stack.push(10)
    assert 20 not in small_stack


def test_contains_on_empty_stack(empty_stack: Stack[int]) -> None:
    """``in`` is always False on an empty stack."""
    assert 1 not in empty_stack


@pytest.mark.parametrize(
    ("items", "probe", "expected"),
    [
        ([1, 2, 3], 1, True),
        ([1, 2, 3], 3, True),
        ([1, 2, 3], 4, False),
        (["a", "b"], "a", True),
        (["a", "b"], "c", False),
        ([None], None, True),
    ],
)
def test_contains_parametrized(items: list, probe: object, expected: bool) -> None:
    """``in`` behaves like membership testing on the underlying list."""
    stack: Stack = Stack(10)
    for item in items:
        stack.push(item)
    assert (probe in stack) is expected


# ---------------------------------------------------------------------------
# Integration / LIFO behaviour
# ---------------------------------------------------------------------------
def test_lifo_ordering() -> None:
    """Elements come off the stack in Last-In-First-Out order."""
    stack: Stack[int] = Stack(5)
    for value in (1, 2, 3):
        stack.push(value)
    assert [stack.pop(), stack.pop(), stack.pop()] == [3, 2, 1]


def test_interleaved_push_pop() -> None:
    """Interleaved pushes and pops maintain correct LIFO semantics."""
    stack: Stack[int] = Stack(3)
    stack.push(1)
    stack.push(2)
    assert stack.pop() == 2
    stack.push(3)
    assert stack.pop() == 3
    assert stack.pop() == 1
    assert stack.is_empty() is True


def test_stack_holds_arbitrary_objects() -> None:
    """The stack is generic and can hold any Python object."""
    sentinel = MagicMock()
    stack: Stack = Stack(2)
    stack.push(sentinel)
    stack.push(None)
    assert stack.peek() is None
    assert stack.pop() is None
    assert stack.pop() is sentinel


# ---------------------------------------------------------------------------
# Isolation: no external I/O is performed by the stack
# ---------------------------------------------------------------------------
def test_stack_does_not_touch_filesystem(empty_stack: Stack[int]) -> None:
    """The stack performs no file I/O; ``open`` must never be called."""
    with patch("builtins.open", side_effect=AssertionError("no I/O allowed")):
        empty_stack.push(1)
        assert empty_stack.pop() == 1
        assert empty_stack.is_empty() is True


# ---------------------------------------------------------------------------
# Module-level ``test_stack`` helper (also used as a doctest)
# ---------------------------------------------------------------------------
def test_module_level_test_stack_helper_runs() -> None:
    """The module's own ``test_stack`` self-check must pass without raising."""
    module = sys.modules[Stack.__module__]
    helper = getattr(module, "test_stack", None)
    if helper is None:  # pragma: no cover - defensive
        pytest.skip("module-level test_stack helper not available")
    assert helper() is None


def test_module_level_test_stack_helper_raises_on_broken_stack() -> None:
    """The helper fails loudly if ``pop`` silently returns instead of raising."""
    module = sys.modules[Stack.__module__]
    helper = getattr(module, "test_stack", None)
    if helper is None:  # pragma: no cover - defensive
        pytest.skip("module-level test_stack helper not available")

    with patch.object(Stack, "pop", return_value=None):
        with pytest.raises(AssertionError):
            helper()