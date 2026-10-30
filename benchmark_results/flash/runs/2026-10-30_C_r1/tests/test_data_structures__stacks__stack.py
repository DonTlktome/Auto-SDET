"""Tests for data_structures/stacks/stack.py.

The module under test is imported by file path so the tests do not depend on
the package layout of the repository.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Locate the module under test (no hard-coded absolute paths).
# ---------------------------------------------------------------------------
_RELATIVE_SOURCE = Path("data_structures") / "stacks" / "stack.py"

# Only ever used as a last-resort hint (the original author's machine).
_LEGACY_SOURCE = Path(
    r"D:\Documents\Study\Projects\algorithms-bench\data_structures\stacks\stack.py"
)

# Loading the module under its canonical dotted name keeps the exception lines
# printed by the module's own doctests
# ("data_structures.stacks.stack.StackOverflowError") identical to what doctest
# expects, even though the file is loaded directly from disk.
_MODULE_NAME = "data_structures.stacks.stack"


def _looks_like_stack_module(path: Path) -> bool:
    """Return True if *path* looks like the stack implementation module."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "class Stack" in text and "StackUnderflowError" in text


def _iter_candidate_paths() -> Iterator[Path]:
    """Yield plausible locations of the module under test."""
    seen: set[Path] = set()
    roots = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for root in roots:
        for base in (root, *root.parents):
            for candidate in (
                base / _RELATIVE_SOURCE,
                base / "stacks" / "stack.py",
                base / "stack.py",
            ):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved
    try:
        legacy = _LEGACY_SOURCE.resolve()
    except OSError:
        return
    if legacy not in seen:
        yield legacy


def _find_source() -> Path:
    """Find the stack module on disk, or raise a helpful FileNotFoundError."""
    for candidate in _iter_candidate_paths():
        if candidate.is_file() and _looks_like_stack_module(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate data_structures/stacks/stack.py next to the test file "
        "or in the current working directory."
    )


_SOURCE_PATH = _find_source()

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE_PATH)
assert _spec is not None and _spec.loader is not None
stack_module = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = stack_module
sys.modules.setdefault("stack_under_test", stack_module)
_spec.loader.exec_module(stack_module)

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
def filled_stack() -> "Stack[int]":
    """Return a stack of limit 3 containing [1, 2, 3]."""
    s: "Stack[int]" = Stack(3)
    for value in (1, 2, 3):
        s.push(value)
    return s


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------
def test_exception_classes_are_defined() -> None:
    """Both custom exceptions exist and derive from BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)


def test_exception_classes_are_distinct() -> None:
    """The two error types must not be the same class nor subclasses of each other."""
    assert StackOverflowError is not StackUnderflowError
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_defaults() -> None:
    """A default stack starts empty with a limit of 10."""
    s = Stack()
    assert s.stack == []
    assert s.limit == 10
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit: int) -> None:
    """A custom limit is stored verbatim and the stack starts empty."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.stack == []


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------
def test_bool_empty_stack_is_false(empty_stack: "Stack[int]") -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_non_empty_stack_is_true(filled_stack: "Stack[int]") -> None:
    """A stack holding at least one element is truthy."""
    assert bool(filled_stack) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
def test_str_empty_stack(empty_stack: "Stack[int]") -> None:
    """str() of an empty stack renders the underlying list."""
    assert str(empty_stack) == "[]"


def test_str_non_empty_stack(filled_stack: "Stack[int]") -> None:
    """str() mirrors the internal list representation."""
    assert str(filled_stack) == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_single_element(empty_stack: "Stack[int]") -> None:
    """Pushing onto an empty stack appends the element and returns None."""
    assert empty_stack.push(42) is None
    assert empty_stack.stack == [42]
    assert empty_stack.size() == 1


def test_push_preserves_order(empty_stack: "Stack[int]") -> None:
    """Elements are stored in insertion order (bottom -> top)."""
    for value in (1, 2, 3):
        empty_stack.push(value)
    assert empty_stack.stack == [1, 2, 3]


def test_push_up_to_limit_succeeds() -> None:
    """Pushing exactly `limit` elements must not raise."""
    s: "Stack[int]" = Stack(3)
    for value in range(3):
        s.push(value)
    assert s.size() == 3
    assert s.is_full() is True


def test_push_beyond_limit_raises_overflow() -> None:
    """Pushing one element past the limit raises StackOverflowError."""
    s: "Stack[int]" = Stack(2)
    s.push(1)
    s.push(2)
    with pytest.raises(StackOverflowError):
        s.push(3)
    # The failed push must not mutate the stack.
    assert s.stack == [1, 2]


def test_push_on_zero_limit_stack_raises_overflow() -> None:
    """A stack with limit 0 can never accept an element."""
    s: "Stack[int]" = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)
    assert s.is_empty() is True


@pytest.mark.parametrize(
    "value",
    [0, -1, 3.14, "text", None, (1, 2), [1, 2], {"a": 1}],
)
def test_push_accepts_arbitrary_types(value: object) -> None:
    """The stack is generic and stores any object unchanged."""
    s: "Stack[object]" = Stack(5)
    s.push(value)
    assert s.peek() is value


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(filled_stack: "Stack[int]") -> None:
    """pop() removes and returns the most recently pushed element (LIFO)."""
    assert filled_stack.pop() == 3
    assert filled_stack.size() == 2
    assert filled_stack.stack == [1, 2]


def test_pop_all_elements_is_lifo() -> None:
    """Popping everything yields elements in reverse insertion order."""
    s: "Stack[int]" = Stack(5)
    for value in (1, 2, 3):
        s.push(value)
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]
    assert s.is_empty() is True


def test_pop_empty_stack_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Popping an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_after_draining_raises_underflow() -> None:
    """Once drained, a further pop raises StackUnderflowError."""
    s: "Stack[int]" = Stack(2)
    s.push(1)
    assert s.pop() == 1
    with pytest.raises(StackUnderflowError):
        s.pop()


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(filled_stack: "Stack[int]") -> None:
    """peek() returns the top element and leaves the stack unchanged."""
    assert filled_stack.peek() == 3
    assert filled_stack.size() == 3
    assert filled_stack.stack == [1, 2, 3]


def test_peek_is_idempotent(filled_stack: "Stack[int]") -> None:
    """Repeated peeks return the same value."""
    assert filled_stack.peek() == filled_stack.peek() == 3


def test_peek_empty_stack_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A newly created stack reports empty."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack: "Stack[int]") -> None:
    """After a push the stack is no longer empty."""
    empty_stack.push(1)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_draining() -> None:
    """Draining the stack makes is_empty() true again."""
    s: "Stack[int]" = Stack(2)
    s.push(1)
    s.pop()
    assert s.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------
def test_is_full_false_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A new stack with a positive limit is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_at_limit() -> None:
    """A stack holding exactly `limit` elements is full."""
    s: "Stack[int]" = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_after_pop() -> None:
    """Popping from a full stack makes it not full again."""
    s: "Stack[int]" = Stack(2)
    s.push(1)
    s.push(2)
    assert s.is_full() is True
    s.pop()
    assert s.is_full() is False


def test_is_full_true_for_zero_limit() -> None:
    """A zero-limit stack is full by definition."""
    assert Stack(0).is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """size() equals the number of elements currently stored."""
    s: "Stack[int]" = Stack(5)
    for value in range(count):
        s.push(value)
    assert s.size() == count


def test_size_decreases_after_pop(filled_stack: "Stack[int]") -> None:
    """size() reflects removals performed by pop()."""
    filled_stack.pop()
    assert filled_stack.size() == 2


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
def test_contains_present_item(filled_stack: "Stack[int]") -> None:
    """An element that was pushed is reported as contained."""
    assert (2 in filled_stack) is True


def test_contains_absent_item(filled_stack: "Stack[int]") -> None:
    """An element never pushed is not contained."""
    assert (99 in filled_stack) is False


def test_contains_on_empty_stack(empty_stack: "Stack[int]") -> None:
    """Nothing is contained in an empty stack."""
    assert (1 in empty_stack) is False


def test_contains_after_pop(filled_stack: "Stack[int]") -> None:
    """A popped element is no longer contained."""
    filled_stack.pop()
    assert (3 in filled_stack) is False


# ---------------------------------------------------------------------------
# Integration / module-level helper
# ---------------------------------------------------------------------------
def test_module_level_test_stack_helper_runs() -> None:
    """The module's own `test_stack()` helper executes without raising."""
    assert stack_module.test_stack() is None


def test_lifo_integration_sequence() -> None:
    """A mixed push/pop/peek sequence behaves as a strict LIFO container."""
    s: "Stack[int]" = Stack(10)
    assert bool(s) is False
    assert s.is_empty() is True
    assert s.is_full() is False
    assert str(s) == "[]"

    for i in range(10):
        assert s.size() == i
        s.push(i)

    assert bool(s) is True
    assert s.is_empty() is False
    assert s.is_full() is True
    assert str(s) == str(list(range(10)))
    assert s.pop() == 9
    assert s.peek() == 8

    s.push(100)
    assert str(s) == str([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])

    with pytest.raises(StackOverflowError):
        s.push(200)

    assert s.size() == 10
    assert 5 in s
    assert 55 not in s


def test_doctests_pass() -> None:
    """All doctests embedded in the source module must pass."""
    import doctest

    results = doctest.testmod(stack_module, verbose=False)
    assert results.failed == 0
