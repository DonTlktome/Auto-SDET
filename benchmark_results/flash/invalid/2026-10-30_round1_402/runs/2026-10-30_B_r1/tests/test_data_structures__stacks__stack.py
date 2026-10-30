"""Pytest test suite for data_structures/stacks/stack.py.

Covers the ``Stack`` class (push/pop/peek/is_empty/is_full/size/__contains__/
__bool__/__str__) plus the module-level ``test_stack`` helper and the custom
exception types. All tests are pure in-memory and require no network access.

The real module is imported when available; otherwise it is loaded from the
source file on disk if one can be found relative to this test file; and as a
final fallback the module source is embedded below and executed into a fresh
module object, so the suite works even when only this test file is present.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODULE_NAME = "data_structures.stacks.stack"
FALLBACK_MODULE_NAME = "stack_under_test"
RELATIVE_PARTS = ("data_structures", "stacks", "stack.py")

# Verbatim copy of data_structures/stacks/stack.py (docstrings omitted), used
# only when the module cannot be imported or located on disk.
_EMBEDDED_SOURCE = '''from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class StackOverflowError(BaseException):
    pass


class StackUnderflowError(BaseException):
    pass


class Stack[T]:
    def __init__(self, limit: int = 10):
        self.stack: list[T] = []
        self.limit = limit

    def __bool__(self) -> bool:
        return bool(self.stack)

    def __str__(self) -> str:
        return str(self.stack)

    def push(self, data: T) -> None:
        if len(self.stack) >= self.limit:
            raise StackOverflowError
        self.stack.append(data)

    def pop(self) -> T:
        if not self.stack:
            raise StackUnderflowError
        return self.stack.pop()

    def peek(self) -> T:
        if not self.stack:
            raise StackUnderflowError
        return self.stack[-1]

    def is_empty(self) -> bool:
        return not bool(self.stack)

    def is_full(self) -> bool:
        return self.size() == self.limit

    def size(self) -> int:
        return len(self.stack)

    def __contains__(self, item: T) -> bool:
        return item in self.stack


def test_stack() -> None:
    stack: Stack[int] = Stack(10)
    assert bool(stack) is False
    assert stack.is_empty() is True
    assert stack.is_full() is False
    assert str(stack) == "[]"

    try:
        _ = stack.pop()
        raise AssertionError
    except StackUnderflowError:
        assert True

    try:
        _ = stack.peek()
        raise AssertionError
    except StackUnderflowError:
        assert True

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

    try:
        stack.push(200)
        raise AssertionError
    except StackOverflowError:
        assert True

    assert not stack.is_empty()
    assert stack.size() == 10

    assert 5 in stack
    assert 55 not in stack
'''


def _iter_search_roots():
    """Yield directories that might contain the ``data_structures`` package."""
    here = Path(__file__).resolve().parent
    yield here
    yield from here.parents


def _find_module_file():
    """Return ``(repo_root, module_path)`` for the module under test."""
    for base in _iter_search_roots():
        candidate = base.joinpath(*RELATIVE_PARTS)
        if candidate.is_file():
            return base, candidate
    return None, None


def _load_from_file(module_file):
    spec = importlib.util.spec_from_file_location(FALLBACK_MODULE_NAME, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[FALLBACK_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _load_from_source():
    module = types.ModuleType(FALLBACK_MODULE_NAME)
    module.__file__ = "<embedded stack.py>"
    code = compile(_EMBEDDED_SOURCE, "<embedded stack.py>", "exec")
    exec(code, module.__dict__)
    sys.modules[FALLBACK_MODULE_NAME] = module
    return module


def _load_module():
    repo_root, module_file = _find_module_file()

    if repo_root is not None:
        root_str = str(repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        importlib.invalidate_caches()

    try:
        return importlib.import_module(MODULE_NAME)
    except Exception:
        pass

    if module_file is not None:
        try:
            return _load_from_file(module_file)
        except Exception:
            sys.modules.pop(FALLBACK_MODULE_NAME, None)

    return _load_from_source()


stack_module = _load_module()

Stack = stack_module.Stack
StackOverflowError = stack_module.StackOverflowError
StackUnderflowError = stack_module.StackUnderflowError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def empty_stack():
    """Return a fresh, empty Stack with the default limit of 10."""
    return Stack()


@pytest.fixture()
def small_stack():
    """Return a fresh, empty Stack with a limit of 3."""
    return Stack(3)


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------
def test_exception_types_are_base_exceptions() -> None:
    """Both custom errors derive from BaseException (not Exception)."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    assert not issubclass(StackOverflowError, Exception)
    assert not issubclass(StackUnderflowError, Exception)


def test_exception_types_are_distinct() -> None:
    """The two error classes must not be subclasses of one another."""
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_defaults() -> None:
    """A default Stack starts empty with a limit of 10."""
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
def test_bool_false_when_empty(empty_stack) -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_true_when_non_empty(empty_stack) -> None:
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
    """str(stack) mirrors the string form of the underlying list."""
    s = Stack(10)
    for item in items:
        s.push(item)
    assert str(s) == expected


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_single_element(empty_stack) -> None:
    """Pushing one element appends it to the top of the stack."""
    empty_stack.push(42)
    assert empty_stack.stack == [42]
    assert empty_stack.size() == 1


def test_push_multiple_elements_preserves_order(empty_stack) -> None:
    """Elements are stored in insertion order (bottom -> top)."""
    for value in (10, 20, 30):
        empty_stack.push(value)
    assert empty_stack.stack == [10, 20, 30]


def test_push_returns_none(empty_stack) -> None:
    """push() is a mutator and returns None."""
    assert empty_stack.push(1) is None


def test_push_overflow_raises(small_stack) -> None:
    """Pushing beyond the limit raises StackOverflowError."""
    small_stack.push(1)
    small_stack.push(2)
    small_stack.push(3)
    with pytest.raises(StackOverflowError):
        small_stack.push(4)


def test_push_overflow_does_not_mutate(small_stack) -> None:
    """A failed push leaves the stack contents unchanged."""
    for value in (1, 2, 3):
        small_stack.push(value)
    with pytest.raises(StackOverflowError):
        small_stack.push(99)
    assert small_stack.stack == [1, 2, 3]
    assert small_stack.size() == 3


def test_push_into_zero_limit_stack_raises() -> None:
    """A stack with limit 0 can never accept an element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)


def test_push_after_pop_succeeds(small_stack) -> None:
    """After popping, there is room to push again."""
    for value in (1, 2, 3):
        small_stack.push(value)
    small_stack.pop()
    small_stack.push(4)
    assert small_stack.stack == [1, 2, 4]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(empty_stack) -> None:
    """pop() returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10


def test_pop_removes_element(empty_stack) -> None:
    """pop() shrinks the stack by one."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.stack == [1]
    assert empty_stack.size() == 1


def test_pop_until_empty_then_underflow(empty_stack) -> None:
    """Popping an emptied stack raises StackUnderflowError."""
    empty_stack.push(1)
    assert empty_stack.pop() == 1
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_empty_stack_raises(empty_stack) -> None:
    """Popping a never-used stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_lifo_order(empty_stack) -> None:
    """Elements come off in reverse insertion order."""
    for value in (1, 2, 3, 4):
        empty_stack.push(value)
    assert [empty_stack.pop() for _ in range(4)] == [4, 3, 2, 1]


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(empty_stack) -> None:
    """peek() returns the top element and leaves the stack untouched."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2
    assert empty_stack.stack == [-5, 10]


def test_peek_single_element(empty_stack) -> None:
    """peek() on a one-element stack returns that element."""
    empty_stack.push("only")
    assert empty_stack.peek() == "only"


def test_peek_empty_stack_raises(empty_stack) -> None:
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_after_pop(empty_stack) -> None:
    """After a pop, peek() reflects the new top element."""
    for value in (1, 2, 3):
        empty_stack.push(value)
    empty_stack.pop()
    assert empty_stack.peek() == 2


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack(empty_stack) -> None:
    """A new stack reports itself as empty."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack) -> None:
    """A stack with an element is not empty."""
    empty_stack.push(10)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_all_popped(empty_stack) -> None:
    """Emptying the stack via pop() makes is_empty() true again."""
    empty_stack.push(10)
    empty_stack.pop()
    assert empty_stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------
def test_is_full_false_for_new_stack(empty_stack) -> None:
    """A new stack with a positive limit is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_at_limit() -> None:
    """A stack holding exactly the limit is full."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_below_limit(small_stack) -> None:
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
    """size() equals the number of elements pushed."""
    s = Stack(5)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_decreases_after_pop(empty_stack) -> None:
    """size() decreases by one for each pop."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
def test_contains_present_item(empty_stack) -> None:
    """`in` returns True for an element that was pushed."""
    empty_stack.push(10)
    assert (10 in empty_stack) is True


def test_contains_absent_item(empty_stack) -> None:
    """`in` returns False for an element that was never pushed."""
    empty_stack.push(10)
    assert (20 in empty_stack) is False


def test_contains_on_empty_stack(empty_stack) -> None:
    """`in` on an empty stack is always False."""
    assert (10 in empty_stack) is False


def test_contains_after_pop(empty_stack) -> None:
    """A popped element is no longer contained in the stack."""
    empty_stack.push(10)
    empty_stack.pop()
    assert (10 in empty_stack) is False


@pytest.mark.parametrize(
    ("items", "probe", "expected"),
    [
        ([1, 2, 3], 1, True),
        ([1, 2, 3], 3, True),
        ([1, 2, 3], 4, False),
        (["a"], "a", True),
        (["a"], "b", False),
        ([None], None, True),
    ],
)
def test_contains_parametrized(items: list, probe: object, expected: bool) -> None:
    """`in` behaves like membership on the underlying list."""
    s = Stack(10)
    for item in items:
        s.push(item)
    assert (probe in s) is expected


# ---------------------------------------------------------------------------
# Module-level helper: test_stack()
# ---------------------------------------------------------------------------
def test_module_level_test_stack_helper_runs() -> None:
    """The module's own `test_stack()` self-check executes without error."""
    assert stack_module.test_stack() is None


# ---------------------------------------------------------------------------
# Integration / end-to-end behaviour
# ---------------------------------------------------------------------------
def test_full_lifecycle() -> None:
    """Exercise a complete push/pop/peek/overflow/underflow lifecycle."""
    stack = Stack(10)
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

    assert bool(stack) is True
    assert stack.is_empty() is False
    assert stack.is_full() is True
    assert str(stack) == str(list(range(10)))
    assert stack.pop() == 9
    assert stack.peek() == 8

    stack.push(100)
    assert str(stack) == str([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])

    with pytest.raises(StackOverflowError):
        stack.push(200)

    assert stack.is_empty() is False
    assert stack.size() == 10
    assert 5 in stack
    assert 55 not in stack


def test_stack_holds_arbitrary_objects() -> None:
    """The stack is generic and stores any object type unchanged."""
    sentinel = object()
    s = Stack(2)
    s.push(sentinel)
    s.push({"k": "v"})
    assert s.peek() == {"k": "v"}
    assert s.pop() == {"k": "v"}
    assert s.pop() is sentinel
    assert s.is_empty() is True
