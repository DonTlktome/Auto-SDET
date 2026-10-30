"""Pytest test suite for data_structures/stacks/stack.py.

The module under test defines a generic LIFO ``Stack`` with a fixed
``limit``, plus ``StackOverflowError`` and ``StackUnderflowError``
exceptions (both deriving from ``BaseException``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The source file uses PEP 695 generic syntax (``class Stack[T]``), so it is
# imported dynamically from its file path to avoid depending on package
# layout / ``__init__.py`` files.  It is registered under the dotted name
# ``data_structures.stacks.stack`` because the doctests embedded in the source
# compare fully-qualified exception names such as
# ``data_structures.stacks.stack.StackOverflowError``.
# ---------------------------------------------------------------------------
MODULE_NAME = "data_structures.stacks.stack"

_TEST_DIR = Path(__file__).resolve().parent

_CANDIDATE_SOURCES = (
    _TEST_DIR / "stack.py",
    _TEST_DIR / "data_structures" / "stacks" / "stack.py",
    _TEST_DIR.parent / "data_structures" / "stacks" / "stack.py",
    Path.cwd() / "data_structures" / "stacks" / "stack.py",
    Path.cwd() / "stack.py",
)


def _looks_like_target(path: Path) -> bool:
    """Return True when ``path`` looks like the stack module under test."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:  # pragma: no cover - defensive
        return False
    return "class StackOverflowError" in text and "class StackUnderflowError" in text


def _find_source() -> Path:
    """Locate ``stack.py`` without relying on a hard-coded absolute path."""
    candidates = []
    env_override = os.environ.get("STACK_MODULE_PATH")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend(_CANDIDATE_SOURCES)

    for candidate in candidates:
        try:
            if candidate.is_file() and _looks_like_target(candidate):
                return candidate
        except OSError:  # pragma: no cover - defensive
            continue

    for root in (_TEST_DIR, Path.cwd()):
        try:
            for candidate in sorted(root.rglob("stack.py")):
                if _looks_like_target(candidate):
                    return candidate
        except OSError:  # pragma: no cover - defensive
            continue

    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError("Unable to locate stack.py; tried:\n" + searched)


def _register_parent_packages(module_name: str) -> None:
    """Create placeholder packages so ``module_name`` is importable by name."""
    parts = module_name.split(".")[:-1]
    for index in range(1, len(parts) + 1):
        package_name = ".".join(parts[:index])
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        if index > 1:
            parent = sys.modules[".".join(parts[: index - 1])]
            setattr(parent, parts[index - 1], package)
        sys.modules[package_name] = package


def _load_module():
    """Load the target module from its file path under its dotted name."""
    source = _find_source()
    _register_parent_packages(MODULE_NAME)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to build an import spec for {source}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module

    parent_name, _, child_name = MODULE_NAME.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)

    spec.loader.exec_module(module)
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
    """Return a fresh, empty stack with the default limit of 10."""
    return Stack()


@pytest.fixture()
def small_stack() -> "Stack[int]":
    """Return a fresh, empty stack with a limit of 3."""
    return Stack(3)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------
def test_exception_classes_are_base_exception_subclasses() -> None:
    """Both custom exceptions must derive from BaseException (not Exception)."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    assert not issubclass(StackOverflowError, Exception)
    assert not issubclass(StackUnderflowError, Exception)


def test_exception_classes_are_distinct() -> None:
    """The two exception types must not be related to each other."""
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_default_limit_and_empty_storage() -> None:
    """A default stack starts empty with a limit of 10."""
    s = Stack()
    assert s.limit == 10
    assert s.stack == []
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit: int) -> None:
    """A custom limit is stored verbatim and storage starts empty."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.stack == []


def test_init_negative_limit_is_stored() -> None:
    """A negative limit is accepted at construction time (no validation)."""
    s = Stack(-1)
    assert s.limit == -1
    assert s.is_empty() is True


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------
def test_bool_false_when_empty(empty_stack: "Stack[int]") -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_true_after_push(empty_stack: "Stack[int]") -> None:
    """A stack holding at least one element is truthy."""
    empty_stack.push(1)
    assert bool(empty_stack) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
def test_str_empty_stack(empty_stack: "Stack[int]") -> None:
    """The string form of an empty stack is the empty list literal."""
    assert str(empty_stack) == "[]"


def test_str_reflects_contents_in_push_order(empty_stack: "Stack[int]") -> None:
    """The string form mirrors the internal list, bottom-to-top."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.push(3)
    assert str(empty_stack) == "[1, 2, 3]"


def test_str_with_mixed_types() -> None:
    """The string form works for heterogeneous element types."""
    s = Stack(3)
    s.push("a")
    s.push(None)
    assert str(s) == "['a', None]"


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_single_element(empty_stack: "Stack[int]") -> None:
    """Pushing onto an empty stack adds the element on top."""
    empty_stack.push(42)
    assert empty_stack.size() == 1
    assert empty_stack.peek() == 42


def test_push_returns_none(empty_stack: "Stack[int]") -> None:
    """``push`` is a mutator and returns ``None``."""
    assert empty_stack.push(1) is None


def test_push_up_to_limit(small_stack: "Stack[int]") -> None:
    """Pushing exactly ``limit`` elements succeeds and fills the stack."""
    for i in range(3):
        small_stack.push(i)
    assert small_stack.size() == 3
    assert small_stack.is_full() is True


def test_push_beyond_limit_raises_overflow(small_stack: "Stack[int]") -> None:
    """Pushing past the limit raises ``StackOverflowError``."""
    for i in range(3):
        small_stack.push(i)
    with pytest.raises(StackOverflowError):
        small_stack.push(99)


def test_push_overflow_does_not_mutate_stack(small_stack: "Stack[int]") -> None:
    """A failed push must leave the stack contents untouched."""
    small_stack.push(1)
    small_stack.push(2)
    small_stack.push(3)
    with pytest.raises(StackOverflowError):
        small_stack.push(4)
    assert small_stack.stack == [1, 2, 3]
    assert small_stack.size() == 3


def test_push_on_zero_limit_stack_raises_overflow() -> None:
    """A stack with limit 0 can never accept an element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)


def test_push_after_pop_reuses_capacity(small_stack: "Stack[int]") -> None:
    """Popping frees a slot so a subsequent push succeeds."""
    small_stack.push(1)
    small_stack.push(2)
    small_stack.push(3)
    assert small_stack.pop() == 3
    small_stack.push(4)  # must not raise
    assert small_stack.stack == [1, 2, 4]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(empty_stack: "Stack[int]") -> None:
    """``pop`` returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10


def test_pop_removes_element(empty_stack: "Stack[int]") -> None:
    """``pop`` decreases the size by one."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.size() == 1
    assert empty_stack.peek() == 1


def test_pop_empty_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Popping an empty stack raises ``StackUnderflowError``."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_until_empty_then_underflow(small_stack: "Stack[int]") -> None:
    """Draining the stack works, and one extra pop underflows."""
    small_stack.push(1)
    small_stack.push(2)
    assert small_stack.pop() == 2
    assert small_stack.pop() == 1
    assert small_stack.is_empty() is True
    with pytest.raises(StackUnderflowError):
        small_stack.pop()


def test_pop_lifo_order() -> None:
    """Elements come off in reverse insertion order."""
    s = Stack(5)
    for value in (1, 2, 3, 4, 5):
        s.push(value)
    assert [s.pop() for _ in range(5)] == [5, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(empty_stack: "Stack[int]") -> None:
    """``peek`` returns the top element and leaves the size unchanged."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2


def test_peek_empty_raises_underflow(empty_stack: "Stack[int]") -> None:
    """Peeking an empty stack raises ``StackUnderflowError``."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_single_element() -> None:
    """Peeking a one-element stack returns that element."""
    s = Stack(1)
    s.push("only")
    assert s.peek() == "only"


def test_peek_after_pop(empty_stack: "Stack[int]") -> None:
    """After a pop, ``peek`` reflects the new top element."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.peek() == 1


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


def test_is_empty_true_after_draining(small_stack: "Stack[int]") -> None:
    """A stack becomes empty again once all elements are popped."""
    small_stack.push(1)
    small_stack.pop()
    assert small_stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------
def test_is_full_false_for_new_default_stack(empty_stack: "Stack[int]") -> None:
    """A fresh default stack is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_when_limit_reached() -> None:
    """A stack holding ``limit`` elements reports full."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_below_limit(small_stack: "Stack[int]") -> None:
    """A stack below its limit is not full."""
    small_stack.push(1)
    small_stack.push(2)
    assert small_stack.is_full() is False


def test_is_full_true_for_zero_limit_stack() -> None:
    """A zero-limit stack is full from the start."""
    assert Stack(0).is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """``size`` equals the number of elements pushed."""
    s = Stack(3)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_decreases_after_pop(small_stack: "Stack[int]") -> None:
    """``size`` decreases by one per pop."""
    small_stack.push(1)
    small_stack.push(2)
    small_stack.pop()
    assert small_stack.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
def test_contains_present_item(small_stack: "Stack[int]") -> None:
    """``in`` is True for an element that was pushed."""
    small_stack.push(10)
    assert 10 in small_stack


def test_contains_absent_item(small_stack: "Stack[int]") -> None:
    """``in`` is False for an element that was never pushed."""
    small_stack.push(10)
    assert 20 not in small_stack


def test_contains_on_empty_stack(empty_stack: "Stack[int]") -> None:
    """``in`` is False for any item on an empty stack."""
    assert 1 not in empty_stack


def test_contains_after_pop(small_stack: "Stack[int]") -> None:
    """A popped element is no longer contained in the stack."""
    small_stack.push(7)
    small_stack.pop()
    assert 7 not in small_stack


def test_contains_with_unhashable_item() -> None:
    """Membership uses list semantics, so unhashable items work too."""
    s = Stack(2)
    s.push([1, 2])
    assert [1, 2] in s
    assert [3] not in s


# ---------------------------------------------------------------------------
# Integration / end-to-end
# ---------------------------------------------------------------------------
def test_full_lifecycle_matches_module_self_test() -> None:
    """Replicate the module's own ``test_stack`` scenario end-to-end."""
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


def test_module_level_test_stack_function_runs() -> None:
    """The module's own ``test_stack`` helper executes without error."""
    stack_module.test_stack()


def test_doctests_pass() -> None:
    """All doctests embedded in the module must pass."""
    import doctest

    results = doctest.testmod(stack_module, verbose=False)
    assert results.failed == 0
