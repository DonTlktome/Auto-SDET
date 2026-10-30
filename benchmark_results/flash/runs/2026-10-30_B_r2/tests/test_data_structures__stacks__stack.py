"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.  The module must be registered in sys.modules
# under its real dotted path, because its docstrings expect exception names of
# the form ``data_structures.stacks.stack.StackOverflowError`` (doctest compares
# the exception message text, which is derived from ``cls.__module__``).
# ---------------------------------------------------------------------------
_MODULE_NAME = "data_structures.stacks.stack"
_SOURCE_RELATIVE = Path("data_structures") / "stacks" / "stack.py"


def _candidate_roots() -> List[Path]:
    """Collect plausible project roots (test dir, its parents, cwd, cwd parents)."""
    roots: List[Path] = []
    here = Path(__file__).resolve()
    roots.append(here.parent)
    roots.extend(here.parent.parents)
    cwd = Path.cwd()
    roots.append(cwd)
    roots.extend(cwd.parents)
    return roots


def _find_source_path() -> Path:
    """Locate ``data_structures/stacks/stack.py`` without relying on one OS path."""
    candidates = [
        # Original (Windows) location, kept as the first try for local runs.
        Path(
            r"D:\Documents\Study\Projects\algorithms-bench\data_structures\stacks\stack.py"
        ),
    ]
    for root in _candidate_roots():
        candidates.append(root / _SOURCE_RELATIVE)
        candidates.append(root / "stack.py")

    checked = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - unreadable path
            continue
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved

    # Last resort: search the working tree for the expected layout.
    for root in (Path.cwd(), Path(__file__).resolve().parent):
        try:
            for match in root.rglob("stack.py"):
                if match.parent.name == "stacks" and match.is_file():
                    return match
        except OSError:  # pragma: no cover - unreadable directory
            continue

    raise FileNotFoundError(
        "Could not locate data_structures/stacks/stack.py; searched: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_module() -> ModuleType:
    """Import the target module, falling back to loading it from its file."""
    try:
        return importlib.import_module(_MODULE_NAME)
    except Exception:  # pragma: no cover - project root not on sys.path
        pass

    source_path = _find_source_path()
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Could not create an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
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
    """Return a fresh, empty stack with the default limit."""
    return Stack()


@pytest.fixture()
def small_stack() -> "Stack[int]":
    """Return a fresh, empty stack with a limit of 3."""
    return Stack(3)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------
def test_stack_overflow_error_is_base_exception_subclass() -> None:
    """StackOverflowError must derive from BaseException (not Exception)."""
    assert issubclass(StackOverflowError, BaseException)
    assert not issubclass(StackOverflowError, Exception)


def test_stack_underflow_error_is_base_exception_subclass() -> None:
    """StackUnderflowError must derive from BaseException (not Exception)."""
    assert issubclass(StackUnderflowError, BaseException)
    assert not issubclass(StackUnderflowError, Exception)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_default_limit_and_empty_storage() -> None:
    """A new stack has an empty list and the default limit of 10."""
    s = Stack()
    assert s.stack == []
    assert s.limit == 10


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit: int) -> None:
    """A custom limit is stored verbatim and storage starts empty."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.stack == []


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------
def test_bool_false_when_empty(empty_stack: "Stack[int]") -> None:
    """An empty stack is falsy."""
    assert bool(empty_stack) is False


def test_bool_true_when_non_empty(empty_stack: "Stack[int]") -> None:
    """A stack with at least one element is truthy."""
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
def test_push_adds_to_top(empty_stack: "Stack[int]") -> None:
    """push appends the element to the end (top) of the stack."""
    empty_stack.push(10)
    empty_stack.push(20)
    assert empty_stack.stack == [10, 20]
    assert empty_stack.peek() == 20


def test_push_returns_none(empty_stack: "Stack[int]") -> None:
    """push returns None."""
    assert empty_stack.push(1) is None


def test_push_raises_overflow_when_full() -> None:
    """Pushing onto a full stack raises StackOverflowError."""
    s = Stack(1)
    s.push(10)
    with pytest.raises(StackOverflowError):
        s.push(20)


def test_push_raises_overflow_at_exact_limit() -> None:
    """The limit is inclusive: the (limit+1)-th push overflows."""
    s = Stack(3)
    for i in range(3):
        s.push(i)
    assert s.size() == 3
    with pytest.raises(StackOverflowError):
        s.push(3)


def test_push_raises_overflow_on_zero_limit() -> None:
    """A stack with limit 0 cannot accept any element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)


def test_push_after_pop_succeeds() -> None:
    """After popping, there is room to push again."""
    s = Stack(2)
    s.push(1)
    s.push(2)
    assert s.pop() == 2
    s.push(3)
    assert s.stack == [1, 3]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(empty_stack: "Stack[int]") -> None:
    """pop removes and returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10
    assert empty_stack.pop() == -5


def test_pop_raises_underflow_when_empty(empty_stack: "Stack[int]") -> None:
    """Popping an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_raises_underflow_after_draining() -> None:
    """Popping one element too many raises StackUnderflowError."""
    s = Stack(2)
    s.push(1)
    assert s.pop() == 1
    with pytest.raises(StackUnderflowError):
        s.pop()


def test_pop_lifo_order_with_many_elements() -> None:
    """Elements come off in reverse insertion order."""
    s = Stack(5)
    for i in range(5):
        s.push(i)
    assert [s.pop() for _ in range(5)] == [4, 3, 2, 1, 0]
    assert s.is_empty() is True


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(empty_stack: "Stack[int]") -> None:
    """peek returns the top element and leaves the stack unchanged."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2
    assert empty_stack.peek() == 10


def test_peek_raises_underflow_when_empty(empty_stack: "Stack[int]") -> None:
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_single_element() -> None:
    """peek on a one-element stack returns that element."""
    s = Stack(1)
    s.push("only")
    assert s.peek() == "only"


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A new stack is empty."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack: "Stack[int]") -> None:
    """A stack with an element is not empty."""
    empty_stack.push(10)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_push_pop(empty_stack: "Stack[int]") -> None:
    """Pushing then popping returns the stack to the empty state."""
    empty_stack.push(10)
    empty_stack.pop()
    assert empty_stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------
def test_is_full_false_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A new stack with a positive limit is not full."""
    assert empty_stack.is_full() is False


def test_is_full_true_when_limit_reached() -> None:
    """A stack is full once size equals limit."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_just_below_limit() -> None:
    """A stack one element short of its limit is not full."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert s.is_full() is False
    s.push(3)
    assert s.is_full() is True


def test_is_full_true_for_zero_limit() -> None:
    """A zero-limit stack is full from the start."""
    assert Stack(0).is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """size equals the number of elements currently on the stack."""
    s = Stack(3)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_decreases_after_pop() -> None:
    """size decreases by one after each pop."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert s.size() == 2
    s.pop()
    assert s.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------
def test_contains_present_item() -> None:
    """`in` returns True for an element on the stack."""
    s = Stack(3)
    s.push(10)
    assert 10 in s


def test_contains_absent_item() -> None:
    """`in` returns False for an element not on the stack."""
    s = Stack(3)
    s.push(10)
    assert 20 not in s


def test_contains_on_empty_stack(empty_stack: "Stack[int]") -> None:
    """`in` returns False on an empty stack."""
    assert 1 not in empty_stack


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
# Integration / module-level test_stack function
# ---------------------------------------------------------------------------
def test_module_level_test_stack_runs() -> None:
    """The module's own `test_stack` helper executes without raising."""
    stack_module.test_stack()


def test_full_lifecycle_scenario() -> None:
    """End-to-end scenario covering push/pop/peek/size/contains together."""
    s = Stack(10)
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

    assert s.is_empty() is False
    assert s.size() == 10
    assert 5 in s
    assert 55 not in s


def test_doctests_pass() -> None:
    """All doctests embedded in the source module must pass."""
    import doctest

    results = doctest.testmod(stack_module, verbose=False)
    assert results.failed == 0
