"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
#
# The source file uses PEP 695 generic syntax (``class Stack[T]``) and is not
# guaranteed to be importable through the normal package machinery (missing
# ``__init__.py`` files, tests executed outside the project tree, absolute
# paths that differ between the authoring machine and the CI container, ...).
# We therefore locate the file on disk at run time and load it by path.
#
# The module is registered under the package-qualified name used by its own
# docstrings (``data_structures.stacks.stack``) so that doctest tracebacks
# such as ``data_structures.stacks.stack.StackOverflowError`` match exactly.
# ---------------------------------------------------------------------------
_SOURCE_FILENAME = "stack.py"
_PACKAGE_PARTS = ("data_structures", "stacks")
_MODULE_NAME = "data_structures.stacks.stack"
_MAX_SEARCH_DEPTH = 6
_IGNORED_DIR_NAMES = {"node_modules", "venv", ".venv", "site-packages", "__pycache__"}


def _looks_like_target(path: Path) -> bool:
    """Return True when *path* looks like the stack implementation."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "class Stack" in text and "StackOverflowError" in text


def _ancestor_dirs():
    """Yield candidate base directories: ancestors of the test file and cwd."""
    seen = set()
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _walk_limited(root: Path, max_depth: int = _MAX_SEARCH_DEPTH):
    """Yield ``(directory, filenames)`` pairs below *root*, depth limited."""
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        relative = Path(dirpath).relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith((".", "__")) and name not in _IGNORED_DIR_NAMES
        ]
        if len(relative.parts) >= max_depth:
            dirnames[:] = []
        yield Path(dirpath), filenames


def _find_source_file() -> Path:
    """Locate ``stack.py`` (the module under test) on the local filesystem."""
    env_path = os.environ.get("STACK_SOURCE_PATH")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    # 1) Conventional layouts relative to any ancestor directory.
    for base in _ancestor_dirs():
        for parts in (_PACKAGE_PARTS, ("stacks",), ()):
            candidate = base.joinpath(*parts, _SOURCE_FILENAME)
            if candidate.is_file() and _looks_like_target(candidate):
                return candidate

    # 2) Bounded walk below the test directory and the current working dir.
    searched = set()
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        resolved = start.resolve()
        if resolved in searched:
            continue
        searched.add(resolved)
        for directory, filenames in _walk_limited(resolved):
            if _SOURCE_FILENAME in filenames:
                candidate = directory / _SOURCE_FILENAME
                if _looks_like_target(candidate):
                    return candidate

    raise FileNotFoundError(
        f"Could not locate {_SOURCE_FILENAME!r} for the module under test"
    )


def _load_module():
    """Load the target module from the file path found on disk."""
    source = _find_source_file()
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


stack_mod = _load_module()

Stack = stack_mod.Stack
StackOverflowError = stack_mod.StackOverflowError
StackUnderflowError = stack_mod.StackUnderflowError


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_stack_overflow_error_is_exception_subclass():
    """StackOverflowError must be raisable/catchable as a BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    with pytest.raises(StackOverflowError):
        raise StackOverflowError


def test_stack_underflow_error_is_exception_subclass():
    """StackUnderflowError must be raisable/catchable as a BaseException."""
    assert issubclass(StackUnderflowError, BaseException)
    with pytest.raises(StackUnderflowError):
        raise StackUnderflowError


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_default_limit_and_empty_storage():
    """A new Stack defaults to limit 10 and an empty internal list."""
    s = Stack()
    assert s.limit == 10
    assert s.stack == []
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit):
    """A custom limit is stored verbatim and storage starts empty."""
    s = Stack(limit)
    assert s.limit == limit
    assert s.stack == []


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------


def test_bool_empty_stack_is_false():
    """An empty stack is falsy."""
    assert bool(Stack()) is False


def test_bool_non_empty_stack_is_true():
    """A stack holding at least one element is truthy."""
    s = Stack()
    s.push(1)
    assert bool(s) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


def test_str_empty_stack():
    """str() of an empty stack renders the empty list."""
    assert str(Stack()) == "[]"


def test_str_matches_internal_list():
    """str() mirrors the internal list representation."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert str(s) == "[1, 2]"
    assert str(s) == str(s.stack)


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


def test_push_adds_to_top():
    """push() appends the element to the top of the stack."""
    s = Stack(3)
    s.push(10)
    s.push(20)
    assert s.stack == [10, 20]
    assert s.peek() == 20


def test_push_returns_none():
    """push() returns None."""
    s = Stack(2)
    assert s.push(1) is None


def test_push_raises_overflow_when_full():
    """push() raises StackOverflowError once the limit is reached."""
    s = Stack(1)
    s.push(10)
    with pytest.raises(StackOverflowError):
        s.push(20)
    # The failed push must not mutate the stack.
    assert s.stack == [10]


def test_push_on_zero_limit_stack_always_overflows():
    """A stack with limit 0 cannot accept any element."""
    s = Stack(0)
    with pytest.raises(StackOverflowError):
        s.push(1)
    assert s.size() == 0


@pytest.mark.parametrize(
    "value",
    [0, -1, 3.14, "text", None, (1, 2), [3, 4], {"a": 1}],
)
def test_push_accepts_arbitrary_types(value):
    """push() stores any object unchanged (generic container)."""
    s = Stack(1)
    s.push(value)
    assert s.peek() == value
    assert s.size() == 1


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------


def test_pop_returns_last_in_first_out():
    """pop() removes and returns the most recently pushed element (LIFO)."""
    s = Stack()
    s.push(-5)
    s.push(10)
    assert s.pop() == 10
    assert s.pop() == -5
    assert s.is_empty()


def test_pop_raises_underflow_on_empty_stack():
    """pop() on an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        Stack().pop()


def test_pop_after_emptying_raises_underflow():
    """Popping past the last element raises StackUnderflowError."""
    s = Stack(2)
    s.push(1)
    s.pop()
    with pytest.raises(StackUnderflowError):
        s.pop()


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------


def test_peek_returns_top_without_removing():
    """peek() returns the top element and leaves the stack unchanged."""
    s = Stack()
    s.push(-5)
    s.push(10)
    assert s.peek() == 10
    assert s.size() == 2
    assert s.peek() == 10


def test_peek_raises_underflow_on_empty_stack():
    """peek() on an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        Stack().peek()


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_is_empty_true_for_new_stack():
    """A freshly created stack reports empty."""
    assert Stack().is_empty() is True


def test_is_empty_false_after_push_and_true_after_pop():
    """is_empty() tracks push/pop transitions."""
    s = Stack()
    s.push(10)
    assert s.is_empty() is False
    s.pop()
    assert s.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------


def test_is_full_false_for_new_stack():
    """A new stack with a positive limit is not full."""
    assert Stack().is_full() is False


def test_is_full_true_when_size_equals_limit():
    """is_full() becomes True exactly when size reaches the limit."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_true_for_zero_limit_stack():
    """A limit-0 stack is full from the start (size == limit == 0)."""
    assert Stack(0).is_full() is True


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_is_full_boundary(limit):
    """is_full() flips only at the exact boundary size == limit."""
    s = Stack(limit)
    for i in range(limit - 1):
        s.push(i)
        assert s.is_full() is False
    s.push(limit - 1)
    assert s.is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------


def test_size_zero_for_new_stack():
    """A new stack has size 0."""
    assert Stack(3).size() == 0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_size_grows_with_pushes(count):
    """size() equals the number of pushed elements."""
    s = Stack(5)
    for i in range(count):
        s.push(i)
        assert s.size() == i + 1
    assert s.size() == count


def test_size_decreases_after_pop():
    """size() decreases by one per pop."""
    s = Stack(3)
    s.push(1)
    s.push(2)
    assert s.size() == 2
    s.pop()
    assert s.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


def test_contains_present_item():
    """`in` returns True for an element on the stack."""
    s = Stack(3)
    s.push(10)
    assert (10 in s) is True


def test_contains_absent_item():
    """`in` returns False for an element not on the stack."""
    s = Stack(3)
    s.push(10)
    assert (20 in s) is False


def test_contains_on_empty_stack():
    """`in` returns False on an empty stack."""
    assert (1 in Stack()) is False


@pytest.mark.parametrize(
    ("items", "needle", "expected"),
    [
        ([1, 2, 3], 2, True),
        ([1, 2, 3], 4, False),
        (["a", "b"], "b", True),
        (["a", "b"], "c", False),
        ([None], None, True),
        ([0], False, True),  # 0 == False in Python
    ],
)
def test_contains_parametrized(items, needle, expected):
    """`in` behaves like membership on the underlying list."""
    s = Stack(len(items) or 1)
    for item in items:
        s.push(item)
    assert (needle in s) is expected


# ---------------------------------------------------------------------------
# Integration / LIFO behaviour
# ---------------------------------------------------------------------------


def test_lifo_ordering_full_cycle():
    """Elements come off in reverse order of insertion (LIFO)."""
    s = Stack(5)
    for i in range(5):
        s.push(i)
    popped = [s.pop() for _ in range(5)]
    assert popped == [4, 3, 2, 1, 0]
    assert s.is_empty() is True


def test_module_level_test_stack_function_runs():
    """The module's own self-test helper executes without error."""
    assert stack_mod.test_stack() is None


def test_doctests_pass():
    """All doctests embedded in the module must pass."""
    import doctest

    results = doctest.testmod(
        stack_mod,
        verbose=False,
        optionflags=doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert results.failed == 0
