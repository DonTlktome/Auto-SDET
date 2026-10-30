"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the target module by file path so the test file is self-contained and
# does not depend on the project being installed as a package.
#
# The path recorded on the original development machine
# (D:\Documents\Study\Projects\algorithms-bench\...) does not exist on the
# machine that runs the tests, so the file is located dynamically instead and
# the module is registered under its natural dotted name so that the doctests
# (which reference ``data_structures.stacks.stack.*`` in their tracebacks) can
# be compared successfully.
# ---------------------------------------------------------------------------
_RELATIVE_SOURCE = Path("data_structures") / "stacks" / "stack.py"
_TEST_DIR = Path(__file__).resolve().parent


def _search_roots() -> list[Path]:
    """Return directories to search, closest to the test file first."""
    roots: list[Path] = []
    for root in [_TEST_DIR, *_TEST_DIR.parents]:
        if root == Path(root.anchor):  # filesystem root: far too expensive
            break
        roots.append(root)
        if len(roots) >= 4:
            break
    return roots


def _candidate_paths() -> list[Path]:
    """All plausible locations of the source module, cheapest first."""
    candidates: list[Path] = []

    env_path = os.environ.get("STACK_SOURCE_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Location recorded on the original (Windows) development machine.
    candidates.append(
        Path(r"D:\Documents\Study\Projects\algorithms-bench") / _RELATIVE_SOURCE
    )

    for root in _search_roots():
        candidates.append(root / _RELATIVE_SOURCE)
        candidates.append(root / "stacks" / _RELATIVE_SOURCE.name)
        candidates.append(root / _RELATIVE_SOURCE.name)

    return candidates


def _locate_source() -> Path:
    """Find the stack source file on disk, wherever the repo is checked out."""
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate.resolve()

    for root in _search_roots():
        try:
            matches = sorted(root.glob(f"**/stacks/{_RELATIVE_SOURCE.name}"))
        except (OSError, ValueError):
            continue
        for match in matches:
            if match.is_file():
                return match.resolve()

    raise FileNotFoundError(
        "Could not locate data_structures/stacks/stack.py; looked at: "
        + ", ".join(str(path) for path in _candidate_paths())
    )


def _module_name_for(path: Path) -> str:
    """Dotted module name mirroring the on-disk package layout."""
    names = [path.stem] + [parent.name for parent in path.parents]
    if names[1:3] == ["stacks", "data_structures"]:
        return "data_structures.stacks.stack"

    parts = [path.stem]
    parent = path.parent
    while (parent / "__init__.py").is_file():
        parts.insert(0, parent.name)
        parent = parent.parent
    return ".".join(parts)


def _register_parent_packages(name: str) -> None:
    """Make sure every parent package of ``name`` is importable."""
    parts = name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name in sys.modules:
            continue
        try:
            importlib.import_module(package_name)
            continue
        except Exception:  # noqa: BLE001 - fall back to a synthetic package
            pass
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package_name] = package


def _load_module() -> types.ModuleType:
    """Load the stack module from the located file path."""
    source_path = _locate_source()
    module_name = _module_name_for(source_path)

    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file:
            try:
                if Path(existing_file).resolve() == source_path:
                    return existing
            except OSError:
                pass

    _register_parent_packages(module_name)

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
def test_exception_classes_are_defined() -> None:
    """Both custom exceptions exist and derive from BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)


def test_exception_classes_are_distinct() -> None:
    """The two custom exceptions are not related to each other."""
    assert not issubclass(StackOverflowError, StackUnderflowError)
    assert not issubclass(StackUnderflowError, StackOverflowError)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_defaults() -> None:
    """A new stack is empty and has the documented default limit of 10."""
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


def test_bool_non_empty_stack_is_true(empty_stack: "Stack[int]") -> None:
    """A stack holding at least one element is truthy."""
    empty_stack.push(1)
    assert bool(empty_stack) is True


def test_bool_after_pop_back_to_empty(empty_stack: "Stack[int]") -> None:
    """Popping the last element makes the stack falsy again."""
    empty_stack.push(1)
    empty_stack.pop()
    assert bool(empty_stack) is False


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
def test_str_empty_stack(empty_stack: "Stack[int]") -> None:
    """str() of an empty stack is the empty list representation."""
    assert str(empty_stack) == "[]"


def test_str_with_elements(small_stack: "Stack[int]") -> None:
    """str() mirrors the underlying list, bottom element first."""
    small_stack.push(1)
    small_stack.push(2)
    assert str(small_stack) == "[1, 2]"


def test_str_with_mixed_types() -> None:
    """str() works for heterogeneous element types."""
    s = Stack(3)
    s.push("a")
    s.push(None)
    assert str(s) == "['a', None]"


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_single_element(empty_stack: "Stack[int]") -> None:
    """Pushing one element stores it and increases the size."""
    empty_stack.push(42)
    assert empty_stack.size() == 1
    assert empty_stack.peek() == 42


def test_push_returns_none(empty_stack: "Stack[int]") -> None:
    """push() returns None."""
    assert empty_stack.push(1) is None


def test_push_preserves_order(small_stack: "Stack[int]") -> None:
    """Elements are stored in insertion order (bottom to top)."""
    for value in (1, 2, 3):
        small_stack.push(value)
    assert small_stack.stack == [1, 2, 3]


def test_push_overflow_raises(small_stack: "Stack[int]") -> None:
    """Pushing beyond the limit raises StackOverflowError."""
    for value in (1, 2, 3):
        small_stack.push(value)
    with pytest.raises(StackOverflowError):
        small_stack.push(4)


def test_push_overflow_does_not_mutate(small_stack: "Stack[int]") -> None:
    """A failed push leaves the stack contents unchanged."""
    for value in (1, 2, 3):
        small_stack.push(value)
    with pytest.raises(StackOverflowError):
        small_stack.push(99)
    assert small_stack.stack == [1, 2, 3]
    assert small_stack.size() == 3


def test_push_into_zero_limit_stack_raises() -> None:
    """A stack with limit 0 cannot accept any element."""
    s = Stack(0)
    assert s.is_full() is True
    with pytest.raises(StackOverflowError):
        s.push(1)


def test_push_after_pop_succeeds(small_stack: "Stack[int]") -> None:
    """Space freed by pop() can be reused by push()."""
    for value in (1, 2, 3):
        small_stack.push(value)
    small_stack.pop()
    small_stack.push(4)
    assert small_stack.stack == [1, 2, 4]


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed(empty_stack: "Stack[int]") -> None:
    """pop() returns the most recently pushed element (LIFO)."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.pop() == 10


def test_pop_decreases_size(empty_stack: "Stack[int]") -> None:
    """pop() reduces the size by one."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.size() == 1


def test_pop_empty_raises(empty_stack: "Stack[int]") -> None:
    """Popping an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.pop()


def test_pop_until_empty_then_raises(small_stack: "Stack[int]") -> None:
    """Popping all elements then popping again raises StackUnderflowError."""
    small_stack.push(1)
    assert small_stack.pop() == 1
    with pytest.raises(StackUnderflowError):
        small_stack.pop()


def test_pop_lifo_order() -> None:
    """Repeated pops yield elements in reverse insertion order."""
    s = Stack(5)
    for value in (1, 2, 3):
        s.push(value)
    assert [s.pop(), s.pop(), s.pop()] == [3, 2, 1]


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------
def test_peek_returns_top_without_removing(empty_stack: "Stack[int]") -> None:
    """peek() returns the top element and leaves the stack unchanged."""
    empty_stack.push(-5)
    empty_stack.push(10)
    assert empty_stack.peek() == 10
    assert empty_stack.size() == 2


def test_peek_empty_raises(empty_stack: "Stack[int]") -> None:
    """Peeking an empty stack raises StackUnderflowError."""
    with pytest.raises(StackUnderflowError):
        empty_stack.peek()


def test_peek_single_element(empty_stack: "Stack[int]") -> None:
    """peek() on a one-element stack returns that element."""
    empty_stack.push("only")
    assert empty_stack.peek() == "only"


def test_peek_after_pop(empty_stack: "Stack[int]") -> None:
    """After a pop, peek() reflects the new top element."""
    empty_stack.push(1)
    empty_stack.push(2)
    empty_stack.pop()
    assert empty_stack.peek() == 1


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
def test_is_empty_true_for_new_stack(empty_stack: "Stack[int]") -> None:
    """A new stack reports is_empty() as True."""
    assert empty_stack.is_empty() is True


def test_is_empty_false_after_push(empty_stack: "Stack[int]") -> None:
    """A stack with an element reports is_empty() as False."""
    empty_stack.push(10)
    assert empty_stack.is_empty() is False


def test_is_empty_true_after_all_popped(empty_stack: "Stack[int]") -> None:
    """is_empty() is True again once every element has been popped."""
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
    """is_full() becomes True exactly when size equals limit."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_false_below_limit(small_stack: "Stack[int]") -> None:
    """is_full() stays False while size is below the limit."""
    small_stack.push(1)
    small_stack.push(2)
    assert small_stack.is_full() is False


def test_is_full_true_for_zero_limit() -> None:
    """A zero-limit stack is full from the start."""
    assert Stack(0).is_full() is True


def test_is_full_false_after_pop(small_stack: "Stack[int]") -> None:
    """Popping from a full stack makes is_full() False."""
    for value in (1, 2, 3):
        small_stack.push(value)
    assert small_stack.is_full() is True
    small_stack.pop()
    assert small_stack.is_full() is False


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count: int) -> None:
    """size() equals the number of elements pushed."""
    s = Stack(3)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_after_pop(empty_stack: "Stack[int]") -> None:
    """size() decreases after a pop."""
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


def test_contains_after_pop(small_stack: "Stack[int]") -> None:
    """A popped element is no longer contained in the stack."""
    small_stack.push(10)
    small_stack.pop()
    assert 10 not in small_stack


def test_contains_duplicates(small_stack: "Stack[int]") -> None:
    """Duplicate values are still reported as contained."""
    small_stack.push(7)
    small_stack.push(7)
    assert 7 in small_stack


# ---------------------------------------------------------------------------
# Integration / module-level test function
# ---------------------------------------------------------------------------
def test_module_level_test_stack_runs() -> None:
    """The module's own `test_stack` helper executes without raising."""
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
    """All doctests embedded in the source module pass."""
    results = doctest.testmod(
        stack_module,
        verbose=False,
        optionflags=doctest.IGNORE_EXCEPTION_DETAIL | doctest.ELLIPSIS,
    )
    assert results.failed == 0
