"""Tests for data_structures.stacks.stack."""

from __future__ import annotations

import doctest
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate and import the module under test.  The original file may live in a
# package layout (data_structures/stacks/stack.py) or next to the tests, and
# the test-suite may run on Windows or Linux, so several strategies are used.
# ---------------------------------------------------------------------------
_WINDOWS_SOURCE_PATH = Path(
    r"D:\Documents\Study\Projects\algorithms-bench\data_structures\stacks\stack.py"
)

_RELATIVE_CANDIDATES = (
    Path("data_structures") / "stacks" / "stack.py",
    Path("stacks") / "stack.py",
    Path("stack.py"),
)


def _candidate_paths():
    """Yield every plausible location of the source file, best first."""
    here = Path(__file__).resolve()
    seen = set()

    def _add(path: Path):
        try:
            key = path.resolve()
        except OSError:  # pragma: no cover - defensive
            key = path
        if key not in seen:
            seen.add(key)
            return path
        return None

    first = _add(_WINDOWS_SOURCE_PATH)
    if first is not None:
        yield first

    roots = [here.parent, *here.parents, Path.cwd()]
    for root in roots:
        for rel in _RELATIVE_CANDIDATES:
            candidate = _add(Path(root) / rel)
            if candidate is not None:
                yield candidate


def _find_source_path() -> Path:
    """Return the path of stack.py, raising FileNotFoundError if absent."""
    for candidate in _candidate_paths():
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - defensive
            continue

    # Last resort: a shallow search from a couple of plausible roots.
    for root in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            for found in sorted(root.glob("**/stacks/stack.py")):
                return found
            for found in sorted(root.glob("**/stack.py")):
                return found
        except OSError:  # pragma: no cover - defensive
            continue

    raise FileNotFoundError("stack.py not found")


def _load_module():
    """Import the module under test, by package name if possible."""
    source_path = _find_source_path()

    # Make the directories above the package importable and try a normal import
    # (this keeps the module's real name, which the doctests refer to).
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    search_roots.extend(Path(__file__).resolve().parents)
    if source_path.parent.name == "stacks" and source_path.parent.parent.name == "data_structures":
        search_roots.insert(0, source_path.parent.parent.parent)
    for root in search_roots:
        try:
            root_str = str(root)
        except OSError:  # pragma: no cover - defensive
            continue
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    try:
        import data_structures.stacks.stack as imported  # noqa: F401

        if Path(imported.__file__).resolve() == source_path.resolve():
            return imported
    except Exception:  # pragma: no cover - fall back to file loading
        pass

    # Fall back to loading the file directly by path.
    spec = importlib.util.spec_from_file_location("stack_under_test", source_path)
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
# Exception classes
# ---------------------------------------------------------------------------
def test_exception_classes_are_base_exceptions():
    """StackOverflowError and StackUnderflowError derive from BaseException."""
    assert issubclass(StackOverflowError, BaseException)
    assert issubclass(StackUnderflowError, BaseException)
    # They are intentionally NOT Exception subclasses (mirrors the source).
    assert not issubclass(StackOverflowError, Exception)
    assert not issubclass(StackUnderflowError, Exception)


def test_exception_classes_can_be_raised_and_caught():
    """Both custom exceptions can be raised and caught by their own type."""
    with pytest.raises(StackOverflowError):
        raise StackOverflowError
    with pytest.raises(StackUnderflowError):
        raise StackUnderflowError


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
def test_init_default_limit_and_empty_state():
    """A new stack has the default limit of 10 and an empty internal list."""
    s = Stack()
    assert s.limit == 10
    assert s.stack == []
    assert s.size() == 0


@pytest.mark.parametrize("limit", [0, 1, 5, 100])
def test_init_custom_limit(limit):
    """A custom limit is stored verbatim and the stack starts empty."""
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
    """A stack with at least one element is truthy."""
    s = Stack()
    s.push(1)
    assert bool(s) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
def test_str_empty_stack():
    """str() of an empty stack is the empty list representation."""
    assert str(Stack()) == "[]"


def test_str_with_elements():
    """str() mirrors the internal list representation."""
    s = Stack(3)
    s.push(1)
    s.push("two")
    assert str(s) == str([1, "two"])


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def test_push_adds_element_to_top():
    """push() appends the element to the top of the stack."""
    s = Stack(3)
    s.push(10)
    assert s.stack == [10]
    s.push(20)
    assert s.stack == [10, 20]
    assert s.peek() == 20


def test_push_returns_none():
    """push() returns None."""
    s = Stack(2)
    assert s.push(1) is None


def test_push_raises_overflow_when_full():
    """push() raises StackOverflowError when the limit is reached."""
    s = Stack(1)
    s.push(10)
    with pytest.raises(StackOverflowError):
        s.push(20)
    # The failed push must not mutate the stack.
    assert s.stack == [10]


def test_push_raises_overflow_on_zero_limit():
    """A stack with limit 0 cannot accept any element."""
    s = Stack(0)
    with pytest.raises(StackOverflowError):
        s.push(1)
    assert s.size() == 0


@pytest.mark.parametrize("value", [0, -1, 3.14, "text", None, [1, 2], (1, 2)])
def test_push_accepts_arbitrary_types(value):
    """push() stores any object without type coercion."""
    s = Stack(1)
    s.push(value)
    assert s.peek() is value


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------
def test_pop_returns_last_pushed_element():
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
    """Popping more times than pushing raises StackUnderflowError."""
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
    """A freshly created stack is empty."""
    assert Stack().is_empty() is True


def test_is_empty_false_after_push_and_true_after_pop():
    """is_empty() tracks the stack contents."""
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


def test_is_full_true_when_limit_reached():
    """is_full() becomes True exactly when size equals limit."""
    s = Stack(1)
    s.push(10)
    assert s.is_full() is True


def test_is_full_true_for_zero_limit_stack():
    """A stack with limit 0 is full immediately (size == limit == 0)."""
    assert Stack(0).is_full() is True


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_is_full_boundary(limit):
    """is_full() is False at limit-1 elements and True at limit elements."""
    s = Stack(limit)
    for i in range(limit - 1):
        s.push(i)
    assert s.is_full() is False
    s.push(limit - 1)
    assert s.is_full() is True


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_size_matches_number_of_pushes(count):
    """size() equals the number of elements currently on the stack."""
    s = Stack(5)
    for i in range(count):
        s.push(i)
    assert s.size() == count


def test_size_decreases_after_pop():
    """size() decreases by one after each pop."""
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
    """The `in` operator finds an element that was pushed."""
    s = Stack(3)
    s.push(10)
    assert (10 in s) is True


def test_contains_absent_item():
    """The `in` operator returns False for an element not on the stack."""
    s = Stack(3)
    s.push(10)
    assert (20 in s) is False


def test_contains_on_empty_stack():
    """The `in` operator returns False on an empty stack."""
    assert (1 in Stack()) is False


@pytest.mark.parametrize(
    ("contents", "item", "expected"),
    [
        ([1, 2, 3], 2, True),
        ([1, 2, 3], 4, False),
        (["a", "b"], "a", True),
        (["a", "b"], "c", False),
        ([None], None, True),
        ([0], False, True),  # 0 == False in Python
    ],
)
def test_contains_parametrized(contents, item, expected):
    """`in` uses standard equality semantics over the stack contents."""
    s = Stack(len(contents) or 1)
    for value in contents:
        s.push(value)
    assert (item in s) is expected


# ---------------------------------------------------------------------------
# Integration / LIFO behaviour
# ---------------------------------------------------------------------------
def test_lifo_ordering():
    """Elements come off the stack in last-in-first-out order."""
    s = Stack(5)
    for value in [1, 2, 3, 4, 5]:
        s.push(value)
    popped = [s.pop() for _ in range(5)]
    assert popped == [5, 4, 3, 2, 1]
    assert s.is_empty() is True


def test_module_level_test_stack_function():
    """The module's own `test_stack` helper runs without raising."""
    stack_module.test_stack()


def test_doctests_pass():
    """All doctests embedded in the module execute successfully."""
    results = doctest.testmod(
        stack_module,
        verbose=False,
        optionflags=doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert results.failed == 0
