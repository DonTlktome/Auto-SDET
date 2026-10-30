"""
Pytest test suite for the Stack data structure.

Covers all public methods of the Stack class, including:
- push / pop
- peek
- is_empty / is_full
- size
- __bool__ / __str__ / __contains__
- StackOverflowError and StackUnderflowError
"""

from __future__ import annotations

import pytest

from data_structures.stacks.stack import (
    Stack,
    StackOverflowError,
    StackUnderflowError,
)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for Stack.__init__."""

    def test_default_limit(self) -> None:
        """A stack created with no arguments should have limit=10."""
        s = Stack[int]()
        assert s.limit == 10
        assert s.size() == 0

    def test_custom_limit(self) -> None:
        """A stack created with a custom limit should store it."""
        s = Stack[int](limit=5)
        assert s.limit == 5

    def test_zero_limit(self) -> None:
        """A stack with limit=0 is valid and immediately full."""
        s = Stack[int](limit=0)
        assert s.limit == 0
        assert s.is_full() is True
        assert s.is_empty() is True


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

class TestPush:
    """Tests for Stack.push."""

    def test_push_single_element(self) -> None:
        """Push a single element and verify size/contents."""
        s = Stack[int]()
        s.push(42)
        assert s.size() == 1
        assert s.peek() == 42

    def test_push_multiple_elements(self) -> None:
        """Push several elements; LIFO order should be maintained."""
        s = Stack[int](limit=5)
        for val in [1, 2, 3]:
            s.push(val)
        assert s.size() == 3
        assert s.peek() == 3

    def test_push_up_to_limit(self) -> None:
        """Filling the stack exactly to its limit should succeed."""
        s = Stack[int](limit=3)
        s.push(1)
        s.push(2)
        s.push(3)
        assert s.is_full() is True
        assert s.size() == 3

    def test_push_beyond_limit_raises_overflow(self) -> None:
        """Pushing beyond the limit must raise StackOverflowError."""
        s = Stack[int](limit=1)
        s.push(10)
        with pytest.raises(StackOverflowError):
            s.push(20)

    def test_push_none(self) -> None:
        """None is a valid payload and should be stored."""
        s = Stack[int | None](limit=3)
        s.push(None)
        assert s.size() == 1
        assert s.peek() is None


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

class TestPop:
    """Tests for Stack.pop."""

    def test_pop_returns_last_pushed(self) -> None:
        """pop() should return the most recently pushed element."""
        s = Stack[int]()
        s.push(10)
        s.push(20)
        assert s.pop() == 20
        assert s.size() == 1

    def test_pop_empties_stack(self) -> None:
        """Repeated pops should eventually empty the stack."""
        s = Stack[int](limit=3)
        s.push(1)
        s.push(2)
        assert s.pop() == 2
        assert s.pop() == 1
        assert s.is_empty() is True

    def test_pop_from_empty_raises_underflow(self) -> None:
        """Popping an empty stack must raise StackUnderflowError."""
        s = Stack[int]()
        with pytest.raises(StackUnderflowError):
            s.pop()

    def test_pop_after_overflow_recovery(self) -> None:
        """After an overflow is caught, the stack should still work."""
        s = Stack[int](limit=2)
        s.push(1)
        s.push(2)
        with pytest.raises(StackOverflowError):
            s.push(3)
        assert s.pop() == 2
        assert s.pop() == 1


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------

class TestPeek:
    """Tests for Stack.peek."""

    def test_peek_returns_top_without_removing(self) -> None:
        """peek() should return the top element but not remove it."""
        s = Stack[int]()
        s.push(99)
        top = s.peek()
        assert top == 99
        assert s.size() == 1

    def test_peek_empty_raises_underflow(self) -> None:
        """Peeking an empty stack must raise StackUnderflowError."""
        s = Stack[int]()
        with pytest.raises(StackUnderflowError):
            s.peek()

    def test_peek_after_multiple_operations(self) -> None:
        """Peek should always reflect the current top."""
        s = Stack[str](limit=5)
        s.push("a")
        s.push("b")
        assert s.peek() == "b"
        s.pop()
        assert s.peek() == "a"


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------

class TestIsEmpty:
    """Tests for Stack.is_empty."""

    def test_new_stack_is_empty(self) -> None:
        """A freshly created stack should be empty."""
        assert Stack[int]().is_empty() is True

    def test_non_empty_stack(self) -> None:
        """After a push the stack should not be empty."""
        s = Stack[int]()
        s.push(1)
        assert s.is_empty() is False

    def test_after_pop_all(self) -> None:
        """After popping all elements the stack should be empty again."""
        s = Stack[int]()
        s.push(1)
        s.pop()
        assert s.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------

class TestIsFull:
    """Tests for Stack.is_full."""

    def test_default_not_full(self) -> None:
        """A new stack with default limit should not be full."""
        assert Stack[int]().is_full() is False

    def test_full_when_size_equals_limit(self) -> None:
        """is_full() returns True exactly when size == limit."""
        s = Stack[int](limit=2)
        s.push(1)
        assert s.is_full() is False
        s.push(2)
        assert s.is_full() is True

    def test_not_full_after_pop(self) -> None:
        """After popping from a full stack it should no longer be full."""
        s = Stack[int](limit=1)
        s.push(1)
        assert s.is_full() is True
        s.pop()
        assert s.is_full() is False


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    """Tests for Stack.size."""

    @pytest.mark.parametrize(
        "operations,expected_size",
        [
            ([], 0),
            (["push"], 1),
            (["push", "push"], 2),
            (["push", "push", "pop"], 1),
            (["push", "pop", "pop"], 0),  # second pop raises, caught
        ],
    )
    def test_size_after_operations(
        self, operations: list[str], expected_size: int
    ) -> None:
        """size() should reflect the number of elements after each operation."""
        s = Stack[int](limit=5)
        for op in operations:
            if op == "push":
                s.push(0)
            elif op == "pop":
                try:
                    s.pop()
                except StackUnderflowError:
                    pass
        assert s.size() == expected_size


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------

class TestBool:
    """Tests for Stack.__bool__."""

    def test_bool_empty(self) -> None:
        """bool(stack) should be False when empty."""
        assert bool(Stack[int]()) is False

    def test_bool_non_empty(self) -> None:
        """bool(stack) should be True when at least one element exists."""
        s = Stack[int]()
        s.push(1)
        assert bool(s) is True

    def test_bool_after_pop_all(self) -> None:
        """bool(stack) should become False after all elements are popped."""
        s = Stack[int]()
        s.push(1)
        s.pop()
        assert bool(s) is False


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestStr:
    """Tests for Stack.__str__."""

    def test_str_empty(self) -> None:
        """str() of an empty stack should be '[]'."""
        assert str(Stack[int]()) == "[]"

    def test_str_with_elements(self) -> None:
        """str() should match the string representation of the internal list."""
        s = Stack[int]()
        s.push(10)
        s.push(20)
        assert str(s) == "[10, 20]"

    def test_str_after_pop(self) -> None:
        """str() should update after a pop."""
        s = Stack[int]()
        s.push(1)
        s.push(2)
        s.pop()
        assert str(s) == "[1]"


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------

class TestContains:
    """Tests for Stack.__contains__."""

    def test_item_in_stack(self) -> None:
        """`in` operator should return True for an element that was pushed."""
        s = Stack[int]()
        s.push(42)
        assert 42 in s

    def test_item_not_in_stack(self) -> None:
        """`in` operator should return False for an element not pushed."""
        s = Stack[int]()
        s.push(1)
        assert 99 not in s

    def test_contains_empty_stack(self) -> None:
        """`in` on an empty stack should always return False."""
        assert 1 not in Stack[int]()

    def test_contains_duplicate(self) -> None:
        """`in` should work when duplicates are present."""
        s = Stack[int]()
        s.push(5)
        s.push(5)
        assert 5 in s


# ---------------------------------------------------------------------------
# Generic type parameter
# ---------------------------------------------------------------------------

class TestGeneric:
    """Tests verifying the Stack works with various types."""

    def test_str_type(self) -> None:
        """Stack should work with strings."""
        s = Stack[str](limit=3)
        s.push("hello")
        assert s.pop() == "hello"

    def test_float_type(self) -> None:
        """Stack should work with floats."""
        s = Stack[float]()
        s.push(3.14)
        assert s.peek() == 3.14

    def test_list_type(self) -> None:
        """Stack should work with list elements."""
        s = Stack[list[int]]()
        s.push([1, 2, 3])
        assert s.pop() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Integration / scenario tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end scenarios mimicking real-world usage."""

    def test_lifo_order(self) -> None:
        """Elements must be retrieved in Last-In-First-Out order."""
        s = Stack[int](limit=100)
        for i in range(50):
            s.push(i)
        for i in range(49, -1, -1):
            assert s.pop() == i

    def test_interleaved_push_pop(self) -> None:
        """Interleaved push/pop should maintain correct state."""
        s = Stack[int](limit=10)
        s.push(1)
        s.push(2)
        assert s.pop() == 2
        s.push(3)
        assert s.peek() == 3
        assert s.pop() == 3
        assert s.pop() == 1
        assert s.is_empty() is True

    def test_limit_one_stack(self) -> None:
        """A stack with limit=1 should only allow one element at a time."""
        s = Stack[int](limit=1)
        assert s.is_empty()
        s.push(100)
        assert s.is_full()
        with pytest.raises(StackOverflowError):
            s.push(200)
        assert s.pop() == 100
        assert s.is_empty()

    def test_large_limit(self) -> None:
        """Stack should handle a large limit without issues."""
        s = Stack[int](limit=10_000)
        for i in range(10_000):
            s.push(i)
        assert s.is_full()
        assert s.size() == 10_000
        assert s.peek() == 9_999