import pytest

from data_structures.stacks.stack import (
    Stack,
    StackOverflowError,
    StackUnderflowError,
)


class TestStackInit:
    """Tests for Stack.__init__."""

    def test_default_limit(self):
        """A stack created without arguments should have a limit of 10."""
        stack = Stack()
        assert stack.limit == 10
        assert stack.size() == 0

    def test_custom_limit(self):
        """A stack created with a custom limit should store that limit."""
        stack = Stack(limit=3)
        assert stack.limit == 3
        assert stack.size() == 0

    def test_zero_limit(self):
        """A stack with limit 0 should be full immediately and reject pushes."""
        stack = Stack(limit=0)
        assert stack.is_full() is True
        with pytest.raises(StackOverflowError):
            stack.push(1)


class TestStackBool:
    """Tests for Stack.__bool__."""

    def test_empty_stack_is_falsy(self):
        """An empty stack should evaluate to False."""
        assert bool(Stack()) is False

    def test_non_empty_stack_is_truthy(self):
        """A stack with elements should evaluate to True."""
        stack = Stack()
        stack.push(1)
        assert bool(stack) is True


class TestStackStr:
    """Tests for Stack.__str__."""

    def test_empty_stack_str(self):
        """An empty stack should stringify to '[]'."""
        assert str(Stack()) == "[]"

    def test_non_empty_stack_str(self):
        """A stack with elements should stringify to its list representation."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert str(stack) == "[1, 2]"


class TestStackPush:
    """Tests for Stack.push."""

    def test_push_single_element(self):
        """Pushing one element should increase size and store the value."""
        stack = Stack()
        stack.push(42)
        assert stack.size() == 1
        assert stack.peek() == 42

    @pytest.mark.parametrize(
        "values",
        [
            [1],
            [1, 2, 3],
            ["a", "b", "c"],
            [None, 0, "", False],
        ],
    )
    def test_push_multiple_elements(self, values):
        """Pushing multiple elements should preserve insertion order."""
        stack = Stack(limit=len(values))
        for value in values:
            stack.push(value)
        assert stack.size() == len(values)
        assert str(stack) == str(values)

    def test_push_at_limit_raises_overflow(self):
        """Pushing beyond the limit should raise StackOverflowError."""
        stack = Stack(limit=2)
        stack.push(1)
        stack.push(2)
        with pytest.raises(StackOverflowError):
            stack.push(3)

    def test_push_at_limit_does_not_modify_stack(self):
        """A failed push should not modify the stack contents."""
        stack = Stack(limit=1)
        stack.push(10)
        with pytest.raises(StackOverflowError):
            stack.push(20)
        assert stack.size() == 1
        assert stack.peek() == 10


class TestStackPop:
    """Tests for Stack.pop."""

    def test_pop_returns_top_element(self):
        """Pop should return the most recently pushed element."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert stack.pop() == 2

    def test_pop_removes_element(self):
        """Pop should remove the element from the stack."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        assert stack.size() == 1
        assert stack.peek() == 1

    def test_pop_lifo_order(self):
        """Pop should follow Last In, First Out ordering."""
        stack = Stack()
        for i in range(5):
            stack.push(i)
        popped = [stack.pop() for _ in range(5)]
        assert popped == [4, 3, 2, 1, 0]

    def test_pop_empty_raises_underflow(self):
        """Popping from an empty stack should raise StackUnderflowError."""
        with pytest.raises(StackUnderflowError):
            Stack().pop()


class TestStackPeek:
    """Tests for Stack.peek."""

    def test_peek_returns_top_element(self):
        """Peek should return the top element without removing it."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert stack.peek() == 2

    def test_peek_does_not_modify_stack(self):
        """Peek should not change the stack size or contents."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.peek()
        assert stack.size() == 2
        assert stack.peek() == 2

    def test_peek_empty_raises_underflow(self):
        """Peeking an empty stack should raise StackUnderflowError."""
        with pytest.raises(StackUnderflowError):
            Stack().peek()


class TestStackIsEmpty:
    """Tests for Stack.is_empty."""

    def test_empty_stack(self):
        """A new stack should report empty."""
        assert Stack().is_empty() is True

    def test_non_empty_stack(self):
        """A stack with elements should not report empty."""
        stack = Stack()
        stack.push(1)
        assert stack.is_empty() is False

    def test_after_pop_all(self):
        """A stack should report empty after all elements are popped."""
        stack = Stack()
        stack.push(1)
        stack.pop()
        assert stack.is_empty() is True


class TestStackIsFull:
    """Tests for Stack.is_full."""

    def test_empty_stack_not_full(self):
        """A new stack should not report full."""
        assert Stack().is_full() is False

    def test_partially_filled_not_full(self):
        """A partially filled stack should not report full."""
        stack = Stack(limit=3)
        stack.push(1)
        assert stack.is_full() is False

    def test_at_limit_is_full(self):
        """A stack at its limit should report full."""
        stack = Stack(limit=2)
        stack.push(1)
        stack.push(2)
        assert stack.is_full() is True


class TestStackSize:
    """Tests for Stack.size."""

    def test_empty_stack_size(self):
        """A new stack should have size 0."""
        assert Stack().size() == 0

    @pytest.mark.parametrize("count", [1, 2, 5, 10])
    def test_size_after_pushes(self, count):
        """Size should reflect the number of pushed elements."""
        stack = Stack(limit=count)
        for i in range(count):
            stack.push(i)
        assert stack.size() == count

    def test_size_after_pop(self):
        """Size should decrease after a pop."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        assert stack.size() == 1


class TestStackContains:
    """Tests for Stack.__contains__."""

    def test_item_present(self):
        """The 'in' operator should return True for present items."""
        stack = Stack()
        stack.push(10)
        assert 10 in stack

    def test_item_absent(self):
        """The 'in' operator should return False for absent items."""
        stack = Stack()
        stack.push(10)
        assert 20 not in stack

    def test_empty_stack_contains_nothing(self):
        """An empty stack should not contain any item."""
        assert 5 not in Stack()


class TestStackGenericTypes:
    """Tests for Stack's generic type support."""

    def test_string_stack(self):
        """Stack should work with string elements."""
        stack = Stack(limit=3)
        stack.push("hello")
        stack.push("world")
        assert stack.pop() == "world"
        assert stack.peek() == "hello"

    def test_mixed_type_stack(self):
        """Stack should work with heterogeneous element types."""
        stack = Stack(limit=3)
        stack.push(1)
        stack.push("two")
        stack.push(3.0)
        assert stack.pop() == 3.0
        assert stack.pop() == "two"
        assert stack.pop() == 1


class TestStackExceptions:
    """Tests for the custom exception classes."""

    def test_overflow_error_is_base_exception(self):
        """StackOverflowError should inherit from BaseException."""
        assert issubclass(StackOverflowError, BaseException)

    def test_underflow_error_is_base_exception(self):
        """StackUnderflowError should inherit from BaseException."""
        assert issubclass(StackUnderflowError, BaseException)

    def test_overflow_and_underflow_are_distinct(self):
        """The two exception types should be distinct classes."""
        assert StackOverflowError is not StackUnderflowError