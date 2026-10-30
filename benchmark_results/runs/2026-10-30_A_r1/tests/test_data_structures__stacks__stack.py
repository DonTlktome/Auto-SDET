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


# ---------------------------------------------------------------------------
# __bool__
# ---------------------------------------------------------------------------

class TestBool:
    """Tests for Stack.__bool__."""

    def test_empty_stack_is_falsy(self):
        """An empty stack should evaluate to False."""
        assert bool(Stack()) is False

    def test_non_empty_stack_is_truthy(self):
        """A stack with at least one element should evaluate to True."""
        stack = Stack()
        stack.push(1)
        assert bool(stack) is True


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------

class TestStr:
    """Tests for Stack.__str__."""

    def test_empty_stack_str(self):
        """An empty stack should stringify to '[]'."""
        assert str(Stack()) == "[]"

    def test_non_empty_stack_str(self):
        """A non-empty stack should stringify to its list representation."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert str(stack) == "[1, 2]"


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

class TestPush:
    """Tests for Stack.push."""

    def test_push_single_element(self):
        """Pushing one element should increase size and place it on top."""
        stack = Stack()
        stack.push(10)
        assert stack.size() == 1
        assert stack.peek() == 10

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
        """Pushing multiple elements should preserve LIFO order."""
        stack = Stack(limit=len(values))
        for value in values:
            stack.push(value)
        assert stack.size() == len(values)
        assert stack.peek() == values[-1]

    def test_push_at_limit_raises_overflow(self):
        """Pushing beyond the limit should raise StackOverflowError."""
        stack = Stack(limit=2)
        stack.push(1)
        stack.push(2)
        with pytest.raises(StackOverflowError):
            stack.push(3)

    def test_push_at_limit_does_not_modify_stack(self):
        """A failed push should not modify the stack contents."""
        stack = Stack(limit=2)
        stack.push(1)
        stack.push(2)
        with pytest.raises(StackOverflowError):
            stack.push(3)
        assert stack.size() == 2
        assert stack.peek() == 2


# ---------------------------------------------------------------------------
# pop
# ---------------------------------------------------------------------------

class TestPop:
    """Tests for Stack.pop."""

    def test_pop_returns_top_element(self):
        """Pop should return the most recently pushed element."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        assert stack.pop() == 2

    def test_pop_removes_element(self):
        """Pop should remove the top element, reducing size."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        assert stack.size() == 1
        assert stack.peek() == 1

    def test_pop_lifo_order(self):
        """Repeated pops should return elements in LIFO order."""
        stack = Stack()
        for value in [1, 2, 3]:
            stack.push(value)
        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1

    def test_pop_empty_raises_underflow(self):
        """Popping from an empty stack should raise StackUnderflowError."""
        with pytest.raises(StackUnderflowError):
            Stack().pop()

    def test_pop_until_empty_then_underflow(self):
        """After draining the stack, further pops should raise underflow."""
        stack = Stack()
        stack.push(1)
        assert stack.pop() == 1
        with pytest.raises(StackUnderflowError):
            stack.pop()


# ---------------------------------------------------------------------------
# peek
# ---------------------------------------------------------------------------

class TestPeek:
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


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------

class TestIsEmpty:
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
        """A stack drained by pops should report empty again."""
        stack = Stack()
        stack.push(1)
        stack.pop()
        assert stack.is_empty() is True


# ---------------------------------------------------------------------------
# is_full
# ---------------------------------------------------------------------------

class TestIsFull:
    """Tests for Stack.is_full."""

    def test_not_full_by_default(self):
        """A new stack with default limit should not be full."""
        assert Stack().is_full() is False

    def test_full_at_limit(self):
        """A stack filled to its limit should report full."""
        stack = Stack(limit=2)
        stack.push(1)
        stack.push(2)
        assert stack.is_full() is True

    def test_not_full_below_limit(self):
        """A stack below its limit should not report full."""
        stack = Stack(limit=2)
        stack.push(1)
        assert stack.is_full() is False

    def test_not_full_after_pop(self):
        """A stack that was full but popped should no longer be full."""
        stack = Stack(limit=1)
        stack.push(1)
        assert stack.is_full() is True
        stack.pop()
        assert stack.is_full() is False


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------

class TestSize:
    """Tests for Stack.size."""

    def test_initial_size_zero(self):
        """A new stack should have size 0."""
        assert Stack().size() == 0

    @pytest.mark.parametrize("count", [1, 2, 5, 10])
    def test_size_after_pushes(self, count):
        """Size should equal the number of pushed elements."""
        stack = Stack(limit=count)
        for i in range(count):
            stack.push(i)
        assert stack.size() == count

    def test_size_after_pops(self):
        """Size should decrease after pops."""
        stack = Stack()
        stack.push(1)
        stack.push(2)
        stack.pop()
        assert stack.size() == 1


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------

class TestContains:
    """Tests for Stack.__contains__."""

    def test_item_present(self):
        """The `in` operator should return True for a present item."""
        stack = Stack()
        stack.push(10)
        assert 10 in stack

    def test_item_absent(self):
        """The `in` operator should return False for an absent item."""
        stack = Stack()
        stack.push(10)
        assert 20 not in stack

    def test_empty_stack_contains_nothing(self):
        """An empty stack should not contain any item."""
        assert 10 not in Stack()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    """Tests for the custom exception classes."""

    def test_overflow_is_base_exception(self):
        """StackOverflowError should be a BaseException subclass."""
        assert issubclass(StackOverflowError, BaseException)

    def test_underflow_is_base_exception(self):
        """StackUnderflowError should be a BaseException subclass."""
        assert issubclass(StackUnderflowError, BaseException)

    def test_overflow_and_underflow_are_distinct(self):
        """The two exception types should be distinct from each other."""
        assert StackOverflowError is not StackUnderflowError


# ---------------------------------------------------------------------------
# Generic type support
# ---------------------------------------------------------------------------

class TestGenericTypes:
    """Tests that the stack works with various element types."""

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3],
            ["x", "y", "z"],
            [(1, 2), (3, 4)],
            [{"a": 1}, {"b": 2}],
        ],
    )
    def test_mixed_element_types(self, values):
        """The stack should handle arbitrary element types."""
        stack = Stack(limit=len(values))
        for value in values:
            stack.push(value)
        assert stack.pop() == values[-1]
        assert stack.peek() == values[-2]