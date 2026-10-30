"""Tests for the QueueByList data structure."""

import pytest

from queue_by_list import QueueByList


class TestInit:
    """Tests for QueueByList.__init__."""

    def test_init_empty(self):
        """Creating a queue with no arguments yields an empty queue."""
        queue = QueueByList()
        assert len(queue) == 0
        assert queue.entries == []

    def test_init_with_list(self):
        """Creating a queue from a list preserves order and contents."""
        queue = QueueByList([10, 20, 30])
        assert queue.entries == [10, 20, 30]
        assert len(queue) == 3

    def test_init_with_generator(self):
        """Creating a queue from a generator consumes it into a list."""
        queue = QueueByList(i**2 for i in range(1, 4))
        assert queue.entries == [1, 4, 9]

    def test_init_with_empty_iterable(self):
        """Passing an empty iterable produces an empty queue."""
        queue = QueueByList([])
        assert len(queue) == 0

    def test_init_with_none(self):
        """Passing None explicitly behaves like no argument (empty queue)."""
        queue = QueueByList(None)
        assert len(queue) == 0


class TestLen:
    """Tests for QueueByList.__len__."""

    def test_len_empty(self):
        """Length of an empty queue is zero."""
        assert len(QueueByList()) == 0

    def test_len_after_puts(self):
        """Length reflects the number of items put into the queue."""
        queue = QueueByList()
        for i in range(1, 11):
            queue.put(i)
        assert len(queue) == 10

    def test_len_after_gets(self):
        """Length decreases as items are removed."""
        queue = QueueByList([1, 2, 3, 4])
        queue.get()
        queue.get()
        assert len(queue) == 2


class TestRepr:
    """Tests for QueueByList.__repr__."""

    def test_repr_empty(self):
        """Empty queue repr is 'Queue(())'."""
        assert repr(QueueByList()) == "Queue(())"

    def test_repr_single_item(self):
        """Single-item queue repr includes a trailing comma."""
        queue = QueueByList()
        queue.put(10)
        assert repr(queue) == "Queue((10,))"

    def test_repr_multiple_items(self):
        """Multiple items are shown in insertion order."""
        queue = QueueByList([10, 20, 30])
        assert repr(queue) == "Queue((10, 20, 30))"

    def test_str_matches_repr(self):
        """str() of the queue matches its repr."""
        queue = QueueByList([1, 2])
        assert str(queue) == repr(queue) == "Queue((1, 2))"


class TestPut:
    """Tests for QueueByList.put."""

    def test_put_single_item(self):
        """Putting an item appends it to the queue."""
        queue = QueueByList()
        queue.put(10)
        assert queue.entries == [10]

    @pytest.mark.parametrize(
        "items",
        [
            [1, 2, 3],
            ["a", "b", "c"],
            [None, 0, False],
        ],
    )
    def test_put_multiple_items(self, items):
        """Putting multiple items preserves FIFO order."""
        queue = QueueByList()
        for item in items:
            queue.put(item)
        assert queue.entries == items

    def test_put_returns_none(self):
        """put() returns None."""
        queue = QueueByList()
        assert queue.put(10) is None


class TestGet:
    """Tests for QueueByList.get."""

    def test_get_returns_front_item(self):
        """get() returns the oldest item (FIFO order)."""
        queue = QueueByList([10, 20, 30])
        assert queue.get() == 10
        assert queue.get() == 20
        assert queue.get() == 30

    def test_get_removes_item(self):
        """get() removes the returned item from the queue."""
        queue = QueueByList([10, 20, 30])
        queue.get()
        assert queue.entries == [20, 30]

    def test_get_empty_raises_index_error(self):
        """get() on an empty queue raises IndexError."""
        queue = QueueByList()
        with pytest.raises(IndexError, match="Queue is empty"):
            queue.get()

    def test_get_until_empty_then_raises(self):
        """Draining the queue then calling get() raises IndexError."""
        queue = QueueByList([1, 2])
        queue.get()
        queue.get()
        with pytest.raises(IndexError):
            queue.get()


class TestRotate:
    """Tests for QueueByList.rotate."""

    def test_rotate_by_one(self):
        """Rotating by one moves the front item to the back."""
        queue = QueueByList([10, 20, 30, 40])
        queue.rotate(1)
        assert queue.entries == [20, 30, 40, 10]

    def test_rotate_by_multiple(self):
        """Rotating by more than one shifts items accordingly."""
        queue = QueueByList([10, 20, 30, 40])
        queue.rotate(2)
        assert queue.entries == [30, 40, 10, 20]

    def test_rotate_by_zero(self):
        """Rotating by zero leaves the queue unchanged."""
        queue = QueueByList([10, 20, 30])
        queue.rotate(0)
        assert queue.entries == [10, 20, 30]

    def test_rotate_full_cycle(self):
        """Rotating by the queue length restores original order."""
        queue = QueueByList([10, 20, 30, 40])
        queue.rotate(4)
        assert queue.entries == [10, 20, 30, 40]

    def test_rotate_empty_queue(self):
        """Rotating an empty queue raises IndexError."""
        queue = QueueByList()
        with pytest.raises(IndexError):
            queue.rotate(3)
        assert queue.entries == []

    def test_rotate_returns_none(self):
        """rotate() returns None."""
        queue = QueueByList([1, 2, 3])
        assert queue.rotate(1) is None


class TestGetFront:
    """Tests for QueueByList.get_front."""

    def test_get_front_returns_first_item(self):
        """get_front() returns the front item without removing it."""
        queue = QueueByList([10, 20, 30])
        assert queue.get_front() == 10

    def test_get_front_does_not_remove(self):
        """get_front() leaves the queue unchanged."""
        queue = QueueByList([10, 20, 30])
        queue.get_front()
        assert queue.entries == [10, 20, 30]

    def test_get_front_after_get(self):
        """get_front() reflects the new front after a get()."""
        queue = QueueByList([10, 20, 30])
        queue.get()
        assert queue.get_front() == 20

    def test_get_front_empty_raises_index_error(self):
        """get_front() on an empty queue raises IndexError."""
        queue = QueueByList()
        with pytest.raises(IndexError):
            queue.get_front()
