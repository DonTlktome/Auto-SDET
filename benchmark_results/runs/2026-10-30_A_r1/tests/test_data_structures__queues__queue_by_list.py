'''Tests for the QueueByList data structure.'''

import pytest

from queue_by_list import QueueByList


class TestQueueByListInit:
    '''Tests for QueueByList.__init__.'''

    def test_init_empty(self):
        '''Creating a queue with no arguments yields an empty queue.'''
        queue = QueueByList()
        assert len(queue) == 0
        assert queue.entries == []

    def test_init_with_list(self):
        '''Creating a queue from a list preserves order and contents.'''
        queue = QueueByList([10, 20, 30])
        assert queue.entries == [10, 20, 30]
        assert len(queue) == 3

    def test_init_with_generator(self):
        '''Creating a queue from a generator consumes it correctly.'''
        queue = QueueByList(i**2 for i in range(1, 4))
        assert queue.entries == [1, 4, 9]

    def test_init_with_none(self):
        '''Passing None explicitly behaves like no argument (empty queue).'''
        queue = QueueByList(None)
        assert queue.entries == []


class TestQueueByListLen:
    '''Tests for QueueByList.__len__.'''

    def test_len_empty(self):
        '''Length of an empty queue is zero.'''
        assert len(QueueByList()) == 0

    def test_len_after_put(self):
        '''Length reflects the number of items after puts.'''
        queue = QueueByList()
        for i in range(1, 11):
            queue.put(i)
        assert len(queue) == 10

    def test_len_after_get(self):
        '''Length decreases after items are removed.'''
        queue = QueueByList([1, 2, 3])
        queue.get()
        assert len(queue) == 2


class TestQueueByListRepr:
    '''Tests for QueueByList.__repr__.'''

    def test_repr_empty(self):
        '''Empty queue repr is 'Queue(())'.'''
        assert repr(QueueByList()) == 'Queue(())'

    def test_repr_single_item(self):
        '''Single-item queue repr includes a trailing comma.'''
        queue = QueueByList()
        queue.put(10)
        assert repr(queue) == 'Queue((10,))'

    def test_repr_multiple_items(self):
        '''Multiple-item queue repr shows all items in order.'''
        queue = QueueByList([10, 20, 30])
        assert repr(queue) == 'Queue((10, 20, 30))'

    def test_str_matches_repr(self):
        '''str() of the queue matches repr().'''
        queue = QueueByList([1, 2])
        assert str(queue) == repr(queue) == 'Queue((1, 2))'


class TestQueueByListPut:
    '''Tests for QueueByList.put.'''

    def test_put_single_item(self):
        '''Putting one item appends it to the queue.'''
        queue = QueueByList()
        queue.put(10)
        assert queue.entries == [10]

    def test_put_multiple_items(self):
        '''Putting multiple items preserves insertion order.'''
        queue = QueueByList()
        queue.put(10)
        queue.put(20)
        queue.put(30)
        assert queue.entries == [10, 20, 30]

    def test_put_returns_none(self):
        '''put() returns None.'''
        queue = QueueByList()
        assert queue.put(1) is None


class TestQueueByListGet:
    '''Tests for QueueByList.get.'''

    def test_get_returns_front_item(self):
        '''get() returns the oldest (front) item.'''
        queue = QueueByList([10, 20, 30])
        assert queue.get() == 10

    def test_get_removes_item(self):
        '''get() removes the returned item from the queue.'''
        queue = QueueByList([10, 20, 30])
        queue.get()
        assert queue.entries == [20, 30]

    def test_get_fifo_order(self):
        '''Repeated get() calls return items in FIFO order.'''
        queue = QueueByList([10, 20, 30])
        assert [queue.get() for _ in range(3)] == [10, 20, 30]

    def test_get_empty_raises_index_error(self):
        '''get() on an empty queue raises IndexError.'''
        queue = QueueByList()
        with pytest.raises(IndexError, match='Queue is empty'):
            queue.get()

    def test_get_after_draining_raises(self):
        '''get() raises IndexError after all items have been removed.'''
        queue = QueueByList([1])
        queue.get()
        with pytest.raises(IndexError):
            queue.get()


class TestQueueByListRotate:
    '''Tests for QueueByList.rotate.'''

    @pytest.mark.parametrize(
        ('initial', 'rotation', 'expected'),
        [
            ([10, 20, 30, 40], 1, [20, 30, 40, 10]),
            ([10, 20, 30, 40], 2, [30, 40, 10, 20]),
            ([10, 20, 30, 40], 3, [40, 10, 20, 30]),
            ([10, 20, 30, 40], 4, [10, 20, 30, 40]),
        ],
    )
    def test_rotate_positive(self, initial, rotation, expected):
        '''Rotating by a positive number shifts items left.'''
        queue = QueueByList(initial)
        queue.rotate(rotation)
        assert queue.entries == expected

    def test_rotate_zero(self):
        '''Rotating by zero leaves the queue unchanged.'''
        queue = QueueByList([10, 20, 30])
        queue.rotate(0)
        assert queue.entries == [10, 20, 30]

    def test_rotate_empty_queue(self):
        '''Rotating an empty queue with a non-zero rotation raises IndexError.'''
        queue = QueueByList()
        with pytest.raises(IndexError, match='pop from empty list'):
            queue.rotate(3)
        assert queue.entries == []

    def test_rotate_returns_none(self):
        '''rotate() returns None.'''
        queue = QueueByList([1, 2, 3])
        assert queue.rotate(1) is None


class TestQueueByListGetFront:
    '''Tests for QueueByList.get_front.'''

    def test_get_front_returns_first_item(self):
        '''get_front() returns the front item without removing it.'''
        queue = QueueByList([10, 20, 30])
        assert queue.get_front() == 10

    def test_get_front_does_not_remove(self):
        '''get_front() leaves the queue unchanged.'''
        queue = QueueByList([10, 20, 30])
        queue.get_front()
        assert queue.entries == [10, 20, 30]

    def test_get_front_empty_raises_index_error(self):
        '''get_front() on an empty queue raises IndexError.'''
        queue = QueueByList()
        with pytest.raises(IndexError):
            queue.get_front()

    def test_get_front_after_get(self):
        '''get_front() reflects the new front after a get().'''
        queue = QueueByList([10, 20, 30])
        queue.get()
        assert queue.get_front() == 20
