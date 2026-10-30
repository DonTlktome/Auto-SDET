"""Tests for data_structures/queues/queue_by_list.py"""

import pytest

from data_structures.queues.queue_by_list import QueueByList


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_empty():
    """A queue created without arguments must be empty."""
    queue = QueueByList()
    assert len(queue) == 0
    assert queue.entries == []


@pytest.mark.parametrize(
    "iterable, expected",
    [
        ([10, 20, 30], [10, 20, 30]),
        ((1, 2, 3), [1, 2, 3]),
        ("abc", ["a", "b", "c"]),
        (range(3), [0, 1, 2]),
        ([], []),
    ],
)
def test_init_with_iterable(iterable, expected):
    """The queue must be populated from any iterable, preserving order."""
    queue = QueueByList(iterable)
    assert queue.entries == expected
    assert len(queue) == len(expected)


def test_init_with_generator():
    """A generator must be fully consumed into the internal list."""
    queue = QueueByList(i**2 for i in range(1, 4))
    assert queue.entries == [1, 4, 9]


def test_init_does_not_alias_input_list():
    """Mutating the source list must not affect the queue's internal storage."""
    source = [1, 2, 3]
    queue = QueueByList(source)
    source.append(4)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

def test_len_empty():
    """An empty queue has length zero."""
    assert len(QueueByList()) == 0


def test_len_after_put_and_get():
    """Length must track puts and gets."""
    queue = QueueByList()
    for i in range(1, 11):
        queue.put(i)
    assert len(queue) == 10
    queue.get()
    queue.get()
    assert len(queue) == 8


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "items, expected",
    [
        ([], "Queue(())"),
        ([10], "Queue((10,))"),
        ([10, 20, 30], "Queue((10, 20, 30))"),
        (["a", "b"], "Queue(('a', 'b'))"),
    ],
)
def test_repr(items, expected):
    """repr must render the entries as a tuple inside 'Queue(...)'."""
    assert repr(QueueByList(items)) == expected


def test_str_matches_repr():
    """str() falls back to __repr__ and must produce the same text."""
    queue = QueueByList([1, 2])
    assert str(queue) == repr(queue) == "Queue((1, 2))"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

def test_put_single_item():
    """put appends an item to the back of the queue."""
    queue = QueueByList()
    queue.put(10)
    assert queue.entries == [10]
    assert len(queue) == 1


def test_put_multiple_items_preserves_order():
    """Multiple puts must preserve FIFO insertion order."""
    queue = QueueByList()
    queue.put(10)
    queue.put(20)
    queue.put(30)
    assert queue.entries == [10, 20, 30]
    assert repr(queue) == "Queue((10, 20, 30))"


def test_put_returns_none():
    """put must return None."""
    queue = QueueByList()
    assert queue.put(1) is None


def test_put_none_value():
    """put must accept None as a legitimate value."""
    queue = QueueByList()
    queue.put(None)
    assert len(queue) == 1
    assert queue.get() is None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_returns_front_item():
    """get removes and returns the front (oldest) item."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    assert queue.entries == [20, 30]


def test_get_fifo_order():
    """Repeated gets must return items in FIFO order."""
    queue = QueueByList([1, 2, 3])
    assert [queue.get(), queue.get(), queue.get()] == [1, 2, 3]
    assert len(queue) == 0


def test_get_after_put():
    """Items put after construction are dequeued after existing ones."""
    queue = QueueByList((10, 20, 30))
    queue.put(40)
    assert queue.get() == 10
    assert queue.get() == 20
    assert queue.get() == 30
    assert len(queue) == 1
    assert queue.get() == 40


def test_get_empty_raises_index_error():
    """get on an empty queue must raise IndexError with the expected message."""
    queue = QueueByList()
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.get()


def test_get_until_empty_then_raises():
    """Draining the queue then calling get again must raise IndexError."""
    queue = QueueByList([1])
    assert queue.get() == 1
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.get()


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rotation, expected",
    [
        (0, [10, 20, 30, 40]),
        (1, [20, 30, 40, 10]),
        (2, [30, 40, 10, 20]),
        (4, [10, 20, 30, 40]),
        (5, [20, 30, 40, 10]),
    ],
)
def test_rotate(rotation, expected):
    """rotate(n) moves the front item to the back n times."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(rotation)
    assert queue.entries == expected


def test_rotate_sequential_calls():
    """Successive rotate calls accumulate on the current state."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(1)
    assert repr(queue) == "Queue((20, 30, 40, 10))"
    queue.rotate(2)
    assert repr(queue) == "Queue((40, 10, 20, 30))"


def test_rotate_empty_queue():
    """Rotating an empty queue raises IndexError from the underlying pop."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.rotate(3)
    assert len(queue) == 0


def test_rotate_single_element():
    """Rotating a one-element queue leaves it unchanged."""
    queue = QueueByList([42])
    queue.rotate(5)
    assert queue.entries == [42]


def test_rotate_negative_is_noop():
    """A negative rotation count performs no rotations."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(-1)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------

def test_get_front_returns_front_without_removing():
    """get_front peeks at the front item and leaves the queue unchanged."""
    queue = QueueByList((10, 20, 30))
    assert queue.get_front() == 10
    assert repr(queue) == "Queue((10, 20, 30))"


def test_get_front_after_get():
    """After a get, get_front reflects the new front item."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    assert queue.get_front() == 20


def test_get_front_empty_raises_index_error():
    """get_front on an empty queue raises IndexError."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.get_front()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_mixed_operations():
    """A mixed sequence of operations behaves like a FIFO queue."""
    queue = QueueByList([1, 2, 3])
    queue.put(4)
    assert queue.get() == 1
    queue.rotate(1)
    assert queue.entries == [3, 4, 2]
    assert queue.get_front() == 3
    assert queue.get() == 3
    assert len(queue) == 2
    assert queue.get() == 4
    assert queue.get() == 2
    assert len(queue) == 0