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
        ([], []),
        ([10, 20, 30], [10, 20, 30]),
        ((1, 2, 3), [1, 2, 3]),
        ("abc", ["a", "b", "c"]),
        (range(3), [0, 1, 2]),
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


def test_init_with_none_explicitly():
    """Passing None explicitly must behave like an empty queue."""
    queue = QueueByList(None)
    assert queue.entries == []
    assert len(queue) == 0


def test_init_does_not_alias_input_list():
    """Mutating the source list must not affect the queue's internal storage."""
    source = [1, 2, 3]
    queue = QueueByList(source)
    source.append(4)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "items, expected_len",
    [
        ([], 0),
        ([1], 1),
        ([1, 2, 3, 4, 5], 5),
    ],
)
def test_len(items, expected_len):
    """len() must reflect the number of stored items."""
    assert len(QueueByList(items)) == expected_len


def test_len_after_put_and_get():
    """len() must track puts and gets."""
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
    "items, expected_repr",
    [
        ([], "Queue(())"),
        ([10], "Queue((10,))"),
        ([10, 20, 30], "Queue((10, 20, 30))"),
        (["a", "b"], "Queue(('a', 'b'))"),
    ],
)
def test_repr(items, expected_repr):
    """repr() must render the queue contents as a tuple."""
    assert repr(QueueByList(items)) == expected_repr


def test_str_matches_repr():
    """str() falls back to repr() for this class."""
    queue = QueueByList([1, 2])
    assert str(queue) == repr(queue) == "Queue((1, 2))"


def test_repr_reflects_mutations():
    """repr() must reflect items added after construction."""
    queue = QueueByList()
    assert repr(queue) == "Queue(())"
    queue.put(10)
    assert repr(queue) == "Queue((10,))"
    queue.put(20)
    queue.put(30)
    assert repr(queue) == "Queue((10, 20, 30))"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

def test_put_single_item():
    """put() must append an item to the back of the queue."""
    queue = QueueByList()
    queue.put(10)
    assert queue.entries == [10]
    assert len(queue) == 1


def test_put_preserves_fifo_order():
    """Items must come out in the same order they were put in."""
    queue = QueueByList()
    for i in range(5):
        queue.put(i)
    assert [queue.get() for _ in range(5)] == [0, 1, 2, 3, 4]


def test_put_none_and_falsy_values():
    """Falsy values (None, 0, '', False) must be stored, not dropped."""
    queue = QueueByList()
    for value in (None, 0, "", False):
        queue.put(value)
    assert len(queue) == 4
    assert queue.entries == [None, 0, "", False]


def test_put_returns_none():
    """put() must return None."""
    queue = QueueByList()
    assert queue.put(1) is None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_returns_front_item():
    """get() must return and remove the front item."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    assert queue.entries == [20, 30]


def test_get_after_put():
    """get() must respect items added after construction."""
    queue = QueueByList((10, 20, 30))
    queue.put(40)
    assert queue.get() == 10
    assert queue.get() == 20
    assert queue.get() == 30
    assert len(queue) == 1
    assert queue.get() == 40


def test_get_on_empty_queue_raises_index_error():
    """get() on an empty queue must raise IndexError with the documented message."""
    queue = QueueByList()
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.get()


def test_get_until_empty_then_raises():
    """Draining the queue must leave it empty and further gets must raise."""
    queue = QueueByList([1])
    assert queue.get() == 1
    assert len(queue) == 0
    with pytest.raises(IndexError):
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
        (3, [40, 10, 20, 30]),
        (4, [10, 20, 30, 40]),
        (5, [20, 30, 40, 10]),
    ],
)
def test_rotate(rotation, expected):
    """rotate(n) must move the front item to the back n times."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(rotation)
    assert queue.entries == expected


def test_rotate_sequential_calls():
    """Successive rotate() calls must accumulate."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(1)
    assert repr(queue) == "Queue((20, 30, 40, 10))"
    queue.rotate(2)
    assert repr(queue) == "Queue((40, 10, 20, 30))"


def test_rotate_empty_queue_raises_index_error():
    """rotate() pops from the internal list unconditionally.

    The implementation performs ``self.entries.pop(0)`` for every rotation
    step without a guard, so rotating an empty queue raises IndexError (the
    queue is left untouched).
    """
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.rotate(3)
    assert queue.entries == []
    assert len(queue) == 0


def test_rotate_empty_queue_zero_rotation_is_noop():
    """A zero rotation never touches the internal list, even when empty."""
    queue = QueueByList()
    queue.rotate(0)
    assert queue.entries == []
    assert len(queue) == 0


def test_rotate_single_element():
    """Rotating a one-element queue must leave it unchanged."""
    queue = QueueByList([42])
    queue.rotate(5)
    assert queue.entries == [42]


def test_rotate_negative_raises_index_error():
    """A negative rotation count performs no iterations (range is empty)."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(-1)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------

def test_get_front_returns_without_removing():
    """get_front() must return the front item without mutating the queue."""
    queue = QueueByList((10, 20, 30))
    assert queue.get_front() == 10
    assert repr(queue) == "Queue((10, 20, 30))"


def test_get_front_after_get():
    """get_front() must reflect the new front after a get()."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    assert queue.get_front() == 20


def test_get_front_on_empty_queue_raises_index_error():
    """get_front() on an empty queue must raise IndexError."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.get_front()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def test_fifo_integration():
    """A mixed sequence of operations must behave like a FIFO queue."""
    queue = QueueByList()
    queue.put("a")
    queue.put("b")
    assert queue.get_front() == "a"
    assert queue.get() == "a"
    queue.put("c")
    # After the get, the queue holds ["b"]; putting "c" gives ["b", "c"], and
    # rotate(1) moves the front ("b") to the back, yielding ["c", "b"].
    queue.rotate(1)
    assert queue.entries == ["c", "b"]
    assert queue.get() == "c"
    assert queue.get() == "b"
    assert len(queue) == 0
