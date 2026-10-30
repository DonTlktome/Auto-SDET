"""Tests for the QueueByList data structure."""

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
        ([], []),
        (range(3), [0, 1, 2]),
    ],
)
def test_init_with_iterable(iterable, expected):
    """The constructor must copy all items from any iterable into a list."""
    queue = QueueByList(iterable)
    assert queue.entries == expected
    assert len(queue) == len(expected)


def test_init_with_generator():
    """The constructor must consume generators correctly."""
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
    """len() must update after put/get operations."""
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
    """repr() must render the queue contents as a tuple."""
    assert repr(QueueByList(items)) == expected


def test_str_matches_repr():
    """str() must fall back to repr() for this class."""
    queue = QueueByList([1, 2])
    assert str(queue) == repr(queue) == "Queue((1, 2))"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

def test_put_single_item():
    """put() must append an item to the back of the queue."""
    queue = QueueByList()
    queue.put(10)
    assert queue.entries == [10]
    assert len(queue) == 1


def test_put_multiple_items_preserves_order():
    """put() must preserve FIFO insertion order."""
    queue = QueueByList()
    queue.put(10)
    queue.put(20)
    queue.put(30)
    assert queue.entries == [10, 20, 30]
    assert repr(queue) == "Queue((10, 20, 30))"


def test_put_returns_none():
    """put() must return None."""
    queue = QueueByList()
    assert queue.put(1) is None


def test_put_none_value():
    """put() must accept None as a legitimate value."""
    queue = QueueByList()
    queue.put(None)
    assert len(queue) == 1
    assert queue.get() is None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_returns_front_item():
    """get() must return and remove the front item (FIFO)."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    assert queue.entries == [20, 30]


def test_get_sequence():
    """Repeated get() calls must return items in insertion order."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    queue.put(40)
    assert queue.get() == 20
    assert queue.get() == 30
    assert len(queue) == 1
    assert queue.get() == 40


def test_get_on_empty_queue_raises_index_error():
    """get() on an empty queue must raise IndexError with the expected message."""
    queue = QueueByList()
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.get()


def test_get_until_empty_then_raises():
    """get() must raise IndexError once the queue is drained."""
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
    """rotate() on an empty queue pops from the internal list and raises IndexError."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.rotate(3)
    assert queue.entries == []


def test_rotate_zero_on_empty_queue_is_noop():
    """A zero rotation never touches the internal list, even when empty."""
    queue = QueueByList()
    queue.rotate(0)
    assert queue.entries == []


def test_rotate_single_element():
    """rotate() on a single-element queue must leave it unchanged."""
    queue = QueueByList([42])
    queue.rotate(5)
    assert queue.entries == [42]


def test_rotate_negative_rotation_is_noop():
    """A negative rotation count must not modify the queue."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(-2)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------

def test_get_front_returns_item_without_removing():
    """get_front() must return the front item and leave the queue intact."""
    queue = QueueByList((10, 20, 30))
    assert queue.get_front() == 10
    assert queue.entries == [10, 20, 30]


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
    """A mixed sequence of operations must behave as a FIFO queue."""
    queue = QueueByList()
    for i in range(5):
        queue.put(i)
    assert [queue.get() for _ in range(5)] == [0, 1, 2, 3, 4]
    assert len(queue) == 0
    with pytest.raises(IndexError):
        queue.get()
