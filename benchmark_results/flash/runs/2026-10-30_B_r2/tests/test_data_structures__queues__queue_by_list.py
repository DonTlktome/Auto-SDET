"""Tests for data_structures/queues/queue_by_list.py"""

import pytest

from data_structures.queues.queue_by_list import QueueByList


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_default_is_empty():
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
    """The constructor must copy every element of the given iterable, in order."""
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


def test_init_with_none_is_empty():
    """Passing None explicitly must behave like passing no iterable."""
    queue = QueueByList(None)
    assert queue.entries == []
    assert len(queue) == 0


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iterable, expected_len",
    [
        ([], 0),
        ([1], 1),
        ([1, 2, 3, 4, 5], 5),
        ("abcdefghijklmnopqrstuvwxyz", 26),
    ],
)
def test_len(iterable, expected_len):
    """len() must report the number of stored items."""
    assert len(QueueByList(iterable)) == expected_len


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
# __repr__ / __str__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "iterable, expected",
    [
        ([], "Queue(())"),
        ([10], "Queue((10,))"),
        ([10, 20, 30], "Queue((10, 20, 30))"),
        (["a", "b"], "Queue(('a', 'b'))"),
    ],
)
def test_repr(iterable, expected):
    """repr() must render the entries as a tuple inside 'Queue(...)'."""
    assert repr(QueueByList(iterable)) == expected


def test_str_matches_repr():
    """str() falls back to __repr__ and must produce the same text."""
    queue = QueueByList([1, 2, 3])
    assert str(queue) == repr(queue) == "Queue((1, 2, 3))"


def test_repr_reflects_mutations():
    """repr() must reflect items added after construction."""
    queue = QueueByList()
    queue.put(10)
    assert repr(queue) == "Queue((10,))"
    queue.put(20)
    queue.put(30)
    assert repr(queue) == "Queue((10, 20, 30))"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


def test_put_appends_to_back():
    """put() must append items at the back of the queue (FIFO order)."""
    queue = QueueByList()
    queue.put(10)
    queue.put(20)
    assert queue.entries == [10, 20]
    assert len(queue) == 2


def test_put_returns_none():
    """put() must return None."""
    queue = QueueByList()
    assert queue.put(1) is None


@pytest.mark.parametrize("item", [None, 0, "", [], {"a": 1}, object()])
def test_put_arbitrary_values(item):
    """put() must accept any object, including falsy and mutable values."""
    queue = QueueByList()
    queue.put(item)
    assert len(queue) == 1
    assert queue.get_front() is item


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_returns_front_item():
    """get() must remove and return the front item (FIFO)."""
    queue = QueueByList((10, 20, 30))
    assert queue.get() == 10
    queue.put(40)
    assert queue.get() == 20
    assert queue.get() == 30
    assert len(queue) == 1
    assert queue.get() == 40


def test_get_on_empty_queue_raises_index_error():
    """get() on an empty queue must raise IndexError with the documented message."""
    queue = QueueByList()
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.get()


def test_get_empties_queue_then_raises():
    """After draining the queue, a further get() must raise IndexError."""
    queue = QueueByList([1])
    assert queue.get() == 1
    assert len(queue) == 0
    with pytest.raises(IndexError):
        queue.get()


def test_get_preserves_order_for_many_items():
    """Draining a queue must yield items in insertion order."""
    items = list(range(50))
    queue = QueueByList(items)
    assert [queue.get() for _ in items] == items


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
    assert queue.entries == [20, 30, 40, 10]
    queue.rotate(2)
    assert queue.entries == [40, 10, 20, 30]


def test_rotate_zero_is_noop():
    """rotate(0) must leave the queue unchanged."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(0)
    assert queue.entries == [1, 2, 3]


def test_rotate_empty_queue():
    """rotate() on an empty queue raises IndexError because it pops from an empty list."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.rotate(3)
    assert len(queue) == 0


def test_rotate_single_element():
    """rotate() on a one-element queue must leave it unchanged."""
    queue = QueueByList([42])
    queue.rotate(7)
    assert queue.entries == [42]


def test_rotate_negative_rotation_is_noop():
    """A negative rotation count results in no iterations, so nothing changes."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(-2)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------


def test_get_front_does_not_remove():
    """get_front() must return the front item without removing it."""
    queue = QueueByList((10, 20, 30))
    assert queue.get_front() == 10
    assert queue.entries == [10, 20, 30]
    assert queue.get() == 10
    assert queue.get_front() == 20


def test_get_front_on_empty_queue_raises_index_error():
    """get_front() on an empty queue must raise IndexError."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.get_front()


def test_get_front_after_put():
    """get_front() must reflect newly added items when the queue was empty."""
    queue = QueueByList()
    queue.put("first")
    queue.put("second")
    assert queue.get_front() == "first"


# ---------------------------------------------------------------------------
# Integration / doctest
# ---------------------------------------------------------------------------


def test_fifo_integration():
    """A mixed sequence of operations must behave like a FIFO queue."""
    queue = QueueByList()
    for i in range(5):
        queue.put(i)
    assert queue.get() == 0
    assert queue.get() == 1
    queue.put(5)
    queue.rotate(1)
    assert queue.entries == [3, 4, 5, 2]
    assert queue.get_front() == 3
    assert len(queue) == 4


def test_doctests_pass():
    """The module's doctests must all pass."""
    import doctest
    import sys

    module = sys.modules[QueueByList.__module__]
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0
