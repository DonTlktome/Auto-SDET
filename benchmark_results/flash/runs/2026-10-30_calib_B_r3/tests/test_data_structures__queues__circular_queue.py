"""Tests for data_structures/queues/circular_queue.py"""

import pytest

from data_structures.queues.circular_queue import CircularQueue


# --------------------------------------------------------------------------- #
# Construction / __init__
# --------------------------------------------------------------------------- #
def test_init_sets_defaults():
    """A new queue must have the requested capacity, empty storage and zero size."""
    cq = CircularQueue(5)
    assert cq.n == 5
    assert cq.array == [None] * 5
    assert cq.front == 0
    assert cq.rear == 0
    assert cq.size == 0


@pytest.mark.parametrize("capacity", [1, 2, 5, 100])
def test_init_various_capacities(capacity):
    """The backing list length must always equal the requested capacity."""
    cq = CircularQueue(capacity)
    assert len(cq.array) == capacity
    assert cq.n == capacity


def test_init_zero_capacity():
    """A zero-capacity queue is allowed to be constructed but is immediately full."""
    cq = CircularQueue(0)
    assert cq.array == []
    assert cq.is_empty() is True
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("A")


# --------------------------------------------------------------------------- #
# __len__
# --------------------------------------------------------------------------- #
def test_len_empty():
    """len() of a fresh queue is 0."""
    assert len(CircularQueue(5)) == 0


def test_len_after_enqueues_and_dequeues():
    """len() tracks the number of stored elements through mixed operations."""
    cq = CircularQueue(5)
    cq.enqueue("A")
    cq.enqueue("B")
    assert len(cq) == 2
    cq.dequeue()
    assert len(cq) == 1
    cq.dequeue()
    assert len(cq) == 0


def test_len_full_queue():
    """len() equals capacity when the queue is completely filled."""
    cq = CircularQueue(3)
    cq.enqueue(1).enqueue(2).enqueue(3)
    assert len(cq) == 3


# --------------------------------------------------------------------------- #
# is_empty
# --------------------------------------------------------------------------- #
def test_is_empty_true_on_new_queue():
    """A newly created queue reports itself as empty."""
    assert CircularQueue(5).is_empty() is True


def test_is_empty_false_after_enqueue():
    """Enqueueing a single element makes the queue non-empty."""
    cq = CircularQueue(5)
    cq.enqueue("A")
    assert cq.is_empty() is False


def test_is_empty_true_after_draining():
    """Draining the queue returns it to the empty state."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    cq.dequeue()
    assert cq.is_empty() is True


# --------------------------------------------------------------------------- #
# first
# --------------------------------------------------------------------------- #
def test_first_on_empty_returns_false():
    """first() returns False (not None) when the queue is empty."""
    assert CircularQueue(5).first() is False


def test_first_returns_front_element():
    """first() returns the element at the front without removing it."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    assert cq.first() == "A"
    assert len(cq) == 2  # unchanged


def test_first_after_dequeue():
    """first() reflects the new front element after a dequeue."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    assert cq.first() == "B"


def test_first_after_wraparound():
    """first() is correct after the front index wraps around the array."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()  # front -> 1
    cq.dequeue()  # front -> 2
    cq.enqueue("D")  # rear wraps to 0
    cq.enqueue("E")  # rear -> 1
    assert cq.first() == "C"
    assert cq.array == ["D", "E", "C"]


# --------------------------------------------------------------------------- #
# enqueue
# --------------------------------------------------------------------------- #
def test_enqueue_returns_self_for_chaining():
    """enqueue() returns the queue instance to allow fluent chaining."""
    cq = CircularQueue(5)
    assert cq.enqueue("A") is cq


def test_enqueue_stores_element_at_rear():
    """enqueue() writes the element at the rear index and advances rear."""
    cq = CircularQueue(5)
    cq.enqueue("A")
    assert cq.array == ["A", None, None, None, None]
    assert cq.rear == 1
    assert cq.size == 1


def test_enqueue_multiple_elements():
    """Sequential enqueues fill the array in order."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B").enqueue("C")
    assert cq.array == ["A", "B", "C", None, None]
    assert cq.size == 3
    assert cq.rear == 3


def test_enqueue_raises_when_full():
    """Enqueueing into a full queue raises 'QUEUE IS FULL'."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("C")
    # state must be unchanged after the failed enqueue
    assert cq.size == 2
    assert cq.array == ["A", "B"]


def test_enqueue_wraps_rear_index():
    """The rear index wraps modulo capacity once it reaches the end."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()
    cq.enqueue("D")
    assert cq.rear == 1
    assert cq.array == ["D", "B", "C"]


@pytest.mark.parametrize("value", [0, -1, "", None, 3.14, [1, 2], {"k": "v"}])
def test_enqueue_accepts_arbitrary_values(value):
    """enqueue() stores any Python object without type restrictions."""
    cq = CircularQueue(2)
    cq.enqueue(value)
    assert cq.first() == value
    assert len(cq) == 1


# --------------------------------------------------------------------------- #
# dequeue
# --------------------------------------------------------------------------- #
def test_dequeue_raises_on_empty():
    """Dequeueing from an empty queue raises 'UNDERFLOW'."""
    cq = CircularQueue(5)
    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


def test_dequeue_returns_fifo_order():
    """dequeue() returns elements in first-in-first-out order."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B").enqueue("C")
    assert cq.dequeue() == "A"
    assert cq.dequeue() == "B"
    assert cq.dequeue() == "C"


def test_dequeue_clears_slot_and_advances_front():
    """dequeue() nulls the vacated slot and advances the front index."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    assert cq.array == [None, "B", None, None, None]
    assert cq.front == 1
    assert cq.size == 1


def test_dequeue_raises_after_draining():
    """A drained queue raises 'UNDERFLOW' on further dequeue calls."""
    cq = CircularQueue(2)
    cq.enqueue("A")
    assert cq.dequeue() == "A"
    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


def test_dequeue_wraps_front_index():
    """The front index wraps modulo capacity once it reaches the end."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()
    cq.dequeue()
    cq.dequeue()
    assert cq.front == 0
    assert cq.size == 0
    assert cq.array == [None, None, None]


# --------------------------------------------------------------------------- #
# Integration / circular behaviour
# --------------------------------------------------------------------------- #
def test_fifo_ordering_with_wraparound():
    """FIFO ordering is preserved across many wrap-arounds of both indices."""
    cq = CircularQueue(5)
    produced = list(range(10))
    consumed = []

    for i, value in enumerate(produced):
        cq.enqueue(value)
        if i % 2 == 0:
            consumed.append(cq.dequeue())

    while not cq.is_empty():
        consumed.append(cq.dequeue())

    assert consumed == produced


def test_alternating_enqueue_dequeue_keeps_size_consistent():
    """size, len() and is_empty() stay consistent under interleaved operations."""
    cq = CircularQueue(4)
    for i in range(20):
        cq.enqueue(i)
        assert len(cq) == 1
        assert cq.is_empty() is False
        assert cq.dequeue() == i
        assert len(cq) == 0
        assert cq.is_empty() is True


def test_full_then_empty_cycle_reuses_capacity():
    """A queue can be refilled to capacity after being fully drained."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("D")
    for expected in ("A", "B", "C"):
        assert cq.dequeue() == expected
    cq.enqueue("X").enqueue("Y").enqueue("Z")
    assert cq.array == ["X", "Y", "Z"]
    assert cq.first() == "X"
    assert len(cq) == 3


def test_capacity_one_queue():
    """A capacity-1 queue behaves correctly for enqueue/dequeue cycles."""
    cq = CircularQueue(1)
    cq.enqueue("A")
    assert cq.first() == "A"
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("B")
    assert cq.dequeue() == "A"
    assert cq.is_empty() is True
    cq.enqueue("C")
    assert cq.dequeue() == "C"
