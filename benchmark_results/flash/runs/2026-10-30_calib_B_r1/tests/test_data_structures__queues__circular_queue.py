"""Tests for data_structures/queues/circular_queue.py"""

import pytest

from data_structures.queues.circular_queue import CircularQueue


# ---------------------------------------------------------------------------
# Construction / __init__
# ---------------------------------------------------------------------------


def test_init_sets_expected_initial_state():
    """A new queue must have the given capacity, an all-None buffer and zero size."""
    cq = CircularQueue(5)
    assert cq.n == 5
    assert cq.array == [None, None, None, None, None]
    assert cq.front == 0
    assert cq.rear == 0
    assert cq.size == 0


@pytest.mark.parametrize("capacity", [1, 2, 10, 100])
def test_init_various_capacities(capacity):
    """The internal buffer length must always equal the requested capacity."""
    cq = CircularQueue(capacity)
    assert len(cq.array) == capacity
    assert cq.n == capacity


def test_init_zero_capacity_creates_empty_buffer():
    """A capacity of zero yields an empty buffer (boundary case)."""
    cq = CircularQueue(0)
    assert cq.array == []
    assert cq.size == 0
    assert cq.is_empty() is True


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_empty_queue_is_zero():
    """len() of a fresh queue is 0."""
    assert len(CircularQueue(5)) == 0


def test_len_tracks_enqueue_and_dequeue():
    """len() must reflect the number of stored elements after each operation."""
    cq = CircularQueue(3)
    cq.enqueue("A")
    assert len(cq) == 1
    cq.enqueue("B")
    assert len(cq) == 2
    cq.dequeue()
    assert len(cq) == 1
    cq.dequeue()
    assert len(cq) == 0


def test_len_after_filling_to_capacity():
    """len() equals capacity once the queue is completely full."""
    cq = CircularQueue(3)
    for item in ("A", "B", "C"):
        cq.enqueue(item)
    assert len(cq) == 3


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_is_empty_true_for_new_queue():
    """A newly created queue is empty."""
    assert CircularQueue(5).is_empty() is True


def test_is_empty_false_after_enqueue():
    """Enqueuing one element makes the queue non-empty."""
    cq = CircularQueue(5)
    cq.enqueue("A")
    assert cq.is_empty() is False


def test_is_empty_true_again_after_draining():
    """Draining the queue returns it to the empty state."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    cq.dequeue()
    assert cq.is_empty() is True


# ---------------------------------------------------------------------------
# first
# ---------------------------------------------------------------------------


def test_first_on_empty_queue_returns_false():
    """first() returns False (sentinel) when the queue is empty."""
    assert CircularQueue(5).first() is False


def test_first_returns_front_element():
    """first() returns the element at the front without removing it."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    assert cq.first() == "A"
    assert len(cq) == 2  # unchanged


def test_first_after_dequeue_returns_next_element():
    """After a dequeue, first() reports the new front element."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    assert cq.first() == "B"


def test_first_returns_false_again_when_emptied():
    """first() falls back to False once all elements are removed."""
    cq = CircularQueue(2)
    cq.enqueue("A")
    cq.dequeue()
    assert cq.first() is False


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_returns_self_for_chaining():
    """enqueue() returns the queue instance to allow method chaining."""
    cq = CircularQueue(5)
    assert cq.enqueue("A") is cq


def test_enqueue_stores_element_at_rear():
    """A single enqueue writes to index 0 and advances rear."""
    cq = CircularQueue(5)
    cq.enqueue("A")
    assert cq.array == ["A", None, None, None, None]
    assert cq.rear == 1
    assert cq.size == 1


def test_enqueue_multiple_elements_in_order():
    """Successive enqueues fill the buffer left to right."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B").enqueue("C")
    assert cq.array == ["A", "B", "C", None, None]
    assert cq.rear == 3
    assert cq.size == 3


def test_enqueue_raises_when_full():
    """Enqueuing into a full queue raises an exception with 'QUEUE IS FULL'."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("C")


def test_enqueue_raises_on_zero_capacity_queue():
    """A zero-capacity queue is always full, so enqueue must raise."""
    cq = CircularQueue(0)
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("A")


def test_enqueue_wraps_around_after_dequeue():
    """rear wraps modulo n so freed slots at the start are reused."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()  # frees index 0
    cq.enqueue("D")  # should land at index 0
    assert cq.array == ["D", "B", "C"]
    assert cq.rear == 1
    assert cq.front == 1
    assert cq.size == 3


@pytest.mark.parametrize("value", [None, 0, "", False, [], {"k": "v"}])
def test_enqueue_accepts_arbitrary_values(value):
    """enqueue() stores any Python object without type restrictions."""
    cq = CircularQueue(2)
    cq.enqueue(value)
    assert cq.size == 1
    assert cq.first() == value


# ---------------------------------------------------------------------------
# dequeue
# ---------------------------------------------------------------------------


def test_dequeue_raises_on_empty_queue():
    """Dequeuing an empty queue raises an exception with 'UNDERFLOW'."""
    cq = CircularQueue(5)
    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


def test_dequeue_returns_front_and_clears_slot():
    """dequeue() returns the front element, clears its slot and advances front."""
    cq = CircularQueue(5)
    cq.enqueue("A").enqueue("B")
    assert cq.dequeue() == "A"
    assert cq.array == [None, "B", None, None, None]
    assert cq.front == 1
    assert cq.size == 1


def test_dequeue_fifo_order():
    """Elements come out in the same order they were inserted."""
    cq = CircularQueue(4)
    for item in ("A", "B", "C", "D"):
        cq.enqueue(item)
    assert [cq.dequeue() for _ in range(4)] == ["A", "B", "C", "D"]


def test_dequeue_raises_after_draining():
    """Once drained, further dequeues raise UNDERFLOW."""
    cq = CircularQueue(2)
    cq.enqueue("A")
    assert cq.dequeue() == "A"
    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


def test_dequeue_wraps_front_index():
    """front wraps modulo n when it passes the end of the buffer."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    assert cq.dequeue() == "A"
    assert cq.dequeue() == "B"
    assert cq.front == 0  # wrapped back to 0
    assert cq.rear == 0
    assert cq.size == 0


# ---------------------------------------------------------------------------
# Integration / circular behaviour
# ---------------------------------------------------------------------------


def test_interleaved_operations_keep_fifo_order():
    """A long interleaved sequence must preserve FIFO semantics."""
    cq = CircularQueue(3)
    cq.enqueue(1).enqueue(2)
    assert cq.dequeue() == 1
    cq.enqueue(3).enqueue(4)
    assert cq.dequeue() == 2
    assert cq.dequeue() == 3
    cq.enqueue(5)
    assert cq.dequeue() == 4
    assert cq.dequeue() == 5
    assert cq.is_empty() is True


def test_queue_can_be_reused_after_full_cycle():
    """After a full enqueue/dequeue cycle the queue is reusable to capacity."""
    cq = CircularQueue(2)
    for _ in range(3):
        cq.enqueue("X").enqueue("Y")
        assert cq.dequeue() == "X"
        assert cq.dequeue() == "Y"
        assert cq.is_empty() is True


def test_capacity_one_queue_behaviour():
    """A capacity-1 queue supports exactly one element at a time."""
    cq = CircularQueue(1)
    cq.enqueue("A")
    assert cq.first() == "A"
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("B")
    assert cq.dequeue() == "A"
    assert cq.is_empty() is True
    cq.enqueue("C")
    assert cq.dequeue() == "C"