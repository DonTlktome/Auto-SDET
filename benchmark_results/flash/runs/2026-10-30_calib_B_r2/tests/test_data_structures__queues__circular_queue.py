"""Tests for data_structures/queues/circular_queue.py"""

import pytest

from data_structures.queues.circular_queue import CircularQueue


# ---------------------------------------------------------------------------
# Construction / __init__
# ---------------------------------------------------------------------------


def test_init_creates_empty_queue_with_given_capacity():
    """A new queue must have the requested capacity, all-None storage and zero size."""
    cq = CircularQueue(5)

    assert cq.n == 5
    assert cq.array == [None, None, None, None, None]
    assert cq.front == 0
    assert cq.rear == 0
    assert cq.size == 0


@pytest.mark.parametrize("capacity", [1, 2, 10, 100])
def test_init_capacity_variants(capacity):
    """The backing list length must always equal the requested capacity."""
    cq = CircularQueue(capacity)

    assert len(cq.array) == capacity
    assert cq.is_empty() is True


def test_init_with_zero_capacity_is_full_immediately():
    """A zero-capacity queue is empty but cannot accept any element."""
    cq = CircularQueue(0)

    assert cq.array == []
    assert cq.is_empty() is True
    assert len(cq) == 0
    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("A")


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_empty_queue_is_zero():
    """len() of a fresh queue must be 0."""
    assert len(CircularQueue(3)) == 0


def test_len_tracks_enqueue_and_dequeue():
    """len() must reflect the number of stored elements after mixed operations."""
    cq = CircularQueue(3)

    cq.enqueue("A")
    assert len(cq) == 1

    cq.enqueue("B")
    assert len(cq) == 2

    cq.dequeue()
    assert len(cq) == 1

    cq.dequeue()
    assert len(cq) == 0


def test_len_returns_int_type():
    """__len__ must return a plain int (required by the len() protocol)."""
    cq = CircularQueue(2)
    cq.enqueue(1)

    result = cq.__len__()

    assert isinstance(result, int)
    assert result == 1


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_is_empty_true_for_new_queue():
    """A newly created queue reports empty."""
    assert CircularQueue(4).is_empty() is True


def test_is_empty_false_after_enqueue_and_true_after_drain():
    """is_empty flips to False on enqueue and back to True once fully drained."""
    cq = CircularQueue(2)

    cq.enqueue("A")
    assert cq.is_empty() is False

    cq.dequeue()
    assert cq.is_empty() is True


# ---------------------------------------------------------------------------
# first
# ---------------------------------------------------------------------------


def test_first_on_empty_queue_returns_false():
    """first() on an empty queue returns the sentinel value False."""
    assert CircularQueue(3).first() is False


def test_first_returns_front_element_without_removing_it():
    """first() peeks at the head element and leaves the queue unchanged."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B")

    assert cq.first() == "A"
    assert len(cq) == 2
    assert cq.dequeue() == "A"


def test_first_after_wraparound():
    """first() must follow the front pointer after the queue wraps around."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()  # front -> 1
    cq.dequeue()  # front -> 2
    cq.enqueue("D")  # rear wraps to 0
    cq.enqueue("E")  # rear wraps to 1

    assert cq.first() == "C"
    assert cq.array == ["D", "E", "C"]


def test_first_returns_falsy_stored_value():
    """A stored falsy value (e.g. 0) is returned as-is, not confused with the sentinel."""
    cq = CircularQueue(2)
    cq.enqueue(0)

    assert cq.first() == 0
    assert cq.is_empty() is False


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_returns_self_for_chaining():
    """enqueue must return the queue instance to allow fluent chaining."""
    cq = CircularQueue(3)

    assert cq.enqueue("A") is cq


def test_enqueue_stores_element_at_rear_and_advances_pointers():
    """enqueue writes at rear, advances rear modulo n and increments size."""
    cq = CircularQueue(4)

    cq.enqueue("A")

    assert cq.array == ["A", None, None, None]
    assert cq.rear == 1
    assert cq.front == 0
    assert cq.size == 1


def test_enqueue_fills_queue_to_capacity():
    """Enqueuing exactly n elements must succeed and fill the array."""
    cq = CircularQueue(3)

    cq.enqueue("A").enqueue("B").enqueue("C")

    assert cq.array == ["A", "B", "C"]
    assert len(cq) == 3
    assert cq.rear == 0  # wrapped back to the start


def test_enqueue_raises_when_full():
    """Enqueuing into a full queue raises 'QUEUE IS FULL' and leaves state intact."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")

    with pytest.raises(Exception, match="QUEUE IS FULL"):
        cq.enqueue("C")

    assert cq.array == ["A", "B"]
    assert len(cq) == 2


def test_enqueue_wraps_around_after_dequeue():
    """After dequeues, enqueue must reuse freed slots at the beginning of the array."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B").enqueue("C")
    cq.dequeue()
    cq.dequeue()

    cq.enqueue("D")
    cq.enqueue("E")

    assert cq.array == ["D", "E", "C"]
    assert cq.rear == 2
    assert cq.front == 2
    assert len(cq) == 3


@pytest.mark.parametrize("value", ["A", 0, -1, 3.14, None, [1, 2], {"k": "v"}])
def test_enqueue_accepts_arbitrary_values(value):
    """enqueue must store any Python object without type restrictions."""
    cq = CircularQueue(2)

    cq.enqueue(value)

    assert cq.array[0] is value
    assert cq.first() is value


# ---------------------------------------------------------------------------
# dequeue
# ---------------------------------------------------------------------------


def test_dequeue_raises_underflow_on_empty_queue():
    """Dequeuing an empty queue raises 'UNDERFLOW'."""
    cq = CircularQueue(3)

    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


def test_dequeue_returns_front_and_clears_slot():
    """dequeue returns the front element, clears its slot and advances front."""
    cq = CircularQueue(3)
    cq.enqueue("A").enqueue("B")

    result = cq.dequeue()

    assert result == "A"
    assert cq.array == [None, "B", None]
    assert cq.front == 1
    assert cq.size == 1


def test_dequeue_last_element_resets_to_empty():
    """Dequeuing the final element leaves an empty queue with front == rear."""
    cq = CircularQueue(2)
    cq.enqueue("A")

    assert cq.dequeue() == "A"
    assert cq.is_empty() is True
    assert cq.front == cq.rear == 1
    assert cq.array == [None, None]


def test_dequeue_wraps_front_pointer():
    """The front pointer must wrap modulo n when it reaches the end of the array."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")

    assert cq.dequeue() == "A"
    assert cq.front == 1
    assert cq.dequeue() == "B"
    assert cq.front == 0


def test_dequeue_raises_underflow_after_draining():
    """Once drained, further dequeues raise UNDERFLOW again."""
    cq = CircularQueue(2)
    cq.enqueue("A")
    cq.dequeue()

    with pytest.raises(Exception, match="UNDERFLOW"):
        cq.dequeue()


# ---------------------------------------------------------------------------
# FIFO behaviour / integration
# ---------------------------------------------------------------------------


def test_fifo_order_preserved_across_wraparound():
    """Elements must come out in insertion order even after many wrap-arounds."""
    cq = CircularQueue(3)
    produced = []

    for i in range(10):
        if len(cq) == cq.n:
            produced.append(cq.dequeue())
        cq.enqueue(i)
        if i % 2 == 0:
            produced.append(cq.dequeue())

    while not cq.is_empty():
        produced.append(cq.dequeue())

    assert produced == list(range(10))


def test_alternating_enqueue_dequeue_keeps_consistent_state():
    """Repeated interleaved operations must keep size, front and rear consistent."""
    cq = CircularQueue(4)

    for i in range(20):
        cq.enqueue(i)
        assert len(cq) == 1
        assert cq.dequeue() == i
        assert cq.is_empty() is True
        assert cq.front == cq.rear

    assert cq.array == [None] * 4


def test_queue_can_be_refilled_after_being_full_and_drained():
    """A queue that was full and then drained must be reusable at full capacity."""
    cq = CircularQueue(2)
    cq.enqueue("A").enqueue("B")
    cq.dequeue()
    cq.dequeue()

    cq.enqueue("C").enqueue("D")

    assert cq.array == ["C", "D"]
    assert cq.dequeue() == "C"
    assert cq.dequeue() == "D"
