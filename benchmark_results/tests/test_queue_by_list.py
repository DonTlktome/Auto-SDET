"""Pytest test suite for QueueByList (queue_by_list.py)."""

from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import pytest

from data_structures.queues.queue_by_list import QueueByList


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for QueueByList.__init__."""

    def test_default_constructor_creates_empty_queue(self) -> None:
        """Positive: no argument → empty queue."""
        q = QueueByList()
        assert len(q) == 0
        assert repr(q) == "Queue(())"

    @pytest.mark.parametrize(
        "iterable, expected_repr",
        [
            ([10, 20, 30], "Queue((10, 20, 30))"),
            ((1, 2), "Queue((1, 2))"),
            (range(3), "Queue((0, 1, 2))"),
            ("abc", "Queue(('a', 'b', 'c'))"),
            ([], "Queue(())"),
        ],
    )
    def test_init_with_various_iterables(
        self, iterable: Iterable, expected_repr: str
    ) -> None:
        """Positive: constructor accepts list, tuple, range, str, empty list."""
        q = QueueByList(iterable)
        assert repr(q) == expected_repr

    def test_init_with_generator(self) -> None:
        """Positive: constructor consumes a generator expression."""
        q = QueueByList(i**2 for i in range(1, 4))
        assert repr(q) == "Queue((1, 4, 9))"

    def test_init_with_none_is_empty(self) -> None:
        """Edge: explicit None yields empty queue."""
        q = QueueByList(None)
        assert len(q) == 0

    def test_init_copies_input_list(self) -> None:
        """Edge: mutating the source list after construction does not affect queue."""
        src = [1, 2, 3]
        q = QueueByList(src)
        src.append(99)
        assert len(q) == 3
        assert q.get_front() == 1


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    """Tests for QueueByList.__len__."""

    def test_len_empty(self) -> None:
        """Positive: empty queue length is 0."""
        assert len(QueueByList()) == 0

    @pytest.mark.parametrize("n", [1, 5, 100])
    def test_len_after_puts(self, n: int) -> None:
        """Positive: len reflects number of put items."""
        q = QueueByList()
        for i in range(n):
            q.put(i)
        assert len(q) == n

    def test_len_after_get(self) -> None:
        """Positive: len decreases after get."""
        q = QueueByList([10, 20, 30])
        q.get()
        assert len(q) == 2

    def test_len_after_rotate(self) -> None:
        """Edge: rotate does not change length."""
        q = QueueByList([1, 2, 3, 4])
        q.rotate(2)
        assert len(q) == 4


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    """Tests for QueueByList.__repr__."""

    def test_repr_empty(self) -> None:
        """Positive: repr of empty queue."""
        assert repr(QueueByList()) == "Queue(())"

    def test_repr_single_item(self) -> None:
        """Positive: repr with one item includes trailing comma."""
        q = QueueByList([42])
        assert repr(q) == "Queue((42,))"

    def test_repr_multiple_items(self) -> None:
        """Positive: repr with multiple items."""
        q = QueueByList([1, 2, 3])
        assert repr(q) == "Queue((1, 2, 3))"

    def test_str_matches_repr(self) -> None:
        """Edge: str(q) equals repr(q)."""
        q = QueueByList([10, 20])
        assert str(q) == repr(q)


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

class TestPut:
    """Tests for QueueByList.put."""

    def test_put_single_item(self) -> None:
        """Positive: put one item into empty queue."""
        q = QueueByList()
        q.put(10)
        assert len(q) == 1
        assert q.get_front() == 10

    def test_put_multiple_items(self) -> None:
        """Positive: put several items; order is FIFO."""
        q = QueueByList()
        items = [1, 2, 3, 4, 5]
        for item in items:
            q.put(item)
        assert len(q) == 5
        # FIFO order
        for expected in items:
            assert q.get() == expected

    def test_put_none_value(self) -> None:
        """Edge: None is a valid item."""
        q = QueueByList()
        q.put(None)
        assert q.get() is None

    def test_put_after_get(self) -> None:
        """Edge: put after some gets maintains correct order."""
        q = QueueByList([10, 20, 30])
        q.get()  # removes 10
        q.put(40)
        assert list(q.entries) == [20, 30, 40]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    """Tests for QueueByList.get."""

    def test_get_returns_front_item(self) -> None:
        """Positive: get returns the oldest item (FIFO)."""
        q = QueueByList([10, 20, 30])
        assert q.get() == 10
        assert q.get() == 20
        assert q.get() == 30

    def test_get_removes_item(self) -> None:
        """Positive: get removes the item from the queue."""
        q = QueueByList([10, 20])
        q.get()
        assert len(q) == 1
        assert q.get_front() == 20

    def test_get_on_empty_queue_raises_index_error(self) -> None:
        """Edge: get on empty queue raises IndexError with message."""
        q = QueueByList()
        with pytest.raises(IndexError, match="Queue is empty"):
            q.get()

    def test_get_until_empty_then_raises(self) -> None:
        """Edge: drain queue then get raises."""
        q = QueueByList([1])
        q.get()
        with pytest.raises(IndexError):
            q.get()

    def test_get_after_rotate_preserves_fifo(self) -> None:
        """Edge: get after rotate still returns correct front."""
        q = QueueByList([10, 20, 30, 40])
        q.rotate(2)  # now [30, 40, 10, 20]
        assert q.get() == 30
        assert q.get() == 40


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

class TestRotate:
    """Tests for QueueByList.rotate."""

    @pytest.mark.parametrize(
        "initial, rotation, expected",
        [
            ([10, 20, 30, 40], 1, [20, 30, 40, 10]),
            ([10, 20, 30, 40], 2, [30, 40, 10, 20]),
            ([10, 20, 30, 40], 3, [40, 10, 20, 30]),
            ([10, 20, 30, 40], 4, [10, 20, 30, 40]),  # full rotation
            ([10, 20, 30, 40], 0, [10, 20, 30, 40]),  # zero rotation
        ],
    )
    def test_rotate_positive(
        self, initial: list, rotation: int, expected: list
    ) -> None:
        """Positive: rotate by various amounts produces correct order."""
        q = QueueByList(initial)
        q.rotate(rotation)
        assert list(q.entries) == expected

    def test_rotate_empty_queue(self) -> None:
        """Edge: rotating an empty queue raises IndexError."""
        q = QueueByList()
        with pytest.raises(IndexError):
            q.rotate(5)

    def test_rotate_single_item(self) -> None:
        """Edge: rotating a single-item queue leaves it unchanged."""
        q = QueueByList([99])
        q.rotate(3)
        assert list(q.entries) == [99]

    def test_rotate_large_rotation(self) -> None:
        """Edge: rotation larger than queue size wraps correctly."""
        q = QueueByList([1, 2, 3])
        q.rotate(7)  # 7 % 3 == 1 effective rotation
        assert list(q.entries) == [2, 3, 1]

    def test_rotate_negative_rotation(self) -> None:
        """Edge: negative rotation does nothing (no rotation)."""
        q = QueueByList([10, 20, 30, 40])
        q.rotate(-1)
        # Negative rotation: range(-1) is empty, so no items are moved.
        assert list(q.entries) == [10, 20, 30, 40]

    def test_rotate_does_not_change_length(self) -> None:
        """Edge: rotate preserves queue length."""
        q = QueueByList([1, 2, 3, 4, 5])
        original_len = len(q)
        q.rotate(3)
        assert len(q) == original_len


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------

class TestGetFront:
    """Tests for QueueByList.get_front."""

    def test_get_front_returns_first_item(self) -> None:
        """Positive: get_front returns the front item without removing it."""
        q = QueueByList([10, 20, 30])
        assert q.get_front() == 10

    def test_get_front_does_not_remove_item(self) -> None:
        """Positive: get_front is idempotent and does not mutate the queue."""
        q = QueueByList([10, 20, 30])
        assert q.get_front() == 10
        assert q.get_front() == 10
        assert len(q) == 3

    def test_get_front_after_get(self) -> None:
        """Positive: get_front reflects new front after a get."""
        q = QueueByList([10, 20, 30])
        q.get()
        assert q.get_front() == 20

    def test_get_front_after_put(self) -> None:
        """Edge: get_front unchanged after put (put goes to back)."""
        q = QueueByList([10, 20])
        q.put(30)
        assert q.get_front() == 10

    def test_get_front_on_empty_queue_raises_index_error(self) -> None:
        """Edge: get_front on empty queue raises IndexError."""
        q = QueueByList()
        with pytest.raises(IndexError):
            q.get_front()

    def test_get_front_after_rotate(self) -> None:
        """Edge: get_front returns correct item after rotation."""
        q = QueueByList([10, 20, 30, 40])
        q.rotate(2)
        assert q.get_front() == 30


# ---------------------------------------------------------------------------
# Integration / combined scenarios
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end scenarios combining multiple operations."""

    def test_full_lifecycle(self) -> None:
        """Simulate a realistic producer-consumer flow."""
        q = QueueByList()

        # Produce
        for i in range(5):
            q.put(i)
        assert len(q) == 5
        assert q.get_front() == 0

        # Consume partially
        consumed = [q.get() for _ in range(3)]
        assert consumed == [0, 1, 2]
        assert len(q) == 2

        # Produce more
        q.put(100)
        q.put(200)
        assert list(q.entries) == [3, 4, 100, 200]

        # Rotate
        q.rotate(1)
        assert list(q.entries) == [4, 100, 200, 3]

        # Consume all
        remaining = []
        while q.entries:
            remaining.append(q.get())
        assert remaining == [4, 100, 200, 3]
        assert len(q) == 0

        with pytest.raises(IndexError):
            q.get()

    def test_generic_type_support(self) -> None:
        """Queue stores heterogeneous types correctly."""
        q = QueueByList([1, "two", 3.0, None])
        assert q.get() == 1
        assert q.get() == "two"
        assert q.get() == 3.0
        assert q.get() is None

    def test_large_queue_performance_smoke(self) -> None:
        """Smoke test: queue handles 10_000 items without error."""
        q = QueueByList(range(10_000))
        assert len(q) == 10_000
        assert q.get_front() == 0
        q.rotate(5_000)
        assert q.get_front() == 5_000
        # drain
        for _ in range(10_000):
            q.get()
        assert len(q) == 0