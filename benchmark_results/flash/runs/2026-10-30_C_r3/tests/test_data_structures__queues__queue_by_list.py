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
        ([None], [None]),
    ],
)
def test_init_with_iterable(iterable, expected):
    """The constructor must copy all items from any iterable into a list."""
    queue = QueueByList(iterable)
    assert queue.entries == expected
    assert len(queue) == len(expected)


def test_init_with_generator():
    """A generator must be fully consumed by the constructor."""
    queue = QueueByList(i**2 for i in range(1, 4))
    assert queue.entries == [1, 4, 9]


def test_init_does_not_alias_input_list():
    """Mutating the original list must not affect the queue's internal storage."""
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
    """Length must grow on put and shrink on get."""
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
    """repr must render the queue contents as a tuple."""
    assert repr(QueueByList(items)) == expected


def test_str_matches_repr():
    """str() should fall back to __repr__."""
    queue = QueueByList([1, 2, 3])
    assert str(queue) == repr(queue) == "Queue((1, 2, 3))"


def test_repr_reflects_mutations():
    """repr must reflect items added after construction."""
    queue = QueueByList()
    queue.put(10)
    queue.put(20)
    queue.put(30)
    assert repr(queue) == "Queue((10, 20, 30))"


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------

def test_put_single_item():
    """put appends a single item."""
    queue = QueueByList()
    queue.put(10)
    assert queue.entries == [10]


def test_put_returns_none():
    """put returns None."""
    queue = QueueByList()
    assert queue.put(10) is None


def test_put_preserves_fifo_order():
    """Items are appended in insertion order."""
    queue = QueueByList()
    for i in range(5):
        queue.put(i)
    assert queue.entries == [0, 1, 2, 3, 4]


def test_put_duplicates_and_none():
    """Duplicates and None values are stored as-is."""
    queue = QueueByList()
    queue.put(None)
    queue.put(None)
    assert queue.entries == [None, None]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_returns_front_item():
    """get returns the front (oldest) item."""
    queue = QueueByList([10, 20, 30])
    assert queue.get() == 10


def test_get_after_put():
    """get must respect FIFO order, returning the original front item first."""
    queue = QueueByList((10, 20, 30))
    queue.put(40)
    assert queue.get() == 10
    assert queue.get() == 20
    assert queue.get() == 30
    assert queue.get() == 40


def test_get_empty_raises_index_error():
    """get on an empty queue raises IndexError."""
    with pytest.raises(IndexError):
        QueueByList().get()


def test_get_until_empty_then_raises():
    """get consumes items and then raises when empty."""
    queue = QueueByList([1, 2])
    assert queue.get() == 1
    assert queue.get() == 2
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
        (4, [10, 20, 30, 40]),
        (5, [20, 30, 40, 10]),
    ],
)
def test_rotate(rotation, expected):
    """rotate moves the front item to the back the given number of times."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(rotation)
    assert queue.entries == expected


def test_rotate_sequential_calls():
    """Successive rotate calls accumulate."""
    queue = QueueByList([10, 20, 30, 40])
    queue.rotate(1)
    assert queue.entries == [20, 30, 40, 10]
    queue.rotate(2)
    assert queue.entries == [40, 10, 20, 30]


def test_rotate_empty_queue():
    """Rotating an empty queue with a positive count raises IndexError."""
    queue = QueueByList()
    with pytest.raises(IndexError):
        queue.rotate(3)


def test_rotate_single_element():
    """Rotating a single-element queue leaves it unchanged."""
    queue = QueueByList([42])
    queue.rotate(5)
    assert queue.entries == [42]


def test_rotate_negative_is_noop():
    """A negative count produces an empty range, so the queue is unchanged."""
    queue = QueueByList([1, 2, 3])
    queue.rotate(-1)
    assert queue.entries == [1, 2, 3]


# ---------------------------------------------------------------------------
# get_front
# ---------------------------------------------------------------------------

def test_get_front_does_not_remove():
    """get_front peeks without removing."""
    queue = QueueByList([10, 20, 30])
    assert queue.get_front() == 10
    assert queue.entries == [10, 20, 30]


def test_get_front_after_get():
    """get_front reflects the item after a get."""
    queue = QueueByList([10, 20, 30])
    assert queue.get() == 10
    assert queue.get_front() == 20


def test_get_front_empty_raises_index_error():
    """get_front on an empty queue raises IndexError."""
    with pytest.raises(IndexError):
        QueueByList().get_front()


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

def test_fifo_integration():
    """A full push/pop cycle preserves strict FIFO ordering."""
    queue = QueueByList()
    for i in range(5):
        queue.put(i)
    results = [queue.get() for _ in range(5)]
    assert results == [0, 1, 2, 3, 4]
