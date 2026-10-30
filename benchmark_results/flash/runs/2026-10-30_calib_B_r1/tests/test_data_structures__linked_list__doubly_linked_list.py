"""Pytest test suite for data_structures/linked_list/doubly_linked_list.py."""

import pytest

from data_structures.linked_list.doubly_linked_list import (
    DoublyLinkedList,
    Node,
    test_doubly_linked_list,
)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
def test_node_initialization_defaults():
    """A fresh Node stores data and has no neighbours."""
    node = Node(42)
    assert node.data == 42
    assert node.previous is None
    assert node.next is None


@pytest.mark.parametrize(
    "data, expected",
    [
        (1, "1"),
        ("abc", "abc"),
        (None, "None"),
        (3.5, "3.5"),
        ([1, 2], "[1, 2]"),
    ],
)
def test_node_str(data, expected):
    """Node.__str__ returns the string form of its data."""
    assert str(Node(data)) == expected


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
def test_new_list_is_empty():
    """A newly created list has no head/tail and reports empty."""
    linked_list = DoublyLinkedList()
    assert linked_list.head is None
    assert linked_list.tail is None
    assert linked_list.is_empty() is True
    assert len(linked_list) == 0
    assert str(linked_list) == ""
    assert tuple(linked_list) == ()


def test_is_empty_after_insert_and_delete():
    """is_empty flips to False on insert and back to True after removal."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(1)
    assert linked_list.is_empty() is False
    linked_list.delete_head()
    assert linked_list.is_empty() is True


# --------------------------------------------------------------------------- #
# Iteration / str / len
# --------------------------------------------------------------------------- #
def test_iter_yields_values_in_order():
    """Iteration walks from head to tail in insertion order."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head("b")
    linked_list.insert_at_head("a")
    linked_list.insert_at_tail("c")
    assert tuple(linked_list) == ("a", "b", "c")


def test_iter_is_lazy_generator():
    """__iter__ returns a generator, not a materialised sequence."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(1)
    iterator = iter(linked_list)
    assert iter(iterator) is iterator
    assert next(iterator) == 1
    with pytest.raises(StopIteration):
        next(iterator)


def test_str_joins_with_arrows():
    """str() renders values joined by '->'."""
    linked_list = DoublyLinkedList()
    for value in ("a", "b", "c"):
        linked_list.insert_at_tail(value)
    assert str(linked_list) == "a->b->c"


def test_len_counts_nodes():
    """len() equals the number of stored nodes."""
    linked_list = DoublyLinkedList()
    for i in range(5):
        linked_list.insert_at_nth(i, i + 1)
    assert len(linked_list) == 5


# --------------------------------------------------------------------------- #
# insert_at_nth / insert_at_head / insert_at_tail
# --------------------------------------------------------------------------- #
def test_insert_at_nth_builds_expected_sequence():
    """Inserting at various positions produces the documented ordering."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_nth(0, 2)
    linked_list.insert_at_nth(0, 1)
    linked_list.insert_at_nth(2, 4)
    linked_list.insert_at_nth(2, 3)
    assert str(linked_list) == "1->2->3->4"


def test_insert_at_nth_into_empty_list_sets_head_and_tail():
    """Inserting into an empty list sets both head and tail to the new node."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_nth(0, "only")
    assert linked_list.head is linked_list.tail
    assert linked_list.head.data == "only"
    assert linked_list.head.previous is None
    assert linked_list.head.next is None


def test_insert_at_nth_middle_links_are_consistent():
    """Middle insertion keeps previous/next pointers consistent."""
    linked_list = DoublyLinkedList()
    for value in (1, 3):
        linked_list.insert_at_tail(value)
    linked_list.insert_at_nth(1, 2)

    first, second, third = (
        linked_list.head,
        linked_list.head.next,
        linked_list.tail,
    )
    assert (first.data, second.data, third.data) == (1, 2, 3)
    assert first.previous is None
    assert first.next is second
    assert second.previous is first
    assert second.next is third
    assert third.previous is second
    assert third.next is None


@pytest.mark.parametrize("index", [-1, -100, 6, 7, 100])
def test_insert_at_nth_out_of_range_raises(index):
    """Out-of-range indices raise IndexError.

    A 5-node list accepts insertion indices 0 through 5 inclusive, so only
    indices outside that closed range are invalid.
    """
    linked_list = DoublyLinkedList()
    for i in range(5):
        linked_list.insert_at_tail(i)
    with pytest.raises(IndexError, match="list index out of range"):
        linked_list.insert_at_nth(index, 666)


def test_insert_at_nth_on_empty_list_rejects_nonzero_index():
    """An empty list only accepts index 0."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError):
        linked_list.insert_at_nth(1, 666)
    assert linked_list.is_empty() is True


def test_insert_at_head_and_tail():
    """insert_at_head prepends and insert_at_tail appends."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head(2)
    linked_list.insert_at_head(1)
    linked_list.insert_at_tail(3)
    assert str(linked_list) == "1->2->3"
    assert linked_list.head.data == 1
    assert linked_list.tail.data == 3


def test_insert_at_head_on_empty_list():
    """insert_at_head works on an empty list."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head("x")
    assert str(linked_list) == "x"
    assert linked_list.head is linked_list.tail


# --------------------------------------------------------------------------- #
# delete_at_nth / delete_head / delete_tail
# --------------------------------------------------------------------------- #
def test_delete_at_nth_sequence():
    """Deleting by index returns the removed values in the documented order."""
    linked_list = DoublyLinkedList()
    for i in range(5):
        linked_list.insert_at_nth(i, i + 1)
    assert linked_list.delete_at_nth(0) == 1
    assert linked_list.delete_at_nth(3) == 5
    assert linked_list.delete_at_nth(1) == 3
    assert str(linked_list) == "2->4"


def test_delete_at_nth_only_node_resets_head_and_tail():
    """Deleting the sole node clears head and tail."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail("solo")
    assert linked_list.delete_at_nth(0) == "solo"
    assert linked_list.head is None
    assert linked_list.tail is None
    assert linked_list.is_empty() is True


def test_delete_at_nth_middle_relinks_neighbours():
    """Deleting a middle node reconnects its neighbours."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete_at_nth(1) == 2
    assert linked_list.head.next is linked_list.tail
    assert linked_list.tail.previous is linked_list.head
    assert str(linked_list) == "1->3"


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_delete_at_nth_out_of_range_raises(index):
    """Deleting at an invalid index raises IndexError."""
    linked_list = DoublyLinkedList()
    for i in range(5):
        linked_list.insert_at_tail(i)
    with pytest.raises(IndexError, match="list index out of range"):
        linked_list.delete_at_nth(index)


def test_delete_at_nth_on_empty_list_raises():
    """Deleting from an empty list raises IndexError."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError):
        linked_list.delete_at_nth(0)


def test_delete_head_and_tail():
    """delete_head removes the first node and delete_tail the last."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete_head() == 1
    assert linked_list.delete_tail() == 3
    assert str(linked_list) == "2"
    assert linked_list.head is linked_list.tail


def test_delete_head_on_empty_list_raises():
    """delete_head on an empty list raises IndexError."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError):
        linked_list.delete_head()


def test_delete_tail_on_empty_list_raises():
    """delete_tail on an empty list raises IndexError."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError):
        linked_list.delete_tail()


# --------------------------------------------------------------------------- #
# delete(data)
# --------------------------------------------------------------------------- #
def test_delete_head_value():
    """Deleting the head value by data removes the first node."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(1) == 1
    assert str(linked_list) == "2->3"
    assert linked_list.head.previous is None


def test_delete_tail_value():
    """Deleting the tail value by data removes the last node."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(3) == 3
    assert str(linked_list) == "1->2"
    assert linked_list.tail.next is None


def test_delete_middle_value():
    """Deleting a middle value by data relinks its neighbours."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(2) == 2
    assert str(linked_list) == "1->3"
    assert linked_list.head.next is linked_list.tail
    assert linked_list.tail.previous is linked_list.head


def test_delete_missing_value_raises():
    """Deleting a value that is not present raises ValueError."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    with pytest.raises(ValueError, match="No data matching given value"):
        linked_list.delete(99)
    assert str(linked_list) == "1->2->3"


def test_delete_from_empty_list_raises():
    """Deleting from an empty list raises (AttributeError on None head)."""
    linked_list = DoublyLinkedList()
    with pytest.raises(AttributeError):
        linked_list.delete(1)


# --------------------------------------------------------------------------- #
# Module-level self test
# --------------------------------------------------------------------------- #
def test_module_level_test_function_passes():
    """The module's own test_doubly_linked_list helper runs without error."""
    assert test_doubly_linked_list() is None


# --------------------------------------------------------------------------- #
# Integration / round-trip behaviour
# --------------------------------------------------------------------------- #
def test_full_round_trip_matches_list_semantics():
    """A sequence of operations matches equivalent Python list behaviour."""
    linked_list = DoublyLinkedList()
    reference = []

    operations = [
        ("insert_at_tail", 1),
        ("insert_at_head", 0),
        ("insert_at_tail", 2),
        ("insert_at_nth", (2, 99)),
        ("delete_head", None),
        ("delete_tail", None),
        ("delete", 99),
    ]

    for op, arg in operations:
        if op == "insert_at_tail":
            linked_list.insert_at_tail(arg)
            reference.append(arg)
        elif op == "insert_at_head":
            linked_list.insert_at_head(arg)
            reference.insert(0, arg)
        elif op == "insert_at_nth":
            index, value = arg
            linked_list.insert_at_nth(index, value)
            reference.insert(index, value)
        elif op == "delete_head":
            assert linked_list.delete_head() == reference.pop(0)
        elif op == "delete_tail":
            assert linked_list.delete_tail() == reference.pop()
        elif op == "delete":
            assert linked_list.delete(arg) == arg
            reference.remove(arg)

        assert list(linked_list) == reference
        assert len(linked_list) == len(reference)

    assert str(linked_list) == "->".join(str(v) for v in reference)


def test_backward_traversal_via_previous_pointers():
    """Walking backwards from tail yields the reverse of forward iteration."""
    linked_list = DoublyLinkedList()
    for value in range(5):
        linked_list.insert_at_tail(value)

    backwards = []
    node = linked_list.tail
    while node is not None:
        backwards.append(node.data)
        node = node.previous

    assert backwards == list(reversed(list(linked_list)))
