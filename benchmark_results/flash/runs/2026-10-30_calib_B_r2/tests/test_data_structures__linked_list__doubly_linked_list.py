"""
Pytest test suite for data_structures/linked_list/doubly_linked_list.py

Covers Node, DoublyLinkedList (iteration, str, len, insertions, deletions,
is_empty) and the module-level test_doubly_linked_list helper.
"""

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
    """A fresh Node stores data and has no previous/next links."""
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


def test_node_links_are_mutable():
    """Node.previous / Node.next can be wired up manually."""
    first, second = Node(1), Node(2)
    first.next = second
    second.previous = first
    assert first.next is second
    assert second.previous is first


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
    assert list(linked_list) == []


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
    """Iteration walks head -> tail yielding node data."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head("b")
    linked_list.insert_at_head("a")
    linked_list.insert_at_tail("c")
    assert tuple(linked_list) == ("a", "b", "c")


def test_iter_is_lazy_generator():
    """__iter__ returns a generator, so it can be consumed only once."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(1)
    iterator = iter(linked_list)
    assert iter(iterator) is iterator
    assert list(iterator) == [1]
    assert list(iterator) == []


def test_str_joins_with_arrows():
    """str() renders values joined by '->'."""
    linked_list = DoublyLinkedList()
    for value in ("a", "b", "c"):
        linked_list.insert_at_tail(value)
    assert str(linked_list) == "a->b->c"


def test_str_single_element():
    """A single-element list renders without any arrow."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head(7)
    assert str(linked_list) == "7"


@pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
def test_len_matches_number_of_insertions(count):
    """len() equals the number of inserted elements."""
    linked_list = DoublyLinkedList()
    for i in range(count):
        linked_list.insert_at_tail(i)
    assert len(linked_list) == count


# --------------------------------------------------------------------------- #
# insert_at_nth / insert_at_head / insert_at_tail
# --------------------------------------------------------------------------- #


def test_insert_at_nth_into_empty_list():
    """Inserting at index 0 of an empty list sets both head and tail."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_nth(0, 2)
    assert linked_list.head is linked_list.tail
    assert linked_list.head.data == 2
    assert linked_list.head.previous is None
    assert linked_list.head.next is None


def test_insert_at_nth_builds_expected_sequence():
    """The doctest sequence of insertions yields '1->2->3->4'."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_nth(0, 2)
    linked_list.insert_at_nth(0, 1)
    linked_list.insert_at_nth(2, 4)
    linked_list.insert_at_nth(2, 3)
    assert str(linked_list) == "1->2->3->4"


def test_insert_at_nth_middle_links_are_consistent():
    """Inserting in the middle keeps previous/next pointers coherent."""
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


def test_insert_at_head_prepends():
    """insert_at_head puts the value at the front."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(2)
    linked_list.insert_at_head(1)
    assert str(linked_list) == "1->2"
    assert linked_list.head.data == 1
    assert linked_list.head.previous is None


def test_insert_at_tail_appends():
    """insert_at_tail puts the value at the back and updates tail."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_head(1)
    linked_list.insert_at_tail(2)
    assert str(linked_list) == "1->2"
    assert linked_list.tail.data == 2
    assert linked_list.tail.next is None


@pytest.mark.parametrize("bad_index", [-1, -100, 1, 5])
def test_insert_at_nth_out_of_range_raises(bad_index):
    """Out-of-range indices raise IndexError and leave the list untouched."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError, match="list index out of range"):
        linked_list.insert_at_nth(bad_index, 666)
    assert linked_list.is_empty() is True


def test_insert_at_nth_index_equal_to_length_is_allowed():
    """Index == len(list) is a valid append position."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(1)
    linked_list.insert_at_nth(1, 2)
    assert str(linked_list) == "1->2"


def test_insert_at_nth_does_not_mutate_on_error():
    """A failed insertion must not corrupt an existing list."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    with pytest.raises(IndexError):
        linked_list.insert_at_nth(4, 99)
    assert str(linked_list) == "1->2->3"
    assert len(linked_list) == 3


# --------------------------------------------------------------------------- #
# delete_at_nth / delete_head / delete_tail
# --------------------------------------------------------------------------- #


def test_delete_at_nth_from_empty_raises():
    """Deleting from an empty list raises IndexError."""
    linked_list = DoublyLinkedList()
    with pytest.raises(IndexError, match="list index out of range"):
        linked_list.delete_at_nth(0)


def test_delete_at_nth_sequence_matches_doctest():
    """The doctest deletion sequence returns 1, 5, 3 leaving '2->4'."""
    linked_list = DoublyLinkedList()
    for i in range(5):
        linked_list.insert_at_nth(i, i + 1)
    assert linked_list.delete_at_nth(0) == 1
    assert linked_list.delete_at_nth(3) == 5
    assert linked_list.delete_at_nth(1) == 3
    assert str(linked_list) == "2->4"
    with pytest.raises(IndexError):
        linked_list.delete_at_nth(2)


def test_delete_at_nth_only_element_resets_head_and_tail():
    """Removing the sole element clears both head and tail."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail("only")
    assert linked_list.delete_at_nth(0) == "only"
    assert linked_list.head is None
    assert linked_list.tail is None
    assert linked_list.is_empty() is True


def test_delete_at_nth_middle_relinks_neighbours():
    """Deleting a middle node reconnects its neighbours."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete_at_nth(1) == 2
    assert str(linked_list) == "1->3"
    assert linked_list.head.next is linked_list.tail
    assert linked_list.tail.previous is linked_list.head


@pytest.mark.parametrize("bad_index", [-1, 3, 10])
def test_delete_at_nth_out_of_range_raises(bad_index):
    """Indices outside [0, len-1] raise IndexError."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    with pytest.raises(IndexError, match="list index out of range"):
        linked_list.delete_at_nth(bad_index)
    assert str(linked_list) == "1->2->3"


def test_delete_head_returns_first_value():
    """delete_head removes and returns the head value."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete_head() == 1
    assert str(linked_list) == "2->3"
    assert linked_list.head.previous is None


def test_delete_head_on_empty_raises():
    """delete_head on an empty list raises IndexError."""
    with pytest.raises(IndexError):
        DoublyLinkedList().delete_head()


def test_delete_tail_returns_last_value():
    """delete_tail removes and returns the tail value."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete_tail() == 3
    assert str(linked_list) == "1->2"
    assert linked_list.tail.next is None


def test_delete_tail_on_empty_raises():
    """delete_tail on an empty list raises IndexError."""
    with pytest.raises(IndexError):
        DoublyLinkedList().delete_tail()


def test_delete_tail_single_element():
    """Deleting the tail of a one-element list empties the list."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail(5)
    assert linked_list.delete_tail() == 5
    assert linked_list.is_empty() is True


# --------------------------------------------------------------------------- #
# delete(data)
# --------------------------------------------------------------------------- #


def test_delete_head_value():
    """delete() removes the head when it matches."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(1) == 1
    assert str(linked_list) == "2->3"


def test_delete_tail_value():
    """delete() removes the tail when it matches."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(3) == 3
    assert str(linked_list) == "1->2"


def test_delete_middle_value():
    """delete() removes a middle node and relinks neighbours."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(2) == 2
    assert str(linked_list) == "1->3"
    assert linked_list.head.next is linked_list.tail
    assert linked_list.tail.previous is linked_list.head


def test_delete_missing_value_raises():
    """delete() raises ValueError when no node matches."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 3):
        linked_list.insert_at_tail(value)
    with pytest.raises(ValueError, match="No data matching given value"):
        linked_list.delete(99)
    assert str(linked_list) == "1->2->3"


def test_delete_only_element():
    """delete() on a single-element list empties it."""
    linked_list = DoublyLinkedList()
    linked_list.insert_at_tail("x")
    assert linked_list.delete("x") == "x"
    assert linked_list.is_empty() is True


def test_delete_removes_first_occurrence_only():
    """delete() stops at the first matching node."""
    linked_list = DoublyLinkedList()
    for value in (1, 2, 1):
        linked_list.insert_at_tail(value)
    assert linked_list.delete(1) == 1
    assert str(linked_list) == "2->1"


# --------------------------------------------------------------------------- #
# Module-level helper
# --------------------------------------------------------------------------- #


def test_module_level_test_doubly_linked_list_helper():
    """The module's own self-check helper runs without raising."""
    assert test_doubly_linked_list() is None


# --------------------------------------------------------------------------- #
# Integration / stress
# --------------------------------------------------------------------------- #


def test_list_behaves_like_python_list():
    """A randomized-ish sequence of operations matches a plain list model."""
    linked_list = DoublyLinkedList()
    model = []

    for i in range(10):
        linked_list.insert_at_tail(i)
        model.append(i)

    linked_list.insert_at_head(-1)
    model.insert(0, -1)

    linked_list.insert_at_nth(5, 100)
    model.insert(5, 100)

    assert list(linked_list) == model
    assert len(linked_list) == len(model)

    assert linked_list.delete_at_nth(5) == model.pop(5)
    assert linked_list.delete_head() == model.pop(0)
    assert linked_list.delete_tail() == model.pop()

    assert list(linked_list) == model
    assert str(linked_list) == "->".join(str(v) for v in model)


def test_forward_and_backward_traversal_agree():
    """Walking next from head and previous from tail yields mirrored data."""
    linked_list = DoublyLinkedList()
    for value in range(6):
        linked_list.insert_at_tail(value)

    forward = []
    node = linked_list.head
    while node is not None:
        forward.append(node.data)
        node = node.next

    backward = []
    node = linked_list.tail
    while node is not None:
        backward.append(node.data)
        node = node.previous

    assert forward == list(range(6))
    assert backward == list(reversed(forward))