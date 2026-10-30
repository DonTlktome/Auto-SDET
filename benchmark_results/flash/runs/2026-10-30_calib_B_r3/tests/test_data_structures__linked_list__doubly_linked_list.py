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
class TestNode:
    """Tests for the Node helper class."""

    def test_node_initialization(self):
        """A new node stores data and has no neighbours."""
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
    def test_node_str(self, data, expected):
        """__str__ returns the string form of the stored data."""
        assert str(Node(data)) == expected

    def test_node_links_are_mutable(self):
        """previous/next attributes can be reassigned (edge case)."""
        first, second = Node(1), Node(2)
        first.next = second
        second.previous = first
        assert first.next is second
        assert second.previous is first


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestConstruction:
    """Tests for construction and the is_empty helper."""

    def test_new_list_is_empty(self):
        """A freshly created list has no head/tail and is empty."""
        ll = DoublyLinkedList()
        assert ll.head is None
        assert ll.tail is None
        assert ll.is_empty() is True
        assert len(ll) == 0
        assert str(ll) == ""
        assert list(ll) == []

    def test_is_empty_false_after_insert(self):
        """is_empty returns False once an element is inserted."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        assert ll.is_empty() is False

    def test_is_empty_true_after_removing_only_element(self):
        """Removing the single element makes the list empty again (edge case)."""
        ll = DoublyLinkedList()
        ll.insert_at_head("x")
        assert ll.delete_head() == "x"
        assert ll.is_empty() is True
        assert ll.head is None and ll.tail is None


# --------------------------------------------------------------------------- #
# Iteration / str / len
# --------------------------------------------------------------------------- #
class TestIterationAndRepresentation:
    """Tests for __iter__, __str__ and __len__."""

    def test_iter_empty(self):
        """Iterating an empty list yields nothing."""
        assert tuple(DoublyLinkedList()) == ()

    def test_iter_order(self):
        """Iteration yields values from head to tail."""
        ll = DoublyLinkedList()
        ll.insert_at_head("b")
        ll.insert_at_head("a")
        ll.insert_at_tail("c")
        assert tuple(ll) == ("a", "b", "c")

    def test_str_joins_with_arrow(self):
        """__str__ joins items with '->'."""
        ll = DoublyLinkedList()
        for value in ("a", "b", "c"):
            ll.insert_at_tail(value)
        assert str(ll) == "a->b->c"

    def test_str_single_element(self):
        """A single element stringifies without separators (edge case)."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(7)
        assert str(ll) == "7"

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 20])
    def test_len_matches_number_of_inserts(self, count):
        """len() equals the number of inserted elements."""
        ll = DoublyLinkedList()
        for i in range(count):
            ll.insert_at_tail(i)
        assert len(ll) == count

    def test_iter_is_lazy_generator(self):
        """__iter__ returns a generator, not a list."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        iterator = iter(ll)
        assert hasattr(iterator, "__next__")
        assert next(iterator) == 1
        with pytest.raises(StopIteration):
            next(iterator)


# --------------------------------------------------------------------------- #
# insert_at_nth / insert_at_head / insert_at_tail
# --------------------------------------------------------------------------- #
class TestInsertion:
    """Tests for the insertion methods."""

    def test_insert_at_head_on_empty(self):
        """Inserting at head of an empty list sets head and tail."""
        ll = DoublyLinkedList()
        ll.insert_at_head(1)
        assert ll.head is ll.tail
        assert ll.head.data == 1
        assert ll.head.previous is None
        assert ll.head.next is None

    def test_insert_at_tail_on_empty(self):
        """Inserting at tail of an empty list sets head and tail."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        assert ll.head is ll.tail
        assert ll.tail.data == 1

    def test_insert_at_head_order(self):
        """Repeated head insertions reverse the insertion order."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_head(value)
        assert str(ll) == "3->2->1"
        assert ll.head.data == 3
        assert ll.tail.data == 1

    def test_insert_at_tail_order(self):
        """Repeated tail insertions preserve the insertion order."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert str(ll) == "1->2->3"
        assert ll.head.data == 1
        assert ll.tail.data == 3

    def test_insert_at_nth_middle(self):
        """Inserting in the middle keeps the doubly linked structure intact."""
        ll = DoublyLinkedList()
        for value in (1, 2, 4):
            ll.insert_at_tail(value)
        ll.insert_at_nth(2, 3)
        assert str(ll) == "1->2->3->4"
        # verify both directions of the links
        node = ll.head
        while node.next:
            assert node.next.previous is node
            node = node.next
        assert node is ll.tail

    @pytest.mark.parametrize(
        "index, data, expected",
        [
            (0, 0, "0->1->2->3"),
            (1, 9, "1->9->2->3"),
            (2, 9, "1->2->9->3"),
            (3, 9, "1->2->3->9"),
            (3, 4, "1->2->3->4"),
        ],
    )
    def test_insert_at_nth_all_positions(self, index, data, expected):
        """Insertion works at every valid index including the boundaries."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        ll.insert_at_nth(index, data)
        assert str(ll) == expected
        assert len(ll) == 4

    @pytest.mark.parametrize("index", [-1, -100, 2, 5, 100])
    def test_insert_at_nth_invalid_index(self, index):
        """Out-of-range indices raise IndexError and leave the list unchanged."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        with pytest.raises(IndexError, match="list index out of range"):
            ll.insert_at_nth(index, 666)
        assert str(ll) == "1"

    def test_insert_at_nth_invalid_index_on_empty_list(self):
        """Any index other than 0 on an empty list raises IndexError (edge case)."""
        ll = DoublyLinkedList()
        with pytest.raises(IndexError):
            ll.insert_at_nth(1, 666)
        assert ll.is_empty() is True

    def test_insert_at_nth_duplicate_values(self):
        """Duplicate values are allowed and inserted at the requested index."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        ll.insert_at_nth(1, 1)
        assert str(ll) == "1->1"
        assert len(ll) == 2


# --------------------------------------------------------------------------- #
# delete_at_nth / delete_head / delete_tail
# --------------------------------------------------------------------------- #
class TestDeletionByIndex:
    """Tests for index-based deletion methods."""

    def test_delete_head_empty_raises(self):
        """Deleting the head of an empty list raises IndexError."""
        with pytest.raises(IndexError, match="list index out of range"):
            DoublyLinkedList().delete_head()

    def test_delete_tail_empty_raises(self):
        """Deleting the tail of an empty list raises IndexError."""
        with pytest.raises(IndexError, match="list index out of range"):
            DoublyLinkedList().delete_tail()

    def test_delete_at_nth_empty_raises(self):
        """Deleting any index from an empty list raises IndexError."""
        with pytest.raises(IndexError, match="list index out of range"):
            DoublyLinkedList().delete_at_nth(0)

    def test_delete_head(self):
        """delete_head removes and returns the first element."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete_head() == 1
        assert str(ll) == "2->3"
        assert ll.head.previous is None

    def test_delete_tail(self):
        """delete_tail removes and returns the last element."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete_tail() == 3
        assert str(ll) == "1->2"
        assert ll.tail.next is None

    def test_delete_only_element(self):
        """Deleting the only element resets head and tail to None (edge case)."""
        ll = DoublyLinkedList()
        ll.insert_at_tail("only")
        assert ll.delete_at_nth(0) == "only"
        assert ll.head is None
        assert ll.tail is None
        assert len(ll) == 0

    def test_delete_at_nth_middle(self):
        """Deleting a middle element relinks its neighbours."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete_at_nth(1) == 2
        assert str(ll) == "1->3"
        assert ll.head.next is ll.tail
        assert ll.tail.previous is ll.head

    @pytest.mark.parametrize(
        "index, expected_value, expected_str",
        [
            (0, 1, "2->3->4"),
            (1, 2, "1->3->4"),
            (2, 3, "1->2->4"),
            (3, 4, "1->2->3"),
        ],
    )
    def test_delete_at_nth_all_positions(self, index, expected_value, expected_str):
        """Deletion works at every valid index."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3, 4):
            ll.insert_at_tail(value)
        assert ll.delete_at_nth(index) == expected_value
        assert str(ll) == expected_str
        assert len(ll) == 3

    @pytest.mark.parametrize("index", [-1, 3, 10])
    def test_delete_at_nth_invalid_index(self, index):
        """Out-of-range deletion raises IndexError and keeps the list intact."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        with pytest.raises(IndexError, match="list index out of range"):
            ll.delete_at_nth(index)
        assert str(ll) == "1->2->3"

    def test_delete_all_elements_one_by_one(self):
        """Deleting from the head repeatedly empties the list."""
        ll = DoublyLinkedList()
        for value in range(5):
            ll.insert_at_tail(value)
        for expected in range(5):
            assert ll.delete_head() == expected
        assert ll.is_empty() is True


# --------------------------------------------------------------------------- #
# delete(data)
# --------------------------------------------------------------------------- #
class TestDeleteByValue:
    """Tests for the value-based delete method."""

    def test_delete_head_value(self):
        """Deleting the head value updates the head pointer."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete(1) == 1
        assert str(ll) == "2->3"
        assert ll.head.data == 2

    def test_delete_tail_value(self):
        """Deleting the tail value updates the tail pointer."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete(3) == 3
        assert str(ll) == "1->2"
        assert ll.tail.data == 2

    def test_delete_middle_value(self):
        """Deleting a middle value relinks the surrounding nodes."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        assert ll.delete(2) == 2
        assert str(ll) == "1->3"
        assert ll.head.next is ll.tail
        assert ll.tail.previous is ll.head

    def test_delete_only_value(self):
        """Deleting the only value empties the list (edge case)."""
        ll = DoublyLinkedList()
        ll.insert_at_tail("solo")
        assert ll.delete("solo") == "solo"
        assert ll.is_empty() is True

    def test_delete_missing_value_raises(self):
        """Deleting a value that is not present raises ValueError."""
        ll = DoublyLinkedList()
        for value in (1, 2, 3):
            ll.insert_at_tail(value)
        with pytest.raises(ValueError, match="No data matching given value"):
            ll.delete(99)
        assert str(ll) == "1->2->3"

    def test_delete_from_empty_list_raises(self):
        """Deleting from an empty list raises AttributeError (documented edge case)."""
        ll = DoublyLinkedList()
        with pytest.raises(AttributeError):
            ll.delete(1)

    def test_delete_first_occurrence_only(self):
        """Only the first matching occurrence is removed."""
        ll = DoublyLinkedList()
        for value in (1, 2, 1):
            ll.insert_at_tail(value)
        assert ll.delete(1) == 1
        assert str(ll) == "2->1"


# --------------------------------------------------------------------------- #
# Integration / module-level helper
# --------------------------------------------------------------------------- #
class TestIntegration:
    """End-to-end behaviour and the module-level self-test."""

    def test_module_level_test_function(self):
        """The module's own test_doubly_linked_list helper passes."""
        assert test_doubly_linked_list() is None

    def test_mixed_operations_keep_links_consistent(self):
        """A sequence of mixed operations keeps forward/backward links valid."""
        ll = DoublyLinkedList()
        for i in range(10):
            ll.insert_at_nth(i, i + 1)
        ll.insert_at_head(0)
        ll.insert_at_tail(11)
        assert str(ll) == "->".join(str(i) for i in range(12))

        assert ll.delete_head() == 0
        assert ll.delete_at_nth(9) == 10
        assert ll.delete_tail() == 11
        assert ll.delete(5) == 5
        assert len(ll) == 8
        assert str(ll) == "1->2->3->4->6->7->8->9"

        # walk forward and verify the backward links
        forward = []
        node = ll.head
        while node:
            forward.append(node.data)
            if node.next:
                assert node.next.previous is node
            node = node.next
        assert forward == [1, 2, 3, 4, 6, 7, 8, 9]

        # walk backward and verify the forward links
        backward = []
        node = ll.tail
        while node:
            backward.append(node.data)
            if node.previous:
                assert node.previous.next is node
            node = node.previous
        assert backward == list(reversed(forward))

    def test_list_accepts_mixed_types(self):
        """The list stores heterogeneous data without coercion."""
        ll = DoublyLinkedList()
        ll.insert_at_tail(1)
        ll.insert_at_tail("two")
        ll.insert_at_tail(None)
        assert list(ll) == [1, "two", None]
        assert str(ll) == "1->two->None"
