"""Pytest test suite for singly_linked_list.py.

Covers Node, LinkedList (iteration, length, repr, indexing, insertion,
deletion, emptiness, reversal) and the module-level helper tests.
"""

from __future__ import annotations

import pytest

from data_structures.linked_list.singly_linked_list import (
    LinkedList,
    Node,
    test_singly_linked_list,
    test_singly_linked_list_2,
)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
class TestNode:
    """Tests for the Node dataclass."""

    def test_node_default_next_is_none(self) -> None:
        """A freshly created Node must have next_node set to None."""
        node = Node(20)
        assert node.data == 20
        assert node.next_node is None

    def test_node_repr(self) -> None:
        """repr()/str() of a Node must be 'Node(<data>)'."""
        assert repr(Node(10)) == "Node(10)"
        assert str(Node(10)) == "Node(10)"
        assert Node(10).__repr__() == "Node(10)"

    @pytest.mark.parametrize(
        "data, expected",
        [
            (20, "Node(20)"),
            ("Hello, world!", "Node(Hello, world!)"),
            (None, "Node(None)"),
            (True, "Node(True)"),
            (-1.5, "Node(-1.5)"),
        ],
    )
    def test_node_repr_parametrized(self, data, expected) -> None:
        """Node repr works for a variety of data types."""
        assert repr(Node(data)) == expected

    def test_node_chaining(self) -> None:
        """A Node can be linked to another Node via next_node."""
        second = Node(2)
        first = Node(1, second)
        assert first.next_node is second
        assert first.next_node.data == 2


# --------------------------------------------------------------------------- #
# LinkedList construction / emptiness
# --------------------------------------------------------------------------- #
class TestLinkedListBasics:
    """Tests for construction, emptiness and length."""

    def test_new_list_is_empty(self) -> None:
        """A new LinkedList has head None and is_empty() True."""
        ll = LinkedList()
        assert ll.head is None
        assert ll.is_empty() is True
        assert len(ll) == 0

    def test_is_empty_after_insert(self) -> None:
        """is_empty() becomes False after inserting an element."""
        ll = LinkedList()
        ll.insert_head("first")
        assert ll.is_empty() is False

    def test_is_empty_after_removing_all(self) -> None:
        """is_empty() returns True again once all elements are removed."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.delete_head()
        assert ll.is_empty() is True

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
    def test_len(self, count: int) -> None:
        """len() equals the number of inserted nodes."""
        ll = LinkedList()
        for i in range(count):
            ll.insert_tail(i)
        assert len(ll) == count


# --------------------------------------------------------------------------- #
# Iteration / repr
# --------------------------------------------------------------------------- #
class TestIterationAndRepr:
    """Tests for __iter__ and __repr__."""

    def test_iter_empty(self) -> None:
        """Iterating an empty list yields nothing."""
        assert list(LinkedList()) == []

    def test_iter_order(self) -> None:
        """Iteration yields data in insertion order."""
        ll = LinkedList()
        for value in ("tail", "tail_1", "tail_2"):
            ll.insert_tail(value)
        assert list(ll) == ["tail", "tail_1", "tail_2"]

    def test_iter_is_lazy_generator(self) -> None:
        """__iter__ returns an iterator (generator), not a list."""
        ll = LinkedList()
        ll.insert_tail(1)
        it = iter(ll)
        assert iter(it) is it
        assert next(it) == 1
        with pytest.raises(StopIteration):
            next(it)

    def test_repr_empty(self) -> None:
        """repr of an empty list is the empty string."""
        assert repr(LinkedList()) == ""
        assert str(LinkedList()) == ""

    def test_repr_multiple(self) -> None:
        """repr joins elements with ' -> '."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_tail(3)
        assert repr(ll) == "1 -> 3"
        assert str(ll) == "1 -> 3"
        ll.insert_tail(5)
        assert f"{ll}" == "1 -> 3 -> 5"

    def test_repr_mixed_types(self) -> None:
        """repr handles mixed types including None and Node instances."""
        ll = LinkedList()
        ll.insert_tail(None)
        ll.insert_tail(Node(7))
        ll.insert_tail(1.5)
        assert repr(ll) == "None -> Node(7) -> 1.5"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for __getitem__."""

    def test_getitem_valid(self) -> None:
        """Indexing returns the data at the given position."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        assert all(ll[i] == i for i in range(10))

    @pytest.mark.parametrize("index", [-1, -10, 10, 100])
    def test_getitem_out_of_range(self, index: int) -> None:
        """Out-of-range indices raise ValueError with the documented message."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        with pytest.raises(ValueError, match="list index out of range."):
            _ = ll[index]

    def test_getitem_empty_list(self) -> None:
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = LinkedList()[0]


class TestSetItem:
    """Tests for __setitem__."""

    def test_setitem_valid(self) -> None:
        """Assigning to a valid index changes the node data."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        ll[0] = 666
        ll[5] = -666
        assert ll[0] == 666
        assert ll[5] == -666
        assert len(ll) == 10

    @pytest.mark.parametrize("index", [-10, -1, 10, 11])
    def test_setitem_out_of_range(self, index: int) -> None:
        """Assigning out of range raises ValueError."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        with pytest.raises(ValueError, match="list index out of range."):
            ll[index] = 666

    def test_setitem_empty_list(self) -> None:
        """Assigning on an empty list raises ValueError."""
        with pytest.raises(ValueError):
            LinkedList()[0] = 1


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsertion:
    """Tests for insert_head, insert_tail and insert_nth."""

    def test_insert_tail(self) -> None:
        """insert_tail appends to the end."""
        ll = LinkedList()
        ll.insert_tail("tail")
        assert repr(ll) == "tail"
        ll.insert_tail("tail_2")
        ll.insert_tail("tail_3")
        assert repr(ll) == "tail -> tail_2 -> tail_3"

    def test_insert_head(self) -> None:
        """insert_head prepends to the front."""
        ll = LinkedList()
        ll.insert_head("head")
        assert repr(ll) == "head"
        ll.insert_head("head_2")
        ll.insert_head("head_3")
        assert repr(ll) == "head_3 -> head_2 -> head"

    def test_insert_nth_middle(self) -> None:
        """insert_nth inserts at the requested middle position."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.insert_nth(1, "fourth")
        assert repr(ll) == "first -> fourth -> second -> third"
        ll.insert_nth(3, "fifth")
        assert repr(ll) == "first -> fourth -> second -> fifth -> third"

    def test_insert_nth_into_empty(self) -> None:
        """insert_nth(0, x) on an empty list sets the head."""
        ll = LinkedList()
        ll.insert_nth(0, "only")
        assert ll.head is not None
        assert ll.head.data == "only"
        assert repr(ll) == "only"

    def test_insert_nth_at_end(self) -> None:
        """insert_nth(len(ll), x) appends to the end."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_nth(1, 2)
        assert repr(ll) == "1 -> 2"

    @pytest.mark.parametrize("index", [-1, -5, 2, 100])
    def test_insert_nth_out_of_range(self, index: int) -> None:
        """insert_nth raises IndexError for invalid indices.

        With a single element (len == 1) the only legal indices are 0 and 1,
        so 2 and above (and any negative index) must raise.
        """
        ll = LinkedList()
        ll.insert_tail("a")
        with pytest.raises(IndexError, match="list index out of range"):
            ll.insert_nth(index, "x")

    def test_insert_nth_out_of_range_empty(self) -> None:
        """insert_nth raises IndexError on an empty list for index > 0."""
        ll = LinkedList()
        with pytest.raises(IndexError):
            ll.insert_nth(1, "x")


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDeletion:
    """Tests for delete_head, delete_tail and delete_nth."""

    def _make(self) -> LinkedList:
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        return ll

    def test_delete_head(self) -> None:
        """delete_head removes and returns the first element."""
        ll = self._make()
        assert ll.delete_head() == "first"
        assert repr(ll) == "second -> third"
        assert ll.delete_head() == "second"
        assert ll.delete_head() == "third"
        assert ll.is_empty() is True

    def test_delete_head_empty(self) -> None:
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_head()

    def test_delete_tail(self) -> None:
        """delete_tail removes and returns the last element."""
        ll = self._make()
        assert ll.delete_tail() == "third"
        assert repr(ll) == "first -> second"
        assert ll.delete_tail() == "second"
        assert ll.delete_tail() == "first"
        assert ll.is_empty() is True

    def test_delete_tail_empty(self) -> None:
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_tail()

    def test_delete_nth_middle(self) -> None:
        """delete_nth removes and returns the element at the index."""
        ll = self._make()
        assert ll.delete_nth(1) == "second"
        assert repr(ll) == "first -> third"

    def test_delete_nth_default_index(self) -> None:
        """delete_nth defaults to index 0 (head)."""
        ll = self._make()
        assert ll.delete_nth() == "first"
        assert repr(ll) == "second -> third"

    def test_delete_nth_last(self) -> None:
        """delete_nth(len-1) removes the tail."""
        ll = self._make()
        assert ll.delete_nth(2) == "third"
        assert repr(ll) == "first -> second"

    @pytest.mark.parametrize("index", [-1, -5, 3, 5])
    def test_delete_nth_out_of_range(self, index: int) -> None:
        """delete_nth raises IndexError for invalid indices."""
        ll = self._make()
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_nth(index)

    def test_delete_nth_empty(self) -> None:
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_nth(0)

    def test_delete_nth_returns_none_data(self) -> None:
        """delete_nth returns None when the stored data is None."""
        ll = LinkedList()
        ll.insert_tail(None)
        assert ll.delete_nth(0) is None
        assert ll.is_empty() is True


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for print_list."""

    def test_print_list_output(self, capsys) -> None:
        """print_list writes the string representation to stdout."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second -> third\n"

    def test_print_list_empty(self, capsys) -> None:
        """print_list on an empty list prints an empty line."""
        LinkedList().print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse."""

    def test_reverse_multiple(self) -> None:
        """reverse inverts the order of the list."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.reverse()
        assert repr(ll) == "third -> second -> first"
        assert len(ll) == 3

    def test_reverse_empty(self) -> None:
        """Reversing an empty list leaves it empty."""
        ll = LinkedList()
        ll.reverse()
        assert ll.is_empty() is True
        assert repr(ll) == ""

    def test_reverse_single(self) -> None:
        """Reversing a single-element list is a no-op."""
        ll = LinkedList()
        ll.insert_tail(42)
        ll.reverse()
        assert repr(ll) == "42"
        assert ll.head is not None
        assert ll.head.next_node is None

    def test_reverse_twice_restores_order(self) -> None:
        """Reversing twice restores the original order."""
        ll = LinkedList()
        for i in range(5):
            ll.insert_tail(i)
        ll.reverse()
        ll.reverse()
        assert list(ll) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helper functions."""

    def test_singly_linked_list_helper(self) -> None:
        """The bundled test_singly_linked_list helper runs without error."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper(self) -> None:
        """The bundled test_singly_linked_list_2 helper runs without error."""
        assert test_singly_linked_list_2() is None


# --------------------------------------------------------------------------- #
# Integration-style scenario
# --------------------------------------------------------------------------- #
def test_full_scenario() -> None:
    """End-to-end scenario mixing all operations."""
    ll = LinkedList()
    assert ll.is_empty() is True
    assert str(ll) == ""

    for i in range(10):
        assert len(ll) == i
        ll.insert_nth(i, i + 1)
    assert str(ll) == " -> ".join(str(i) for i in range(1, 11))

    ll.insert_head(0)
    ll.insert_tail(11)
    assert str(ll) == " -> ".join(str(i) for i in range(12))

    assert ll.delete_head() == 0
    assert ll.delete_nth(9) == 10
    assert ll.delete_tail() == 11
    assert len(ll) == 9
    assert str(ll) == " -> ".join(str(i) for i in range(1, 10))

    for i in range(9):
        ll[i] = -i
    assert all(ll[i] == -i for i in range(9))

    ll.reverse()
    assert str(ll) == " -> ".join(str(i) for i in range(-8, 1))
