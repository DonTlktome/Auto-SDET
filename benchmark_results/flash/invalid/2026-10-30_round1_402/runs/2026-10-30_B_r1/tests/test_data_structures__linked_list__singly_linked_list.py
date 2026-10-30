"""Pytest test suite for singly_linked_list.py.

Covers Node, LinkedList (all public methods) and the module-level
``test_singly_linked_list`` / ``test_singly_linked_list_2`` helpers.
No network access or third-party dependencies are required.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

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

    @pytest.mark.parametrize(
        "data, expected",
        [
            (20, "Node(20)"),
            ("Hello, world!", "Node(Hello, world!)"),
            (None, "Node(None)"),
            (True, "Node(True)"),
            (3.5, "Node(3.5)"),
        ],
    )
    def test_repr(self, data, expected):
        """repr()/str() of a Node renders as 'Node(<data>)'."""
        node = Node(data)
        assert repr(node) == expected
        assert str(node) == expected

    def test_default_next_node_is_none(self):
        """A freshly created Node has next_node set to None."""
        node = Node(1)
        assert node.next_node is None

    def test_next_node_can_be_linked(self):
        """next_node can be assigned another Node instance."""
        first = Node(1)
        second = Node(2)
        first.next_node = second
        assert first.next_node is second
        assert first.next_node.data == 2

    def test_equality_is_value_based(self):
        """Dataclass equality compares data and next_node."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)
        assert Node(1) != Node(1, Node(2))


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestInitAndIsEmpty:
    """Tests for __init__ and is_empty."""

    def test_new_list_is_empty(self):
        """A new LinkedList has head None and is_empty() True."""
        ll = LinkedList()
        assert ll.head is None
        assert ll.is_empty() is True
        assert len(ll) == 0

    def test_is_empty_false_after_insert(self):
        """is_empty() returns False once a node is inserted."""
        ll = LinkedList()
        ll.insert_head("first")
        assert ll.is_empty() is False

    def test_is_empty_true_after_removing_only_node(self):
        """Removing the only node makes the list empty again."""
        ll = LinkedList()
        ll.insert_tail(1)
        assert ll.delete_head() == 1
        assert ll.is_empty() is True
        assert ll.head is None


# --------------------------------------------------------------------------- #
# Iteration / len / repr
# --------------------------------------------------------------------------- #
class TestIterLenRepr:
    """Tests for __iter__, __len__ and __repr__."""

    def test_iter_empty(self):
        """Iterating an empty list yields nothing."""
        assert list(LinkedList()) == []

    def test_iter_yields_data_in_order(self):
        """Iteration yields node data in insertion order."""
        ll = LinkedList()
        for value in ("tail", "tail_1", "tail_2"):
            ll.insert_tail(value)
        assert list(ll) == ["tail", "tail_1", "tail_2"]

    def test_iter_returns_iterator(self):
        """__iter__ returns a generator/iterator object."""
        ll = LinkedList()
        ll.insert_tail(1)
        iterator = iter(ll)
        assert iter(iterator) is iterator
        assert next(iterator) == 1
        with pytest.raises(StopIteration):
            next(iterator)

    @pytest.mark.parametrize(
        "values, expected_len",
        [
            ([], 0),
            ([1], 1),
            ([1, 2, 3], 3),
            (list(range(100)), 100),
        ],
    )
    def test_len(self, values, expected_len):
        """len() equals the number of nodes."""
        ll = LinkedList()
        for v in values:
            ll.insert_tail(v)
        assert len(ll) == expected_len

    def test_repr_empty(self):
        """repr of an empty list is the empty string."""
        assert repr(LinkedList()) == ""
        assert str(LinkedList()) == ""

    def test_repr_multiple(self):
        """repr joins node data with ' -> '."""
        ll = LinkedList()
        for v in (1, 3, 5):
            ll.insert_tail(v)
        assert repr(ll) == "1 -> 3 -> 5"
        assert str(ll) == "1 -> 3 -> 5"
        assert f"{ll}" == "1 -> 3 -> 5"

    def test_repr_with_node_instances(self):
        """Node instances inside the list are rendered via their repr."""
        ll = LinkedList()
        ll.insert_tail(Node(7))
        assert repr(ll) == "Node(7)"


# --------------------------------------------------------------------------- #
# insert_nth / insert_head / insert_tail
# --------------------------------------------------------------------------- #
class TestInsert:
    """Tests for insert_nth, insert_head and insert_tail."""

    def test_insert_nth_into_empty(self):
        """Inserting at index 0 into an empty list sets the head."""
        ll = LinkedList()
        ll.insert_nth(0, "a")
        assert ll.head is not None
        assert ll.head.data == "a"
        assert repr(ll) == "a"

    def test_insert_nth_middle(self):
        """Inserting in the middle keeps the surrounding order."""
        ll = LinkedList()
        for v in ("first", "second", "third"):
            ll.insert_tail(v)
        ll.insert_nth(1, "fourth")
        assert repr(ll) == "first -> fourth -> second -> third"
        ll.insert_nth(3, "fifth")
        assert repr(ll) == "first -> fourth -> second -> fifth -> third"

    def test_insert_nth_at_end(self):
        """Inserting at index == len appends to the tail."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_nth(1, 2)
        assert repr(ll) == "1 -> 2"

    @pytest.mark.parametrize("bad_index", [-1, -10, 2, 5])
    def test_insert_nth_out_of_range(self, bad_index):
        """Inserting at an invalid index raises IndexError."""
        ll = LinkedList()
        ll.insert_tail(1)
        with pytest.raises(IndexError):
            ll.insert_nth(bad_index, "x")
        assert repr(ll) == "1"

    def test_insert_nth_negative_on_empty(self):
        """Negative index on an empty list raises IndexError."""
        ll = LinkedList()
        with pytest.raises(IndexError):
            ll.insert_nth(-1, "x")

    def test_insert_head_order(self):
        """insert_head prepends, so the last inserted is first."""
        ll = LinkedList()
        for v in ("head", "head_2", "head_3"):
            ll.insert_head(v)
        assert repr(ll) == "head_3 -> head_2 -> head"

    def test_insert_tail_order(self):
        """insert_tail appends, preserving insertion order."""
        ll = LinkedList()
        for v in ("tail", "tail_2", "tail_3"):
            ll.insert_tail(v)
        assert repr(ll) == "tail -> tail_2 -> tail_3"

    def test_insert_none_and_falsy_values(self):
        """Falsy data (None, 0, '', False) is stored correctly."""
        ll = LinkedList()
        for v in (None, 0, "", False):
            ll.insert_tail(v)
        assert len(ll) == 4
        assert list(ll) == [None, 0, "", False]


# --------------------------------------------------------------------------- #
# __getitem__ / __setitem__
# --------------------------------------------------------------------------- #
class TestIndexing:
    """Tests for __getitem__ and __setitem__."""

    @pytest.fixture()
    def ll(self) -> LinkedList:
        """A list containing 0..9."""
        linked = LinkedList()
        for i in range(10):
            linked.insert_nth(i, i)
        return linked

    @pytest.mark.parametrize("index", [0, 1, 5, 9])
    def test_getitem_valid(self, ll, index):
        """Indexing returns the data at that position."""
        assert ll[index] == index

    @pytest.mark.parametrize("index", [-1, -10, 10, 100])
    def test_getitem_out_of_range(self, ll, index):
        """Out-of-range indexing raises ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = ll[index]

    def test_getitem_empty_list(self):
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = LinkedList()[0]

    @pytest.mark.parametrize("index, value", [(0, 666), (5, -666), (9, "x")])
    def test_setitem_valid(self, ll, index, value):
        """Assigning to a valid index replaces the node data."""
        ll[index] = value
        assert ll[index] == value
        assert len(ll) == 10

    @pytest.mark.parametrize("index", [-1, -10, 10])
    def test_setitem_out_of_range(self, ll, index):
        """Assigning to an invalid index raises ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            ll[index] = 666

    def test_setitem_does_not_change_length(self, ll):
        """Replacing data never changes the list length."""
        ll[3] = "changed"
        assert len(ll) == 10
        assert list(ll) == [0, 1, 2, "changed", 4, 5, 6, 7, 8, 9]


# --------------------------------------------------------------------------- #
# delete_head / delete_tail / delete_nth
# --------------------------------------------------------------------------- #
class TestDelete:
    """Tests for delete_head, delete_tail and delete_nth."""

    @pytest.fixture()
    def ll(self) -> LinkedList:
        """A list containing 'first', 'second', 'third'."""
        linked = LinkedList()
        for v in ("first", "second", "third"):
            linked.insert_tail(v)
        return linked

    def test_delete_head(self, ll):
        """delete_head removes and returns the first element."""
        assert ll.delete_head() == "first"
        assert repr(ll) == "second -> third"
        assert ll.delete_head() == "second"
        assert ll.delete_head() == "third"
        assert ll.is_empty() is True

    def test_delete_head_empty(self):
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_head()

    def test_delete_tail(self, ll):
        """delete_tail removes and returns the last element."""
        assert ll.delete_tail() == "third"
        assert repr(ll) == "first -> second"
        assert ll.delete_tail() == "second"
        assert ll.delete_tail() == "first"
        assert ll.is_empty() is True

    def test_delete_tail_empty(self):
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_tail()

    def test_delete_tail_single_element(self):
        """delete_tail on a one-element list empties the list."""
        ll = LinkedList()
        ll.insert_tail(42)
        assert ll.delete_tail() == 42
        assert ll.head is None

    def test_delete_nth_middle(self, ll):
        """delete_nth removes the node at the given index."""
        assert ll.delete_nth(1) == "second"
        assert repr(ll) == "first -> third"

    def test_delete_nth_default_index_is_zero(self, ll):
        """delete_nth defaults to index 0 (head)."""
        assert ll.delete_nth() == "first"
        assert repr(ll) == "second -> third"

    def test_delete_nth_last_index(self, ll):
        """delete_nth(len - 1) removes the tail."""
        assert ll.delete_nth(2) == "third"
        assert repr(ll) == "first -> second"

    @pytest.mark.parametrize("bad_index", [-1, -5, 3, 5])
    def test_delete_nth_out_of_range(self, ll, bad_index):
        """delete_nth with an invalid index raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_nth(bad_index)
        assert len(ll) == 3

    def test_delete_nth_empty(self):
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_nth(0)

    def test_delete_nth_returns_none_data(self):
        """delete_nth returns None when the stored data is None."""
        ll = LinkedList()
        ll.insert_tail(None)
        ll.insert_tail(1)
        assert ll.delete_nth(0) is None
        assert repr(ll) == "1"


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse."""

    def test_reverse_multiple(self):
        """reverse inverts the order of the list."""
        ll = LinkedList()
        for v in ("first", "second", "third"):
            ll.insert_tail(v)
        ll.reverse()
        assert repr(ll) == "third -> second -> first"

    def test_reverse_empty(self):
        """Reversing an empty list leaves it empty."""
        ll = LinkedList()
        ll.reverse()
        assert ll.head is None
        assert repr(ll) == ""

    def test_reverse_single(self):
        """Reversing a one-element list is a no-op."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.reverse()
        assert repr(ll) == "1"
        assert len(ll) == 1

    def test_reverse_twice_restores_order(self):
        """Reversing twice restores the original order."""
        ll = LinkedList()
        for v in range(5):
            ll.insert_tail(v)
        ll.reverse()
        ll.reverse()
        assert list(ll) == [0, 1, 2, 3, 4]

    def test_reverse_preserves_length(self):
        """reverse does not change the number of nodes."""
        ll = LinkedList()
        for v in range(7):
            ll.insert_tail(v)
        ll.reverse()
        assert len(ll) == 7


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for print_list."""

    def test_print_list_output(self, capsys):
        """print_list writes the string representation to stdout."""
        ll = LinkedList()
        for v in ("first", "second", "third"):
            ll.insert_tail(v)
        ll.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second -> third\n"

    def test_print_list_empty(self, capsys):
        """print_list on an empty list prints an empty line."""
        LinkedList().print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_print_list_uses_print(self):
        """print_list delegates to the builtin print with the list itself."""
        ll = LinkedList()
        ll.insert_tail(1)
        with patch.object(builtins, "print") as mock_print:
            ll.print_list()
        mock_print.assert_called_once_with(ll)


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helper functions."""

    def test_singly_linked_list_helper(self):
        """The bundled helper test runs without raising."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper(self):
        """The bundled mixed-type helper test runs without raising."""
        assert test_singly_linked_list_2() is None


# --------------------------------------------------------------------------- #
# Integration-style scenario
# --------------------------------------------------------------------------- #
def test_full_scenario():
    """End-to-end scenario mixing inserts, deletes, indexing and reversal."""
    ll = LinkedList()
    assert ll.is_empty() is True

    for i in range(10):
        assert len(ll) == i
        ll.insert_nth(i, i + 1)
    assert repr(ll) == " -> ".join(str(i) for i in range(1, 11))

    ll.insert_head(0)
    ll.insert_tail(11)
    assert repr(ll) == " -> ".join(str(i) for i in range(12))

    assert ll.delete_head() == 0
    assert ll.delete_nth(9) == 10
    assert ll.delete_tail() == 11
    assert len(ll) == 9
    assert repr(ll) == " -> ".join(str(i) for i in range(1, 10))

    for i in range(9):
        ll[i] = -i
    assert all(ll[i] == -i for i in range(9))

    ll.reverse()
    assert repr(ll) == " -> ".join(str(i) for i in range(-8, 1))
