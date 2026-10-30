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
        """A freshly created Node has next_node set to None."""
        node = Node(20)
        assert node.data == 20
        assert node.next_node is None

    def test_node_with_explicit_next(self) -> None:
        """A Node can be linked to another Node via next_node."""
        tail = Node("tail")
        head = Node("head", tail)
        assert head.next_node is tail
        assert head.next_node.data == "tail"

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
    def test_node_repr(self, data: object, expected: str) -> None:
        """repr()/str() of a Node renders as 'Node(<data>)'."""
        node = Node(data)
        assert repr(node) == expected
        assert str(node) == expected

    def test_node_equality(self) -> None:
        """Dataclass equality compares data and next_node."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)
        assert Node(1, Node(2)) != Node(1)


# --------------------------------------------------------------------------- #
# LinkedList construction / emptiness
# --------------------------------------------------------------------------- #
class TestLinkedListBasics:
    """Tests for construction, emptiness and length."""

    def test_new_list_is_empty(self) -> None:
        """A new LinkedList has head None, is empty and has length 0."""
        ll = LinkedList()
        assert ll.head is None
        assert ll.is_empty() is True
        assert len(ll) == 0
        assert list(ll) == []

    def test_is_empty_after_insert_and_delete(self) -> None:
        """is_empty flips to False on insert and back to True on delete."""
        ll = LinkedList()
        ll.insert_head("first")
        assert ll.is_empty() is False
        assert ll.delete_head() == "first"
        assert ll.is_empty() is True

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
    def test_len_matches_number_of_inserts(self, count: int) -> None:
        """len() equals the number of inserted elements."""
        ll = LinkedList()
        for i in range(count):
            ll.insert_tail(i)
        assert len(ll) == count

    def test_iter_yields_data_in_order(self) -> None:
        """Iteration yields node data from head to tail."""
        ll = LinkedList()
        for value in ("tail", "tail_1", "tail_2"):
            ll.insert_tail(value)
        assert list(ll) == ["tail", "tail_1", "tail_2"]

    def test_iter_on_empty_list(self) -> None:
        """Iterating an empty list yields nothing."""
        assert list(LinkedList()) == []


# --------------------------------------------------------------------------- #
# repr
# --------------------------------------------------------------------------- #
class TestRepr:
    """Tests for LinkedList.__repr__."""

    def test_repr_empty(self) -> None:
        """An empty list renders as an empty string."""
        assert repr(LinkedList()) == ""
        assert str(LinkedList()) == ""

    def test_repr_single_element(self) -> None:
        """A single element renders without arrows."""
        ll = LinkedList()
        ll.insert_tail(1)
        assert repr(ll) == "1"

    def test_repr_multiple_elements(self) -> None:
        """Multiple elements are joined with ' -> '."""
        ll = LinkedList()
        for value in (1, 3, 5):
            ll.insert_tail(value)
        assert repr(ll) == "1 -> 3 -> 5"
        assert f"{ll}" == "1 -> 3 -> 5"

    def test_repr_with_mixed_types(self) -> None:
        """Mixed types are stringified individually."""
        ll = LinkedList()
        ll.insert_tail("a")
        ll.insert_tail(None)
        ll.insert_tail(Node(7))
        assert repr(ll) == "a -> None -> Node(7)"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for LinkedList.__getitem__."""

    @pytest.fixture()
    def ll(self) -> LinkedList:
        """A list containing 0..9."""
        linked = LinkedList()
        for i in range(10):
            linked.insert_nth(i, i)
        return linked

    @pytest.mark.parametrize("index", [0, 1, 5, 9])
    def test_getitem_valid(self, ll: LinkedList, index: int) -> None:
        """Valid indices return the stored data."""
        assert ll[index] == index

    @pytest.mark.parametrize("index", [-1, -10, 10, 100])
    def test_getitem_out_of_range(self, ll: LinkedList, index: int) -> None:
        """Out-of-range indices raise ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = ll[index]

    def test_getitem_empty_list(self) -> None:
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = LinkedList()[0]


class TestSetItem:
    """Tests for LinkedList.__setitem__."""

    @pytest.fixture()
    def ll(self) -> LinkedList:
        """A list containing 0..9."""
        linked = LinkedList()
        for i in range(10):
            linked.insert_nth(i, i)
        return linked

    def test_setitem_head_and_middle(self, ll: LinkedList) -> None:
        """Assigning to valid indices updates the stored data."""
        ll[0] = 666
        ll[5] = -666
        assert ll[0] == 666
        assert ll[5] == -666
        assert len(ll) == 10

    def test_setitem_tail(self, ll: LinkedList) -> None:
        """Assigning to the last index updates the tail node."""
        ll[9] = "last"
        assert ll[9] == "last"
        assert repr(ll).endswith("-> last")

    @pytest.mark.parametrize("index", [-1, -10, 10])
    def test_setitem_out_of_range(self, ll: LinkedList, index: int) -> None:
        """Out-of-range assignment raises ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            ll[index] = 0


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsertion:
    """Tests for insert_head, insert_tail and insert_nth."""

    def test_insert_head_order(self) -> None:
        """insert_head prepends elements."""
        ll = LinkedList()
        for value in ("head", "head_2", "head_3"):
            ll.insert_head(value)
        assert repr(ll) == "head_3 -> head_2 -> head"

    def test_insert_tail_order(self) -> None:
        """insert_tail appends elements."""
        ll = LinkedList()
        for value in ("tail", "tail_2", "tail_3"):
            ll.insert_tail(value)
        assert repr(ll) == "tail -> tail_2 -> tail_3"

    def test_insert_nth_middle(self) -> None:
        """insert_nth inserts at the requested position."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.insert_nth(1, "fourth")
        assert repr(ll) == "first -> fourth -> second -> third"
        ll.insert_nth(3, "fifth")
        assert repr(ll) == "first -> fourth -> second -> fifth -> third"

    def test_insert_nth_into_empty_list(self) -> None:
        """insert_nth(0, x) on an empty list sets the head."""
        ll = LinkedList()
        ll.insert_nth(0, "only")
        assert ll.head is not None
        assert ll.head.data == "only"
        assert len(ll) == 1

    def test_insert_nth_at_end(self) -> None:
        """insert_nth(len(ll), x) appends to the tail."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_nth(1, 2)
        assert repr(ll) == "1 -> 2"

    @pytest.mark.parametrize("index", [-1, -5, 2, 5])
    def test_insert_nth_out_of_range(self, index: int) -> None:
        """insert_nth raises IndexError for invalid indices."""
        ll = LinkedList()
        ll.insert_tail("a")
        with pytest.raises(IndexError, match="list index out of range"):
            ll.insert_nth(index, "x")

    def test_insert_nth_out_of_range_empty_list(self) -> None:
        """insert_nth on an empty list rejects any index but 0."""
        ll = LinkedList()
        with pytest.raises(IndexError):
            ll.insert_nth(1, "x")


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDeletion:
    """Tests for delete_head, delete_tail and delete_nth."""

    @pytest.fixture()
    def ll(self) -> LinkedList:
        """A list containing 'first', 'second', 'third'."""
        linked = LinkedList()
        for value in ("first", "second", "third"):
            linked.insert_tail(value)
        return linked

    def test_delete_head(self, ll: LinkedList) -> None:
        """delete_head removes and returns the first element."""
        assert ll.delete_head() == "first"
        assert repr(ll) == "second -> third"
        assert ll.delete_head() == "second"
        assert ll.delete_head() == "third"
        assert ll.is_empty() is True

    def test_delete_tail(self, ll: LinkedList) -> None:
        """delete_tail removes and returns the last element."""
        assert ll.delete_tail() == "third"
        assert repr(ll) == "first -> second"
        assert ll.delete_tail() == "second"
        assert ll.delete_tail() == "first"
        assert ll.is_empty() is True

    def test_delete_nth_middle(self, ll: LinkedList) -> None:
        """delete_nth removes and returns the element at the index."""
        assert ll.delete_nth(1) == "second"
        assert repr(ll) == "first -> third"
        assert len(ll) == 2

    def test_delete_nth_default_index_is_zero(self, ll: LinkedList) -> None:
        """delete_nth defaults to index 0."""
        assert ll.delete_nth() == "first"
        assert repr(ll) == "second -> third"

    def test_delete_nth_last_index(self, ll: LinkedList) -> None:
        """delete_nth(len(ll) - 1) removes the tail."""
        assert ll.delete_nth(2) == "third"
        assert repr(ll) == "first -> second"

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_delete_nth_out_of_range(self, ll: LinkedList, index: int) -> None:
        """delete_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_nth(index)

    def test_delete_head_on_empty_list(self) -> None:
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_head()

    def test_delete_tail_on_empty_list(self) -> None:
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_tail()

    def test_delete_nth_on_empty_list(self) -> None:
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_nth(0)


# --------------------------------------------------------------------------- #
# Reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for LinkedList.reverse."""

    def test_reverse_multiple(self) -> None:
        """Reversing a multi-element list inverts the order."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.reverse()
        assert repr(ll) == "third -> second -> first"
        assert len(ll) == 3

    def test_reverse_single_element(self) -> None:
        """Reversing a one-element list leaves it unchanged."""
        ll = LinkedList()
        ll.insert_tail(42)
        ll.reverse()
        assert repr(ll) == "42"
        assert ll.head is not None
        assert ll.head.next_node is None

    def test_reverse_empty_list(self) -> None:
        """Reversing an empty list keeps it empty."""
        ll = LinkedList()
        ll.reverse()
        assert ll.is_empty() is True
        assert repr(ll) == ""

    def test_reverse_twice_restores_order(self) -> None:
        """Reversing twice restores the original order."""
        ll = LinkedList()
        for i in range(5):
            ll.insert_tail(i)
        original = repr(ll)
        ll.reverse()
        ll.reverse()
        assert repr(ll) == original


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for LinkedList.print_list."""

    def test_print_list_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_list writes the string representation plus a newline."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second -> third\n"

    def test_print_list_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_list on an empty list prints just a newline."""
        LinkedList().print_list()
        assert capsys.readouterr().out == "\n"


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helper functions."""

    def test_singly_linked_list_helper(self) -> None:
        """The bundled helper test_singly_linked_list runs without error."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper(self) -> None:
        """The bundled helper test_singly_linked_list_2 runs without error."""
        assert test_singly_linked_list_2() is None
