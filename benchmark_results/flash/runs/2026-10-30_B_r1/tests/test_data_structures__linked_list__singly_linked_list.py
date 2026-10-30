"""Pytest test suite for singly_linked_list.py.

Covers Node, LinkedList (iteration, length, repr, indexing, insertion,
deletion, emptiness, reversal) and the module-level helper functions.
All tests are self-contained and require no network access.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from data_structures.linked_list.singly_linked_list import (
    LinkedList,
    Node,
    main,
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

    @pytest.mark.parametrize(
        "data, expected",
        [
            (20, "Node(20)"),
            ("Hello, world!", "Node(Hello, world!)"),
            (None, "Node(None)"),
            (True, "Node(True)"),
            (-192.55555, "Node(-192.55555)"),
        ],
    )
    def test_node_repr(self, data: object, expected: str) -> None:
        """repr/str of a Node renders as 'Node(<data>)'."""
        node = Node(data)
        assert repr(node) == expected
        assert str(node) == expected
        assert node.__repr__() == expected

    def test_node_links_to_next_node(self) -> None:
        """A Node can be linked to another Node via next_node."""
        tail = Node(2)
        head = Node(1, tail)
        assert head.next_node is tail
        assert head.next_node.data == 2

    def test_node_equality_is_dataclass_based(self) -> None:
        """Dataclass equality compares data and next_node."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestLinkedListBasics:
    """Tests for construction, emptiness and length."""

    def test_new_list_is_empty(self) -> None:
        """A new LinkedList has head None and is_empty() True."""
        linked_list = LinkedList()
        assert linked_list.head is None
        assert linked_list.is_empty() is True
        assert len(linked_list) == 0

    def test_is_empty_false_after_insert(self) -> None:
        """is_empty() returns False once a node is inserted."""
        linked_list = LinkedList()
        linked_list.insert_head("first")
        assert linked_list.is_empty() is False

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
    def test_len_matches_number_of_insertions(self, count: int) -> None:
        """len() equals the number of inserted elements."""
        linked_list = LinkedList()
        for i in range(count):
            linked_list.insert_tail(i)
        assert len(linked_list) == count

    def test_len_after_deletions(self) -> None:
        """len() decreases correctly after head/tail deletions."""
        linked_list = LinkedList()
        linked_list.insert_tail("tail")
        assert len(linked_list) == 1
        linked_list.insert_head("head")
        assert len(linked_list) == 2
        assert linked_list.delete_tail() == "tail"
        assert len(linked_list) == 1
        assert linked_list.delete_head() == "head"
        assert len(linked_list) == 0


# --------------------------------------------------------------------------- #
# Iteration / repr
# --------------------------------------------------------------------------- #
class TestIterAndRepr:
    """Tests for __iter__ and __repr__."""

    def test_iter_empty_list(self) -> None:
        """Iterating an empty list yields nothing."""
        assert list(LinkedList()) == []

    def test_iter_yields_data_in_order(self) -> None:
        """Iteration yields node data in insertion order."""
        linked_list = LinkedList()
        for value in ("tail", "tail_1", "tail_2"):
            linked_list.insert_tail(value)
        assert list(linked_list) == ["tail", "tail_1", "tail_2"]

    def test_iter_returns_generator(self) -> None:
        """__iter__ returns an iterator object."""
        linked_list = LinkedList()
        linked_list.insert_tail(1)
        iterator = iter(linked_list)
        assert iter(iterator) is iterator
        assert next(iterator) == 1
        with pytest.raises(StopIteration):
            next(iterator)

    def test_repr_empty_list(self) -> None:
        """An empty list renders as an empty string."""
        assert repr(LinkedList()) == ""
        assert str(LinkedList()) == ""

    def test_repr_multiple_items(self) -> None:
        """repr joins items with ' -> '."""
        linked_list = LinkedList()
        linked_list.insert_tail(1)
        linked_list.insert_tail(3)
        assert linked_list.__repr__() == "1 -> 3"
        assert repr(linked_list) == "1 -> 3"
        assert str(linked_list) == "1 -> 3"
        linked_list.insert_tail(5)
        assert f"{linked_list}" == "1 -> 3 -> 5"

    def test_repr_with_mixed_types(self) -> None:
        """repr stringifies non-string data, including Node instances."""
        linked_list = LinkedList()
        linked_list.insert_tail(Node(10))
        linked_list.insert_tail(None)
        linked_list.insert_tail(12.20)
        assert repr(linked_list) == "Node(10) -> None -> 12.2"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for __getitem__."""

    @pytest.fixture()
    def linked_list(self) -> LinkedList:
        """A list containing 0..9."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        return ll

    def test_getitem_valid_indices(self, linked_list: LinkedList) -> None:
        """Every valid index returns the matching value."""
        assert all(linked_list[i] == i for i in range(10))

    def test_getitem_first_and_last(self, linked_list: LinkedList) -> None:
        """Boundary indices 0 and len-1 are accessible."""
        assert linked_list[0] == 0
        assert linked_list[len(linked_list) - 1] == 9

    @pytest.mark.parametrize("index", [-10, -1, 10, 11, 100])
    def test_getitem_out_of_range(self, linked_list: LinkedList, index: int) -> None:
        """Out-of-range indices raise ValueError with the documented message."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = linked_list[index]

    def test_getitem_empty_list(self) -> None:
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = LinkedList()[0]


class TestSetItem:
    """Tests for __setitem__."""

    @pytest.fixture()
    def linked_list(self) -> LinkedList:
        """A list containing 0..9."""
        ll = LinkedList()
        for i in range(10):
            ll.insert_nth(i, i)
        return ll

    def test_setitem_updates_data(self, linked_list: LinkedList) -> None:
        """Assigning to a valid index replaces the node data."""
        linked_list[0] = 666
        linked_list[5] = -666
        assert linked_list[0] == 666
        assert linked_list[5] == -666
        assert len(linked_list) == 10

    def test_setitem_last_index(self, linked_list: LinkedList) -> None:
        """Assigning to the last index works."""
        linked_list[9] = "last"
        assert linked_list[9] == "last"

    @pytest.mark.parametrize("index", [-10, -1, 10, 50])
    def test_setitem_out_of_range(self, linked_list: LinkedList, index: int) -> None:
        """Out-of-range assignment raises ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            linked_list[index] = 666

    def test_setitem_does_not_change_length(self, linked_list: LinkedList) -> None:
        """Assignment never changes the list length."""
        linked_list[3] = "x"
        assert len(linked_list) == 10


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsertion:
    """Tests for insert_tail, insert_head and insert_nth."""

    def test_insert_tail_appends(self) -> None:
        """insert_tail appends to the end, preserving order."""
        linked_list = LinkedList()
        linked_list.insert_tail("tail")
        assert repr(linked_list) == "tail"
        linked_list.insert_tail("tail_2")
        assert repr(linked_list) == "tail -> tail_2"
        linked_list.insert_tail("tail_3")
        assert repr(linked_list) == "tail -> tail_2 -> tail_3"

    def test_insert_head_prepends(self) -> None:
        """insert_head prepends to the front, reversing insertion order."""
        linked_list = LinkedList()
        linked_list.insert_head("head")
        assert repr(linked_list) == "head"
        linked_list.insert_head("head_2")
        assert repr(linked_list) == "head_2 -> head"
        linked_list.insert_head("head_3")
        assert repr(linked_list) == "head_3 -> head_2 -> head"

    def test_insert_nth_middle(self) -> None:
        """insert_nth inserts between existing nodes."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        linked_list.insert_nth(1, "fourth")
        assert repr(linked_list) == "first -> fourth -> second -> third"
        linked_list.insert_nth(3, "fifth")
        assert repr(linked_list) == "first -> fourth -> second -> fifth -> third"

    def test_insert_nth_into_empty_list(self) -> None:
        """insert_nth(0, x) on an empty list sets the head."""
        linked_list = LinkedList()
        linked_list.insert_nth(0, "only")
        assert linked_list.head is not None
        assert linked_list.head.data == "only"
        assert len(linked_list) == 1

    def test_insert_nth_at_end(self) -> None:
        """insert_nth(len, x) behaves like insert_tail."""
        linked_list = LinkedList()
        linked_list.insert_tail(1)
        linked_list.insert_nth(1, 2)
        assert repr(linked_list) == "1 -> 2"

    @pytest.mark.parametrize("index", [-1, -5, 2, 100])
    def test_insert_nth_out_of_range(self, index: int) -> None:
        """insert_nth raises IndexError for invalid indices.

        The list holds a single element, so valid insertion indices are 0 and 1.
        """
        linked_list = LinkedList()
        linked_list.insert_tail(0)
        with pytest.raises(IndexError, match="list index out of range"):
            linked_list.insert_nth(index, "x")

    def test_insert_nth_out_of_range_on_empty_list(self) -> None:
        """insert_nth(1, x) on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            LinkedList().insert_nth(1, "x")

    def test_insert_none_and_node_values(self) -> None:
        """None and Node instances are stored as-is."""
        linked_list = LinkedList()
        linked_list.insert_tail(None)
        node = Node("payload")
        linked_list.insert_tail(node)
        assert linked_list[0] is None
        assert linked_list[1] is node


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDeletion:
    """Tests for delete_head, delete_tail and delete_nth."""

    @pytest.fixture()
    def linked_list(self) -> LinkedList:
        """A list containing 'first', 'second', 'third'."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        return ll

    def test_delete_head_returns_data(self, linked_list: LinkedList) -> None:
        """delete_head returns the removed data and shrinks the list."""
        assert linked_list.delete_head() == "first"
        assert repr(linked_list) == "second -> third"
        assert linked_list.delete_head() == "second"
        assert repr(linked_list) == "third"
        assert linked_list.delete_head() == "third"
        assert linked_list.is_empty() is True

    def test_delete_tail_returns_data(self, linked_list: LinkedList) -> None:
        """delete_tail returns the removed data and shrinks the list."""
        assert linked_list.delete_tail() == "third"
        assert repr(linked_list) == "first -> second"
        assert linked_list.delete_tail() == "second"
        assert repr(linked_list) == "first"
        assert linked_list.delete_tail() == "first"
        assert linked_list.is_empty() is True

    def test_delete_nth_middle(self, linked_list: LinkedList) -> None:
        """delete_nth removes a middle node and returns its data."""
        assert linked_list.delete_nth(1) == "second"
        assert repr(linked_list) == "first -> third"
        assert len(linked_list) == 2

    def test_delete_nth_default_index_is_zero(self, linked_list: LinkedList) -> None:
        """delete_nth() defaults to index 0."""
        assert linked_list.delete_nth() == "first"
        assert repr(linked_list) == "second -> third"

    def test_delete_nth_last_index(self, linked_list: LinkedList) -> None:
        """delete_nth(len-1) removes the tail node."""
        assert linked_list.delete_nth(2) == "third"
        assert repr(linked_list) == "first -> second"

    @pytest.mark.parametrize("index", [-1, -10, 3, 5, 100])
    def test_delete_nth_out_of_range(
        self, linked_list: LinkedList, index: int
    ) -> None:
        """delete_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="List index out of range."):
            linked_list.delete_nth(index)

    def test_delete_head_on_empty_list(self) -> None:
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_head()

    def test_delete_tail_on_empty_list(self) -> None:
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_tail()

    def test_delete_none_data(self) -> None:
        """Deleting a node whose data is None returns None."""
        linked_list = LinkedList()
        linked_list.insert_tail(None)
        assert linked_list.delete_head() is None
        assert linked_list.is_empty() is True


# --------------------------------------------------------------------------- #
# Reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse()."""

    def test_reverse_multiple_items(self) -> None:
        """reverse() reverses the order of the list in place."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        linked_list.reverse()
        assert repr(linked_list) == "third -> second -> first"

    def test_reverse_empty_list(self) -> None:
        """Reversing an empty list leaves it empty."""
        linked_list = LinkedList()
        linked_list.reverse()
        assert linked_list.is_empty() is True
        assert repr(linked_list) == ""

    def test_reverse_single_item(self) -> None:
        """Reversing a one-element list is a no-op."""
        linked_list = LinkedList()
        linked_list.insert_tail("only")
        linked_list.reverse()
        assert repr(linked_list) == "only"
        assert len(linked_list) == 1

    def test_reverse_twice_restores_order(self) -> None:
        """Reversing twice restores the original order."""
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        linked_list.reverse()
        linked_list.reverse()
        assert list(linked_list) == [0, 1, 2, 3, 4]

    def test_reverse_preserves_length(self) -> None:
        """reverse() does not change the number of nodes."""
        linked_list = LinkedList()
        for i in range(7):
            linked_list.insert_tail(i)
        linked_list.reverse()
        assert len(linked_list) == 7


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for print_list()."""

    def test_print_list_outputs_repr(self, capsys: pytest.CaptureFixture) -> None:
        """print_list writes the string representation plus a newline."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        linked_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second -> third\n"

    def test_print_list_empty(self, capsys: pytest.CaptureFixture) -> None:
        """print_list on an empty list prints just a newline."""
        LinkedList().print_list()
        assert capsys.readouterr().out == "\n"


# --------------------------------------------------------------------------- #
# Module-level helper functions
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helpers and main()."""

    def test_test_singly_linked_list_helper(self) -> None:
        """The bundled integration helper runs without assertion errors."""
        assert test_singly_linked_list() is None

    def test_test_singly_linked_list_2_helper(self) -> None:
        """The bundled mixed-type integration helper runs cleanly."""
        assert test_singly_linked_list_2() is None

    def test_main_with_patched_input(self) -> None:
        """main() runs end-to-end with input() and doctest.testmod patched."""
        inputs = iter(
            [
                "head1",
                "head2",
                "tail1",
                "tail2",
                "new_value",
            ]
        )
        with patch.object(builtins, "input", lambda *a, **k: next(inputs)), patch(
            "doctest.testmod", return_value=(0, 0)
        ) as mock_testmod:
            main()
        mock_testmod.assert_called_once()

    def test_main_prints_final_length(self, capsys: pytest.CaptureFixture) -> None:
        """main() reports the final list length after all operations."""
        inputs = iter(["a", "b", "c", "d", "z"])
        with patch.object(builtins, "input", lambda *a, **k: next(inputs)), patch(
            "doctest.testmod", return_value=(0, 0)
        ):
            main()
        out = capsys.readouterr().out
        # 4 inserts, 2 deletions -> 2 remaining nodes
        assert "length of linked_list is : 2" in out
        assert "Element at Position 1:" in out
