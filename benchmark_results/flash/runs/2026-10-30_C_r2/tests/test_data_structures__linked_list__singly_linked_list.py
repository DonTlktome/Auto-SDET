"""Pytest test suite for singly_linked_list.py.

Covers Node, LinkedList (iteration, length, repr, indexing, insertion,
deletion, emptiness, reversal) and the module-level helper tests.
All tests are self-contained and require no network access.
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
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def empty_list() -> LinkedList:
    """Return a freshly created, empty LinkedList."""
    return LinkedList()


@pytest.fixture()
def sample_list() -> LinkedList:
    """Return a LinkedList containing 1 -> 2 -> 3."""
    ll = LinkedList()
    for value in (1, 2, 3):
        ll.insert_tail(value)
    return ll


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
class TestNode:
    """Tests for the Node dataclass."""

    def test_node_default_next_is_none(self) -> None:
        """A Node created with only data must have next_node set to None."""
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
            (3.5, "Node(3.5)"),
        ],
    )
    def test_node_repr(self, data: object, expected: str) -> None:
        """repr()/str() of a Node must be 'Node(<data>)' for various types."""
        node = Node(data)
        assert repr(node) == expected
        assert str(node) == expected

    def test_node_links_to_next_node(self) -> None:
        """A Node can be linked to another Node via next_node."""
        tail = Node("tail")
        head = Node("head", tail)
        assert head.next_node is tail
        assert head.next_node.data == "tail"

    def test_node_equality_is_dataclass_based(self) -> None:
        """Two Nodes with equal data and links compare equal (dataclass eq)."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestInitAndIsEmpty:
    """Tests for __init__ and is_empty."""

    def test_new_list_head_is_none(self, empty_list: LinkedList) -> None:
        """A new LinkedList must have head None."""
        assert empty_list.head is None

    def test_is_empty_true_on_new_list(self, empty_list: LinkedList) -> None:
        """is_empty returns True for a newly created list."""
        assert empty_list.is_empty() is True

    def test_is_empty_false_after_insert(self, empty_list: LinkedList) -> None:
        """is_empty returns False once a node is inserted."""
        empty_list.insert_head("first")
        assert empty_list.is_empty() is False

    def test_is_empty_true_after_removing_only_node(self, empty_list: LinkedList) -> None:
        """is_empty returns True again after the only node is deleted."""
        empty_list.insert_tail(1)
        empty_list.delete_head()
        assert empty_list.is_empty() is True


# --------------------------------------------------------------------------- #
# Iteration / len / repr
# --------------------------------------------------------------------------- #
class TestIterLenRepr:
    """Tests for __iter__, __len__ and __repr__."""

    def test_iter_empty(self, empty_list: LinkedList) -> None:
        """Iterating an empty list yields nothing."""
        assert list(empty_list) == []

    def test_iter_yields_data_in_order(self, sample_list: LinkedList) -> None:
        """Iteration yields node data in insertion order."""
        assert list(sample_list) == [1, 2, 3]

    def test_iter_is_lazy_generator(self, sample_list: LinkedList) -> None:
        """__iter__ returns a generator, so partial consumption is possible."""
        iterator = iter(sample_list)
        assert next(iterator) == 1
        assert next(iterator) == 2
        assert next(iterator) == 3
        with pytest.raises(StopIteration):
            next(iterator)

    @pytest.mark.parametrize(
        "values, expected_len",
        [
            ([], 0),
            ([1], 1),
            ([1, 2], 2),
            (list(range(10)), 10),
        ],
    )
    def test_len(self, values: list, expected_len: int) -> None:
        """len() equals the number of inserted nodes."""
        ll = LinkedList()
        for value in values:
            ll.insert_tail(value)
        assert len(ll) == expected_len

    def test_len_after_deletions(self, sample_list: LinkedList) -> None:
        """len() decreases correctly after deletions."""
        sample_list.delete_head()
        assert len(sample_list) == 2
        sample_list.delete_tail()
        assert len(sample_list) == 1
        sample_list.delete_nth(0)
        assert len(sample_list) == 0

    def test_repr_empty(self, empty_list: LinkedList) -> None:
        """repr of an empty list is the empty string."""
        assert repr(empty_list) == ""
        assert str(empty_list) == ""

    def test_repr_single(self, empty_list: LinkedList) -> None:
        """repr of a single-element list is just that element."""
        empty_list.insert_tail(1)
        assert repr(empty_list) == "1"

    def test_repr_multiple(self, sample_list: LinkedList) -> None:
        """repr joins elements with ' -> '."""
        assert repr(sample_list) == "1 -> 3".replace("3", "2 -> 3")
        assert str(sample_list) == "1 -> 2 -> 3"
        assert f"{sample_list}" == "1 -> 2 -> 3"

    def test_repr_with_node_instances(self, empty_list: LinkedList) -> None:
        """repr uses str() of each element, including Node instances."""
        empty_list.insert_tail(Node(10))
        empty_list.insert_tail(None)
        assert repr(empty_list) == "Node(10) -> None"


# --------------------------------------------------------------------------- #
# __getitem__ / __setitem__
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for __getitem__."""

    def test_getitem_valid_indices(self, sample_list: LinkedList) -> None:
        """Indexing returns the data at each valid position."""
        assert sample_list[0] == 1
        assert sample_list[1] == 2
        assert sample_list[2] == 3

    @pytest.mark.parametrize("index", [-1, -10, 3, 100])
    def test_getitem_out_of_range(self, sample_list: LinkedList, index: int) -> None:
        """Out-of-range indices raise ValueError with the documented message."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = sample_list[index]

    def test_getitem_empty_list(self, empty_list: LinkedList) -> None:
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = empty_list[0]

    def test_getitem_returns_node_data_not_node(self, empty_list: LinkedList) -> None:
        """__getitem__ returns the stored data, not the Node object."""
        node = Node("payload")
        empty_list.insert_tail(node)
        assert empty_list[0] is node


class TestSetItem:
    """Tests for __setitem__."""

    def test_setitem_updates_data(self, sample_list: LinkedList) -> None:
        """Assigning to a valid index replaces the node's data."""
        sample_list[0] = 666
        sample_list[2] = -666
        assert sample_list[0] == 666
        assert sample_list[2] == -666
        assert list(sample_list) == [666, 2, -666]

    @pytest.mark.parametrize("index", [-1, -10, 3, 50])
    def test_setitem_out_of_range(self, sample_list: LinkedList, index: int) -> None:
        """Assigning out of range raises ValueError and leaves list unchanged."""
        with pytest.raises(ValueError, match="list index out of range."):
            sample_list[index] = 0
        assert list(sample_list) == [1, 2, 3]

    def test_setitem_on_empty_list(self, empty_list: LinkedList) -> None:
        """Assigning on an empty list raises ValueError."""
        with pytest.raises(ValueError):
            empty_list[0] = 1


# --------------------------------------------------------------------------- #
# insert_nth / insert_head / insert_tail
# --------------------------------------------------------------------------- #
class TestInsert:
    """Tests for insert_nth, insert_head and insert_tail."""

    def test_insert_tail_into_empty(self, empty_list: LinkedList) -> None:
        """insert_tail on an empty list sets the head."""
        empty_list.insert_tail("tail")
        assert empty_list.head is not None
        assert empty_list.head.data == "tail"
        assert list(empty_list) == ["tail"]

    def test_insert_tail_appends_in_order(self, empty_list: LinkedList) -> None:
        """Repeated insert_tail appends to the end."""
        for value in ("a", "b", "c"):
            empty_list.insert_tail(value)
        assert list(empty_list) == ["a", "b", "c"]

    def test_insert_head_prepends(self, empty_list: LinkedList) -> None:
        """insert_head puts each new value at the front."""
        for value in ("head", "head_2", "head_3"):
            empty_list.insert_head(value)
        assert list(empty_list) == ["head_3", "head_2", "head"]

    def test_insert_head_into_empty(self, empty_list: LinkedList) -> None:
        """insert_head on an empty list behaves like insert_tail."""
        empty_list.insert_head(42)
        assert list(empty_list) == [42]
        assert len(empty_list) == 1

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, ["new", "first", "second", "third"]),
            (1, ["first", "new", "second", "third"]),
            (2, ["first", "second", "new", "third"]),
            (3, ["first", "second", "third", "new"]),
        ],
    )
    def test_insert_nth_positions(
        self, index: int, expected: list[str]
    ) -> None:
        """insert_nth inserts at every valid position including the end."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        ll.insert_nth(index, "new")
        assert list(ll) == expected

    def test_insert_nth_into_empty_at_zero(self, empty_list: LinkedList) -> None:
        """insert_nth(0, x) on an empty list creates the head."""
        empty_list.insert_nth(0, "only")
        assert list(empty_list) == ["only"]

    @pytest.mark.parametrize("index", [-1, -5, 4, 100])
    def test_insert_nth_invalid_index(
        self, sample_list: LinkedList, index: int
    ) -> None:
        """insert_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="list index out of range"):
            sample_list.insert_nth(index, "x")
        assert list(sample_list) == [1, 2, 3]

    def test_insert_nth_does_not_break_links(self, empty_list: LinkedList) -> None:
        """Inserting in the middle keeps the chain intact and traversable."""
        for value in range(5):
            empty_list.insert_tail(value)
        empty_list.insert_nth(2, "mid")
        assert list(empty_list) == [0, 1, "mid", 2, 3, 4]
        assert len(empty_list) == 6


# --------------------------------------------------------------------------- #
# delete_head / delete_tail / delete_nth
# --------------------------------------------------------------------------- #
class TestDelete:
    """Tests for delete_head, delete_tail and delete_nth."""

    def test_delete_head_returns_first(self, sample_list: LinkedList) -> None:
        """delete_head returns the first element and removes it."""
        assert sample_list.delete_head() == 1
        assert list(sample_list) == [2, 3]

    def test_delete_tail_returns_last(self, sample_list: LinkedList) -> None:
        """delete_tail returns the last element and removes it."""
        assert sample_list.delete_tail() == 3
        assert list(sample_list) == [1, 2]

    def test_delete_head_until_empty(self, sample_list: LinkedList) -> None:
        """Deleting repeatedly empties the list and then raises IndexError."""
        assert sample_list.delete_head() == 1
        assert sample_list.delete_head() == 2
        assert sample_list.delete_head() == 3
        assert sample_list.is_empty() is True
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_head()

    def test_delete_tail_until_empty(self, sample_list: LinkedList) -> None:
        """Deleting from the tail repeatedly empties the list."""
        assert sample_list.delete_tail() == 3
        assert sample_list.delete_tail() == 2
        assert sample_list.delete_tail() == 1
        assert sample_list.is_empty() is True
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_tail()

    def test_delete_head_on_empty(self, empty_list: LinkedList) -> None:
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_head()

    def test_delete_tail_on_empty(self, empty_list: LinkedList) -> None:
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_tail()

    def test_delete_nth_middle(self, sample_list: LinkedList) -> None:
        """delete_nth removes and returns the element at the given index."""
        assert sample_list.delete_nth(1) == 2
        assert list(sample_list) == [1, 3]

    def test_delete_nth_default_index_is_zero(self, sample_list: LinkedList) -> None:
        """delete_nth defaults to index 0."""
        assert sample_list.delete_nth() == 1
        assert list(sample_list) == [2, 3]

    def test_delete_nth_last_index(self, sample_list: LinkedList) -> None:
        """delete_nth(len - 1) removes the tail."""
        assert sample_list.delete_nth(len(sample_list) - 1) == 3
        assert list(sample_list) == [1, 2]

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_delete_nth_invalid_index(
        self, sample_list: LinkedList, index: int
    ) -> None:
        """delete_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_nth(index)
        assert list(sample_list) == [1, 2, 3]

    def test_delete_nth_on_empty(self, empty_list: LinkedList) -> None:
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_nth(0)

    def test_delete_nth_returns_none_data(self, empty_list: LinkedList) -> None:
        """A stored None value is returned as None (not confused with failure)."""
        empty_list.insert_tail(None)
        empty_list.insert_tail(1)
        assert empty_list.delete_nth(0) is None
        assert list(empty_list) == [1]


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse."""

    def test_reverse_multiple(self, sample_list: LinkedList) -> None:
        """reverse inverts the order of the list."""
        sample_list.reverse()
        assert list(sample_list) == [3, 2, 1]
        assert repr(sample_list) == "3 -> 2 -> 1"

    def test_reverse_empty(self, empty_list: LinkedList) -> None:
        """Reversing an empty list leaves it empty."""
        empty_list.reverse()
        assert empty_list.is_empty() is True
        assert list(empty_list) == []

    def test_reverse_single(self, empty_list: LinkedList) -> None:
        """Reversing a single-element list is a no-op."""
        empty_list.insert_tail("only")
        empty_list.reverse()
        assert list(empty_list) == ["only"]

    def test_reverse_twice_restores_order(self, sample_list: LinkedList) -> None:
        """Reversing twice returns the original order."""
        sample_list.reverse()
        sample_list.reverse()
        assert list(sample_list) == [1, 2, 3]

    def test_reverse_updates_head(self, sample_list: LinkedList) -> None:
        """After reverse, head points to the former tail node."""
        sample_list.reverse()
        assert sample_list.head is not None
        assert sample_list.head.data == 3
        assert sample_list.head.next_node is not None
        assert sample_list.head.next_node.data == 2


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for print_list."""

    def test_print_list_outputs_repr(self, sample_list: LinkedList, capsys) -> None:
        """print_list writes the string representation to stdout."""
        sample_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "1 -> 2 -> 3\n"

    def test_print_list_empty(self, empty_list: LinkedList, capsys) -> None:
        """print_list on an empty list prints an empty line."""
        empty_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_print_list_uses_print_builtin(self, sample_list: LinkedList) -> None:
        """print_list delegates to the builtin print (mocked here)."""
        with patch.object(builtins, "print") as mock_print:
            sample_list.print_list()
        mock_print.assert_called_once()
        assert str(mock_print.call_args.args[0]) == "1 -> 2 -> 3"


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helper functions."""

    def test_singly_linked_list_helper_runs(self) -> None:
        """The bundled integration helper executes without assertion errors."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper_runs(self) -> None:
        """The bundled mixed-type helper executes without assertion errors."""
        assert test_singly_linked_list_2() is None


# --------------------------------------------------------------------------- #
# Integration / mixed types
# --------------------------------------------------------------------------- #
class TestIntegration:
    """End-to-end behaviour across multiple operations."""

    def test_mixed_types_roundtrip(self) -> None:
        """A list of heterogeneous values survives insert/delete/reverse."""
        values = [-9, 100, Node(77345112), "dlrow olleH", None, 12.20]
        ll = LinkedList()
        for value in values:
            ll.insert_tail(value)
        assert len(ll) == len(values)
        assert ll.delete_head() == -9
        assert ll.delete_tail() == 12.20
        ll.reverse()
        assert list(ll) == [None, "dlrow olleH", Node(77345112), 100]

    def test_operations_sequence(self) -> None:
        """A longer sequence of operations keeps the list consistent."""
        ll = LinkedList()
        for i in range(10):
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

    def test_large_list_length_and_traversal(self) -> None:
        """A large list reports the correct length and traversal order."""
        ll = LinkedList()
        n = 1000
        for i in range(n):
            ll.insert_tail(i)
        assert len(ll) == n
        assert list(ll) == list(range(n))
        ll.reverse()
        assert list(ll) == list(range(n - 1, -1, -1))