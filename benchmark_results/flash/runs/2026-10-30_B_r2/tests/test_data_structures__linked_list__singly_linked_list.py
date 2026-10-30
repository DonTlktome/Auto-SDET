"""Pytest test suite for singly_linked_list.py.

Covers Node, LinkedList (iteration, length, repr, indexing, insertion,
deletion, emptiness, reversal) plus the module-level helper tests.
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
        """__repr__ renders the wrapped data inside 'Node(...)'."""
        assert repr(Node(data)) == expected
        assert str(Node(data)) == expected

    def test_next_node_defaults_to_none(self):
        """A new Node has no successor by default."""
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


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestInitAndIsEmpty:
    """Tests for __init__ and is_empty."""

    def test_new_list_head_is_none(self, empty_list):
        """A new list has head set to None."""
        assert empty_list.head is None

    def test_is_empty_true_for_new_list(self, empty_list):
        """is_empty returns True for a fresh list."""
        assert empty_list.is_empty() is True

    def test_is_empty_false_after_insert(self, empty_list):
        """is_empty returns False once a node is inserted."""
        empty_list.insert_head("first")
        assert empty_list.is_empty() is False

    def test_is_empty_true_after_all_deletions(self, sample_list):
        """is_empty returns True again after every node is removed."""
        while not sample_list.is_empty():
            sample_list.delete_head()
        assert sample_list.is_empty() is True
        assert len(sample_list) == 0


# --------------------------------------------------------------------------- #
# Iteration / len / repr
# --------------------------------------------------------------------------- #
class TestIterLenRepr:
    """Tests for __iter__, __len__ and __repr__."""

    def test_iter_empty(self, empty_list):
        """Iterating an empty list yields nothing."""
        assert list(empty_list) == []

    def test_iter_order(self, sample_list):
        """Iteration yields data in insertion order."""
        assert list(sample_list) == [1, 2, 3]

    def test_iter_is_lazy_generator(self, sample_list):
        """__iter__ returns a generator, not a materialised list."""
        iterator = iter(sample_list)
        assert iter(iterator) is iterator
        assert next(iterator) == 1

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
    def test_len(self, count):
        """len equals the number of inserted nodes."""
        ll = LinkedList()
        for i in range(count):
            ll.insert_tail(i)
        assert len(ll) == count

    def test_repr_empty(self, empty_list):
        """repr of an empty list is the empty string."""
        assert repr(empty_list) == ""
        assert str(empty_list) == ""

    def test_repr_single(self):
        """repr of a one-element list is just that element."""
        ll = LinkedList()
        ll.insert_tail(1)
        assert repr(ll) == "1"

    def test_repr_multiple(self, sample_list):
        """repr joins elements with ' -> '."""
        assert repr(sample_list) == "1 -> 2 -> 3"
        assert f"{sample_list}" == "1 -> 2 -> 3"

    def test_repr_mixed_types(self):
        """repr stringifies non-string data."""
        ll = LinkedList()
        ll.insert_tail("a")
        ll.insert_tail(None)
        ll.insert_tail(2.5)
        assert repr(ll) == "a -> None -> 2.5"


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for __getitem__."""

    @pytest.mark.parametrize("index, expected", [(0, 1), (1, 2), (2, 3)])
    def test_getitem_valid(self, sample_list, index, expected):
        """Valid indices return the stored data."""
        assert sample_list[index] == expected

    @pytest.mark.parametrize("index", [-1, -10, 3, 100])
    def test_getitem_out_of_range(self, sample_list, index):
        """Out-of-range indices raise ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = sample_list[index]

    def test_getitem_empty_list(self, empty_list):
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = empty_list[0]


class TestSetItem:
    """Tests for __setitem__."""

    def test_setitem_first(self, sample_list):
        """Assigning index 0 replaces the head data."""
        sample_list[0] = 666
        assert sample_list[0] == 666
        assert list(sample_list) == [666, 2, 3]

    def test_setitem_middle(self, sample_list):
        """Assigning a middle index replaces only that node."""
        sample_list[1] = -666
        assert list(sample_list) == [1, -666, 3]

    def test_setitem_last(self, sample_list):
        """Assigning the last index replaces the tail data."""
        sample_list[2] = "tail"
        assert list(sample_list) == [1, 2, "tail"]

    @pytest.mark.parametrize("index", [-1, -10, 3, 50])
    def test_setitem_out_of_range(self, sample_list, index):
        """Out-of-range assignment raises ValueError and mutates nothing."""
        with pytest.raises(ValueError, match="list index out of range."):
            sample_list[index] = 999
        assert list(sample_list) == [1, 2, 3]

    def test_setitem_empty_list(self, empty_list):
        """Assigning into an empty list raises ValueError."""
        with pytest.raises(ValueError):
            empty_list[0] = 1


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsert:
    """Tests for insert_head, insert_tail and insert_nth."""

    def test_insert_head_into_empty(self, empty_list):
        """Inserting at head of an empty list sets the head."""
        empty_list.insert_head("head")
        assert list(empty_list) == ["head"]
        assert empty_list.head.data == "head"

    def test_insert_head_prepends(self, sample_list):
        """insert_head prepends data in LIFO order."""
        sample_list.insert_head(0)
        sample_list.insert_head(-1)
        assert list(sample_list) == [-1, 0, 1, 2, 3]

    def test_insert_tail_into_empty(self, empty_list):
        """Inserting at tail of an empty list sets the head."""
        empty_list.insert_tail("tail")
        assert list(empty_list) == ["tail"]

    def test_insert_tail_appends(self, sample_list):
        """insert_tail appends data in FIFO order."""
        sample_list.insert_tail(4)
        sample_list.insert_tail(5)
        assert list(sample_list) == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, [9, 1, 2, 3]),
            (1, [1, 9, 2, 3]),
            (2, [1, 2, 9, 3]),
            (3, [1, 2, 3, 9]),
        ],
    )
    def test_insert_nth_valid(self, sample_list, index, expected):
        """insert_nth places data at the requested position."""
        sample_list.insert_nth(index, 9)
        assert list(sample_list) == expected

    def test_insert_nth_at_len_appends(self, sample_list):
        """Index equal to len is allowed and appends."""
        sample_list.insert_nth(len(sample_list), 4)
        assert list(sample_list) == [1, 2, 3, 4]

    @pytest.mark.parametrize("index", [-1, -5, 4, 100])
    def test_insert_nth_out_of_range(self, sample_list, index):
        """Invalid indices raise IndexError and leave the list unchanged."""
        with pytest.raises(IndexError, match="list index out of range"):
            sample_list.insert_nth(index, 9)
        assert list(sample_list) == [1, 2, 3]

    def test_insert_nth_into_empty_at_zero(self, empty_list):
        """Inserting at index 0 into an empty list works."""
        empty_list.insert_nth(0, "x")
        assert list(empty_list) == ["x"]

    def test_insert_nth_into_empty_at_nonzero(self, empty_list):
        """Inserting at index > 0 into an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.insert_nth(1, "x")

    def test_insert_none_and_node_objects(self, empty_list):
        """None and Node instances are stored as plain data."""
        empty_list.insert_tail(None)
        empty_list.insert_tail(Node(7))
        assert list(empty_list) == [None, Node(7)]
        assert repr(empty_list) == "None -> Node(7)"


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDelete:
    """Tests for delete_head, delete_tail and delete_nth."""

    def test_delete_head_returns_data(self, sample_list):
        """delete_head returns the removed head data."""
        assert sample_list.delete_head() == 1
        assert list(sample_list) == [2, 3]

    def test_delete_head_until_empty(self, sample_list):
        """Repeated delete_head drains the list then raises IndexError."""
        assert sample_list.delete_head() == 1
        assert sample_list.delete_head() == 2
        assert sample_list.delete_head() == 3
        assert sample_list.is_empty() is True
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_head()

    def test_delete_head_empty(self, empty_list):
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_head()

    def test_delete_tail_returns_data(self, sample_list):
        """delete_tail returns the removed tail data."""
        assert sample_list.delete_tail() == 3
        assert list(sample_list) == [1, 2]

    def test_delete_tail_single_element(self):
        """delete_tail on a one-element list empties it."""
        ll = LinkedList()
        ll.insert_tail("only")
        assert ll.delete_tail() == "only"
        assert ll.is_empty() is True

    def test_delete_tail_empty(self, empty_list):
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_tail()

    @pytest.mark.parametrize(
        "index, removed, remaining",
        [
            (0, 1, [2, 3]),
            (1, 2, [1, 3]),
            (2, 3, [1, 2]),
        ],
    )
    def test_delete_nth_valid(self, sample_list, index, removed, remaining):
        """delete_nth removes and returns the node at the index."""
        assert sample_list.delete_nth(index) == removed
        assert list(sample_list) == remaining

    def test_delete_nth_default_index_is_zero(self, sample_list):
        """delete_nth defaults to index 0."""
        assert sample_list.delete_nth() == 1
        assert list(sample_list) == [2, 3]

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_delete_nth_out_of_range(self, sample_list, index):
        """Invalid indices raise IndexError and leave the list unchanged."""
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_nth(index)
        assert list(sample_list) == [1, 2, 3]

    def test_delete_nth_empty(self, empty_list):
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_nth(0)

    def test_delete_nth_returns_none_data(self):
        """A stored None is returned as None (not confused with failure)."""
        ll = LinkedList()
        ll.insert_tail(None)
        ll.insert_tail(1)
        assert ll.delete_nth(0) is None
        assert list(ll) == [1]


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #
class TestPrintList:
    """Tests for print_list."""

    def test_print_list_output(self, sample_list, capsys):
        """print_list writes the repr of the list to stdout."""
        sample_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "1 -> 2 -> 3\n"

    def test_print_list_empty(self, empty_list, capsys):
        """print_list on an empty list prints a blank line."""
        empty_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_print_list_uses_builtin_print(self, sample_list):
        """print_list delegates to the builtin print with the list itself."""
        with patch.object(builtins, "print") as mock_print:
            sample_list.print_list()
        mock_print.assert_called_once_with(sample_list)


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse."""

    def test_reverse_multiple(self, sample_list):
        """reverse inverts the order of the nodes."""
        sample_list.reverse()
        assert list(sample_list) == [3, 2, 1]
        assert repr(sample_list) == "3 -> 2 -> 1"

    def test_reverse_empty(self, empty_list):
        """Reversing an empty list is a no-op."""
        empty_list.reverse()
        assert empty_list.head is None
        assert list(empty_list) == []

    def test_reverse_single(self):
        """Reversing a one-element list keeps it unchanged."""
        ll = LinkedList()
        ll.insert_tail(42)
        ll.reverse()
        assert list(ll) == [42]

    def test_reverse_twice_restores_order(self, sample_list):
        """Reversing twice restores the original order."""
        sample_list.reverse()
        sample_list.reverse()
        assert list(sample_list) == [1, 2, 3]

    def test_reverse_preserves_length(self, sample_list):
        """reverse does not change the length."""
        before = len(sample_list)
        sample_list.reverse()
        assert len(sample_list) == before


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module's own test helper functions."""

    def test_singly_linked_list_helper(self):
        """The bundled integration test passes without raising."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper(self):
        """The bundled mixed-type integration test passes."""
        assert test_singly_linked_list_2() is None


# --------------------------------------------------------------------------- #
# Integration / regression
# --------------------------------------------------------------------------- #
class TestIntegration:
    """End-to-end scenarios combining several operations."""

    def test_full_lifecycle(self):
        """A sequence of mixed operations keeps the list consistent."""
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
        assert [ll[i] for i in range(9)] == [-i for i in range(9)]

        ll.reverse()
        assert repr(ll) == " -> ".join(str(i) for i in range(-8, 1))

    def test_head_pointer_after_operations(self, sample_list):
        """head always points at the first element's node."""
        assert sample_list.head.data == 1
        sample_list.insert_head(0)
        assert sample_list.head.data == 0
        sample_list.delete_head()
        assert sample_list.head.data == 1
        sample_list.reverse()
        assert sample_list.head.data == 3