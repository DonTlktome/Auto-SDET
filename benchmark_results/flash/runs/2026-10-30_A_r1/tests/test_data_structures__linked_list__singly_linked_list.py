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
    def test_repr_various_data(self, data, expected):
        """__repr__ renders the wrapped data for many data types."""
        assert repr(Node(data)) == expected

    def test_repr_matches_str(self):
        """str() and repr() of a Node are identical."""
        node = Node(10)
        assert str(node) == repr(node) == "Node(10)"

    def test_default_next_node_is_none(self):
        """A Node created without next_node has next_node set to None."""
        assert Node("x").next_node is None

    def test_next_node_can_be_linked(self):
        """A Node can be linked to another Node via next_node."""
        tail = Node(2)
        head = Node(1, tail)
        assert head.next_node is tail
        assert head.next_node.data == 2

    def test_equality_is_dataclass_based(self):
        """Two Nodes with equal data and next_node compare equal."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestInitAndIsEmpty:
    """Tests for __init__ and is_empty."""

    def test_new_list_head_is_none(self, empty_list):
        """A new LinkedList has head set to None."""
        assert empty_list.head is None

    def test_is_empty_true_for_new_list(self, empty_list):
        """is_empty returns True for a freshly created list."""
        assert empty_list.is_empty() is True

    def test_is_empty_false_after_insert(self, empty_list):
        """is_empty returns False once a node is inserted."""
        empty_list.insert_head("first")
        assert empty_list.is_empty() is False

    def test_is_empty_true_after_removing_only_node(self, empty_list):
        """Removing the only node makes the list empty again."""
        empty_list.insert_tail(1)
        empty_list.delete_head()
        assert empty_list.is_empty() is True


# --------------------------------------------------------------------------- #
# Iteration / len / repr
# --------------------------------------------------------------------------- #
class TestIterationAndLen:
    """Tests for __iter__ and __len__."""

    def test_iter_empty_list(self, empty_list):
        """Iterating an empty list yields nothing."""
        assert list(empty_list) == []

    def test_iter_yields_data_in_order(self, sample_list):
        """Iteration yields node data in insertion order."""
        assert list(sample_list) == [1, 2, 3]

    def test_iter_is_lazy_generator(self, sample_list):
        """__iter__ returns a generator, not a materialized sequence."""
        iterator = iter(sample_list)
        assert iter(iterator) is iterator
        assert next(iterator) == 1

    def test_len_empty(self, empty_list):
        """len() of an empty list is 0."""
        assert len(empty_list) == 0

    @pytest.mark.parametrize("count", [1, 2, 5, 10])
    def test_len_after_inserts(self, count):
        """len() equals the number of inserted nodes."""
        ll = LinkedList()
        for i in range(count):
            ll.insert_tail(i)
        assert len(ll) == count

    def test_len_after_deletes(self, sample_list):
        """len() decreases correctly after deletions."""
        sample_list.delete_head()
        assert len(sample_list) == 2
        sample_list.delete_tail()
        assert len(sample_list) == 1


class TestRepr:
    """Tests for __repr__."""

    def test_repr_empty_list(self, empty_list):
        """An empty list renders as an empty string."""
        assert repr(empty_list) == ""
        assert str(empty_list) == ""

    def test_repr_single_element(self):
        """A single-element list renders without arrows."""
        ll = LinkedList()
        ll.insert_tail(1)
        assert repr(ll) == "1"

    def test_repr_multiple_elements(self, sample_list):
        """Elements are joined with ' -> '."""
        assert repr(sample_list) == "1 -> 2 -> 3"
        assert str(sample_list) == "1 -> 2 -> 3"

    def test_repr_with_mixed_types(self):
        """Non-string data is stringified in the representation."""
        ll = LinkedList()
        ll.insert_tail("a")
        ll.insert_tail(None)
        ll.insert_tail(2.5)
        assert repr(ll) == "a -> None -> 2.5"


# --------------------------------------------------------------------------- #
# __getitem__ / __setitem__
# --------------------------------------------------------------------------- #
class TestGetItem:
    """Tests for __getitem__."""

    def test_getitem_first_and_last(self, sample_list):
        """Valid indices return the corresponding data."""
        assert sample_list[0] == 1
        assert sample_list[2] == 3

    @pytest.mark.parametrize("index", [-1, -10, 3, 100])
    def test_getitem_out_of_range_raises(self, sample_list, index):
        """Out-of-range indices raise ValueError with the documented message."""
        with pytest.raises(ValueError, match=r"list index out of range\."):
            _ = sample_list[index]

    def test_getitem_on_empty_list_raises(self, empty_list):
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = empty_list[0]


class TestSetItem:
    """Tests for __setitem__."""

    def test_setitem_updates_data(self, sample_list):
        """Assigning to a valid index replaces the node data."""
        sample_list[0] = 666
        sample_list[1] = -666
        assert list(sample_list) == [666, -666, 3]

    def test_setitem_does_not_change_length(self, sample_list):
        """Assignment never changes the list length."""
        sample_list[2] = "x"
        assert len(sample_list) == 3

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_setitem_out_of_range_raises(self, sample_list, index):
        """Out-of-range assignment raises ValueError."""
        with pytest.raises(ValueError, match=r"list index out of range\."):
            sample_list[index] = 0

    def test_setitem_on_empty_list_raises(self, empty_list):
        """Assigning into an empty list raises ValueError."""
        with pytest.raises(ValueError):
            empty_list[0] = 1


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsert:
    """Tests for insert_tail, insert_head and insert_nth."""

    def test_insert_tail_into_empty(self, empty_list):
        """Inserting at the tail of an empty list sets the head."""
        empty_list.insert_tail("tail")
        assert empty_list.head is not None
        assert empty_list.head.data == "tail"
        assert repr(empty_list) == "tail"

    def test_insert_tail_appends_in_order(self, empty_list):
        """Repeated tail insertions preserve order."""
        for value in ("a", "b", "c"):
            empty_list.insert_tail(value)
        assert list(empty_list) == ["a", "b", "c"]

    def test_insert_head_prepends(self, empty_list):
        """Head insertions reverse the insertion order."""
        for value in ("head", "head_2", "head_3"):
            empty_list.insert_head(value)
        assert list(empty_list) == ["head_3", "head_2", "head"]

    def test_insert_head_into_empty(self, empty_list):
        """Inserting at the head of an empty list works."""
        empty_list.insert_head(0)
        assert list(empty_list) == [0]

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, [99, 1, 2, 3]),
            (1, [1, 99, 2, 3]),
            (2, [1, 2, 99, 3]),
            (3, [1, 2, 3, 99]),
        ],
    )
    def test_insert_nth_positions(self, sample_list, index, expected):
        """insert_nth places data at the requested index."""
        sample_list.insert_nth(index, 99)
        assert list(sample_list) == expected

    def test_insert_nth_into_empty_at_zero(self, empty_list):
        """insert_nth(0, ...) on an empty list sets the head."""
        empty_list.insert_nth(0, "first")
        assert list(empty_list) == ["first"]

    @pytest.mark.parametrize("index", [-1, -5, 4, 10])
    def test_insert_nth_invalid_index_raises(self, sample_list, index):
        """Invalid insertion indices raise IndexError."""
        with pytest.raises(IndexError, match="list index out of range"):
            sample_list.insert_nth(index, "bad")

    def test_insert_nth_does_not_mutate_input_node(self, empty_list):
        """Inserting a Node object stores it as data, not as a link."""
        node = Node(5)
        empty_list.insert_tail(node)
        assert empty_list.head.data is node
        assert empty_list.head.next_node is None


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDelete:
    """Tests for delete_head, delete_tail and delete_nth."""

    def test_delete_head_returns_first(self, sample_list):
        """delete_head returns the first element and removes it."""
        assert sample_list.delete_head() == 1
        assert list(sample_list) == [2, 3]

    def test_delete_tail_returns_last(self, sample_list):
        """delete_tail returns the last element and removes it."""
        assert sample_list.delete_tail() == 3
        assert list(sample_list) == [1, 2]

    def test_delete_head_until_empty(self):
        """Deleting heads repeatedly empties the list."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        assert [ll.delete_head() for _ in range(3)] == ["first", "second", "third"]
        assert ll.is_empty() is True

    def test_delete_tail_until_empty(self):
        """Deleting tails repeatedly empties the list."""
        ll = LinkedList()
        for value in ("first", "second", "third"):
            ll.insert_tail(value)
        assert [ll.delete_tail() for _ in range(3)] == ["third", "second", "first"]
        assert ll.is_empty() is True

    def test_delete_head_on_empty_raises(self, empty_list):
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_head()

    def test_delete_tail_on_empty_raises(self, empty_list):
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_tail()

    @pytest.mark.parametrize(
        "index, expected_value, expected_remaining",
        [
            (0, 1, [2, 3]),
            (1, 2, [1, 3]),
            (2, 3, [1, 2]),
        ],
    )
    def test_delete_nth_positions(
        self, sample_list, index, expected_value, expected_remaining
    ):
        """delete_nth removes and returns the element at the index."""
        assert sample_list.delete_nth(index) == expected_value
        assert list(sample_list) == expected_remaining

    def test_delete_nth_default_index_is_zero(self, sample_list):
        """delete_nth defaults to index 0."""
        assert sample_list.delete_nth() == 1
        assert list(sample_list) == [2, 3]

    @pytest.mark.parametrize("index", [-1, -5, 3, 5])
    def test_delete_nth_invalid_index_raises(self, sample_list, index):
        """Invalid deletion indices raise IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_nth(index)

    def test_delete_nth_on_empty_raises(self, empty_list):
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_nth(0)

    def test_delete_nth_returns_none_data(self):
        """A stored None value is returned as None (not confused with failure)."""
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

    def test_print_list_outputs_repr(self, sample_list, capsys):
        """print_list writes the string representation plus a newline."""
        sample_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "1 -> 2 -> 3\n"

    def test_print_list_empty(self, empty_list, capsys):
        """print_list on an empty list prints an empty line."""
        empty_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_print_list_uses_print_builtin(self, sample_list):
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
        """Reversing a multi-element list inverts the order."""
        sample_list.reverse()
        assert list(sample_list) == [3, 2, 1]
        assert repr(sample_list) == "3 -> 2 -> 1"

    def test_reverse_empty(self, empty_list):
        """Reversing an empty list leaves it empty."""
        empty_list.reverse()
        assert empty_list.head is None
        assert list(empty_list) == []

    def test_reverse_single_element(self):
        """Reversing a one-element list is a no-op."""
        ll = LinkedList()
        ll.insert_tail(42)
        ll.reverse()
        assert list(ll) == [42]
        assert len(ll) == 1

    def test_reverse_twice_restores_order(self, sample_list):
        """Reversing twice restores the original order."""
        sample_list.reverse()
        sample_list.reverse()
        assert list(sample_list) == [1, 2, 3]

    def test_reverse_preserves_length(self, sample_list):
        """Reversal does not change the number of nodes."""
        sample_list.reverse()
        assert len(sample_list) == 3


# --------------------------------------------------------------------------- #
# Module-level helper tests
# --------------------------------------------------------------------------- #
class TestModuleHelpers:
    """Tests for the module-level test helper functions."""

    def test_singly_linked_list_helper(self):
        """The bundled helper test executes without assertion errors."""
        assert test_singly_linked_list() is None

    def test_singly_linked_list_2_helper(self):
        """The bundled mixed-type helper test executes without errors."""
        assert test_singly_linked_list_2() is None


# --------------------------------------------------------------------------- #
# Integration-style scenario
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
        assert all(ll[i] == -i for i in range(9))

        ll.reverse()
        assert repr(ll) == " -> ".join(str(i) for i in range(-8, 1))

    def test_mixed_types_roundtrip(self):
        """Mixed data types survive insertion, deletion and reversal."""
        values = [-9, 100, Node(77345112), "dlrow olleH", 7, None, 12.20]
        ll = LinkedList()
        for value in values:
            ll.insert_tail(value)

        assert ll.is_empty() is False
        assert len(ll) == len(values)
        assert ll.delete_head() == -9
        assert ll.delete_tail() == 12.20

        ll.reverse()
        assert list(ll) == list(reversed(values[1:-1]))