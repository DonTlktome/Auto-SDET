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
    def test_repr_returns_expected_string(self, data, expected):
        """__repr__ should render as 'Node(<data>)' for various data types."""
        assert repr(Node(data)) == expected

    def test_str_matches_repr(self):
        """str() falls back to __repr__ and must produce the same text."""
        node = Node(10)
        assert str(node) == repr(node) == "Node(10)"

    def test_next_node_defaults_to_none(self):
        """A Node created without next_node must have next_node set to None."""
        assert Node("x").next_node is None

    def test_next_node_can_be_linked(self):
        """A Node should accept another Node as its next_node."""
        tail = Node(2)
        head = Node(1, tail)
        assert head.next_node is tail
        assert head.next_node.data == 2

    def test_equality_is_value_based(self):
        """Dataclass equality compares data and next_node."""
        assert Node(1) == Node(1)
        assert Node(1) != Node(2)


# --------------------------------------------------------------------------- #
# Construction / emptiness
# --------------------------------------------------------------------------- #
class TestConstruction:
    """Tests for __init__ and is_empty."""

    def test_new_list_head_is_none(self, empty_list):
        """A new LinkedList must have head set to None."""
        assert empty_list.head is None

    def test_is_empty_true_for_new_list(self, empty_list):
        """is_empty returns True for a freshly created list."""
        assert empty_list.is_empty() is True

    def test_is_empty_false_after_insert(self, empty_list):
        """is_empty returns False once at least one element is inserted."""
        empty_list.insert_head("first")
        assert empty_list.is_empty() is False

    def test_is_empty_true_after_removing_all(self, sample_list):
        """is_empty returns True again after all elements are deleted."""
        while not sample_list.is_empty():
            sample_list.delete_head()
        assert sample_list.is_empty() is True


# --------------------------------------------------------------------------- #
# Iteration / len / repr
# --------------------------------------------------------------------------- #
class TestIterationAndRepr:
    """Tests for __iter__, __len__ and __repr__."""

    def test_iter_empty_list_yields_nothing(self, empty_list):
        """Iterating an empty list must yield no items."""
        assert list(empty_list) == []

    def test_iter_yields_data_in_order(self, sample_list):
        """Iteration yields node data in insertion order."""
        assert list(sample_list) == [1, 2, 3]

    def test_len_empty(self, empty_list):
        """len() of an empty list is 0."""
        assert len(empty_list) == 0

    @pytest.mark.parametrize("count", [1, 2, 5, 10])
    def test_len_matches_number_of_inserts(self, count):
        """len() equals the number of inserted elements."""
        ll = LinkedList()
        for i in range(count):
            ll.insert_tail(i)
        assert len(ll) == count

    def test_repr_empty_list_is_empty_string(self, empty_list):
        """repr of an empty list is the empty string."""
        assert repr(empty_list) == ""
        assert str(empty_list) == ""

    def test_repr_joins_with_arrow(self, sample_list):
        """repr joins element strings with ' -> '."""
        assert repr(sample_list) == "1 -> 2 -> 3"
        assert str(sample_list) == "1 -> 2 -> 3"

    def test_repr_single_element(self):
        """repr of a single-element list has no arrow."""
        ll = LinkedList()
        ll.insert_tail(42)
        assert repr(ll) == "42"

    def test_repr_uses_str_of_data(self):
        """repr uses str() of each element, including Node instances."""
        ll = LinkedList()
        ll.insert_tail(Node(7))
        ll.insert_tail(None)
        assert repr(ll) == "Node(7) -> None"


# --------------------------------------------------------------------------- #
# __getitem__ / __setitem__
# --------------------------------------------------------------------------- #
class TestIndexing:
    """Tests for __getitem__ and __setitem__."""

    def test_getitem_returns_data(self, sample_list):
        """Indexing returns the data stored at that position."""
        assert sample_list[0] == 1
        assert sample_list[1] == 2
        assert sample_list[2] == 3

    @pytest.mark.parametrize("index", [-1, -10, 3, 100])
    def test_getitem_out_of_range_raises_value_error(self, sample_list, index):
        """Out-of-range (including negative) indices raise ValueError."""
        with pytest.raises(ValueError, match="list index out of range."):
            _ = sample_list[index]

    def test_getitem_on_empty_list_raises(self, empty_list):
        """Indexing an empty list raises ValueError."""
        with pytest.raises(ValueError):
            _ = empty_list[0]

    def test_setitem_updates_data(self, sample_list):
        """Assigning to an index replaces the stored data."""
        sample_list[0] = 666
        sample_list[2] = -666
        assert list(sample_list) == [666, 2, -666]

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_setitem_out_of_range_raises_value_error(self, sample_list, index):
        """Assigning out of range raises ValueError and leaves list intact."""
        with pytest.raises(ValueError, match="list index out of range."):
            sample_list[index] = 0
        assert list(sample_list) == [1, 2, 3]

    def test_setitem_does_not_change_length(self, sample_list):
        """Replacing data must not alter the list length."""
        sample_list[1] = "new"
        assert len(sample_list) == 3
        assert sample_list[1] == "new"


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
class TestInsertion:
    """Tests for insert_head, insert_tail and insert_nth."""

    def test_insert_head_on_empty_list(self, empty_list):
        """Inserting at head of an empty list sets the head."""
        empty_list.insert_head("head")
        assert list(empty_list) == ["head"]
        assert empty_list.head.data == "head"

    def test_insert_head_prepends(self, sample_list):
        """insert_head puts the new element first."""
        sample_list.insert_head(0)
        assert list(sample_list) == [0, 1, 2, 3]

    def test_insert_tail_on_empty_list(self, empty_list):
        """Inserting at tail of an empty list sets the head."""
        empty_list.insert_tail("tail")
        assert list(empty_list) == ["tail"]

    def test_insert_tail_appends(self, sample_list):
        """insert_tail appends the new element at the end."""
        sample_list.insert_tail(4)
        assert list(sample_list) == [1, 2, 3, 4]

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
        """insert_nth places the element at the requested index."""
        sample_list.insert_nth(index, 99)
        assert list(sample_list) == expected

    @pytest.mark.parametrize("index", [-1, -5, 4, 10])
    def test_insert_nth_out_of_range_raises_index_error(self, sample_list, index):
        """insert_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="list index out of range"):
            sample_list.insert_nth(index, 0)
        assert list(sample_list) == [1, 2, 3]

    def test_insert_nth_into_empty_list(self, empty_list):
        """insert_nth(0, x) on an empty list works."""
        empty_list.insert_nth(0, "only")
        assert list(empty_list) == ["only"]

    def test_insert_nth_into_empty_list_invalid_index(self, empty_list):
        """insert_nth with index > 0 on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.insert_nth(1, "x")

    def test_insert_none_and_node_values(self, empty_list):
        """None and Node instances are valid payloads."""
        empty_list.insert_tail(None)
        empty_list.insert_tail(Node(5))
        assert list(empty_list) == [None, Node(5)]


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
class TestDeletion:
    """Tests for delete_head, delete_tail and delete_nth."""

    def test_delete_head_returns_first(self, sample_list):
        """delete_head returns the first element's data and removes it."""
        assert sample_list.delete_head() == 1
        assert list(sample_list) == [2, 3]

    def test_delete_head_single_element(self):
        """delete_head on a one-element list empties it."""
        ll = LinkedList()
        ll.insert_tail("only")
        assert ll.delete_head() == "only"
        assert ll.is_empty() is True

    def test_delete_head_empty_raises(self, empty_list):
        """delete_head on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_head()

    def test_delete_tail_returns_last(self, sample_list):
        """delete_tail returns the last element's data and removes it."""
        assert sample_list.delete_tail() == 3
        assert list(sample_list) == [1, 2]

    def test_delete_tail_single_element(self):
        """delete_tail on a one-element list empties it."""
        ll = LinkedList()
        ll.insert_tail("only")
        assert ll.delete_tail() == "only"
        assert ll.is_empty() is True

    def test_delete_tail_empty_raises(self, empty_list):
        """delete_tail on an empty list raises IndexError."""
        with pytest.raises(IndexError, match="List index out of range."):
            empty_list.delete_tail()

    @pytest.mark.parametrize(
        "index, expected_value, expected_list",
        [
            (0, 1, [2, 3]),
            (1, 2, [1, 3]),
            (2, 3, [1, 2]),
        ],
    )
    def test_delete_nth_valid(self, sample_list, index, expected_value, expected_list):
        """delete_nth removes and returns the element at the given index."""
        assert sample_list.delete_nth(index) == expected_value
        assert list(sample_list) == expected_list

    def test_delete_nth_default_index_is_zero(self, sample_list):
        """delete_nth defaults to index 0."""
        assert sample_list.delete_nth() == 1
        assert list(sample_list) == [2, 3]

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_delete_nth_invalid_raises(self, sample_list, index):
        """delete_nth raises IndexError for invalid indices."""
        with pytest.raises(IndexError, match="List index out of range."):
            sample_list.delete_nth(index)
        assert list(sample_list) == [1, 2, 3]

    def test_delete_nth_on_empty_list_raises(self, empty_list):
        """delete_nth on an empty list raises IndexError."""
        with pytest.raises(IndexError):
            empty_list.delete_nth(0)

    def test_delete_nth_returns_none_payload(self):
        """A stored None payload is returned as None (not confused with error)."""
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
        """print_list writes the list representation to stdout."""
        sample_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "1 -> 2 -> 3\n"

    def test_print_list_empty(self, empty_list, capsys):
        """print_list on an empty list prints an empty line."""
        empty_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_print_list_uses_builtin_print(self, sample_list):
        """print_list delegates to the builtin print (mocked here)."""
        with patch.object(builtins, "print") as mock_print:
            sample_list.print_list()
        mock_print.assert_called_once()
        assert str(mock_print.call_args.args[0]) == "1 -> 2 -> 3"


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #
class TestReverse:
    """Tests for reverse."""

    def test_reverse_empty_list(self, empty_list):
        """Reversing an empty list leaves it empty."""
        empty_list.reverse()
        assert empty_list.is_empty() is True
        assert list(empty_list) == []

    def test_reverse_single_element(self):
        """Reversing a one-element list is a no-op."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.reverse()
        assert list(ll) == [1]

    def test_reverse_multiple_elements(self, sample_list):
        """Reversing a multi-element list reverses the order."""
        sample_list.reverse()
        assert list(sample_list) == [3, 2, 1]
        assert repr(sample_list) == "3 -> 2 -> 1"

    def test_reverse_twice_restores_original(self, sample_list):
        """Reversing twice returns the original order."""
        sample_list.reverse()
        sample_list.reverse()
        assert list(sample_list) == [1, 2, 3]

    def test_reverse_preserves_length(self, sample_list):
        """Reversing does not change the length."""
        sample_list.reverse()
        assert len(sample_list) == 3

    def test_reverse_then_index(self, sample_list):
        """Indexing works correctly after a reversal."""
        sample_list.reverse()
        assert sample_list[0] == 3
        assert sample_list[2] == 1


# --------------------------------------------------------------------------- #
# Integration / mixed data types
# --------------------------------------------------------------------------- #
class TestIntegration:
    """End-to-end behaviour across multiple operations."""

    def test_mixed_data_types_roundtrip(self):
        """A list of heterogeneous values keeps order and repr formatting."""
        values = [-9, 100, Node(77345112), "dlrow olleH", 7, None, 12.20]
        ll = LinkedList()
        for value in values:
            ll.insert_tail(value)
        assert list(ll) == values
        assert repr(ll) == (
            "-9 -> 100 -> Node(77345112) -> dlrow olleH -> 7 -> None -> 12.2"
        )

    def test_insert_and_delete_sequence(self):
        """Interleaved inserts and deletes keep the list consistent."""
        ll = LinkedList()
        ll.insert_tail("first")
        ll.insert_tail("second")
        ll.insert_tail("third")
        assert ll.delete_nth(1) == "second"
        ll.insert_nth(1, "fourth")
        assert repr(ll) == "first -> fourth -> third"
        assert ll.delete_head() == "first"
        assert ll.delete_tail() == "third"
        assert list(ll) == ["fourth"]

    def test_module_level_test_singly_linked_list(self):
        """The module's own doctest-style test function passes."""
        assert test_singly_linked_list() is None

    def test_module_level_test_singly_linked_list_2(self):
        """The module's mixed-type test function passes."""
        assert test_singly_linked_list_2() is None