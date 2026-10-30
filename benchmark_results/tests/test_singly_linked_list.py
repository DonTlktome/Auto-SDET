"""
Pytest test suite for the singly_linked_list module.

Covers:
- Node class (construction, repr)
- LinkedList class:
  - __init__, is_empty, __len__, __repr__, __iter__
  - insert_head, insert_tail, insert_nth
  - delete_head, delete_tail, delete_nth
  - __getitem__, __setitem__
  - reverse
  - Edge cases: empty list operations, out-of-range indexing, single-element lists
"""

import pytest
from singly_linked_list import Node, LinkedList


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestNode:
    """Tests for the Node dataclass."""

    @pytest.mark.parametrize(
        "data, expected_repr",
        [
            (20, "Node(20)"),
            ("Hello, world!", "Node(Hello, world!)"),
            (None, "Node(None)"),
            (True, "Node(True)"),
            (3.14, "Node(3.14)"),
            ([1, 2, 3], "Node([1, 2, 3])"),
        ],
    )
    def test_node_repr(self, data, expected_repr):
        """Node.__repr__ should return the expected string representation."""
        node = Node(data)
        assert repr(node) == expected_repr

    def test_node_default_next_is_none(self):
        """A newly created Node should have next_node defaulting to None."""
        node = Node(42)
        assert node.next_node is None

    def test_node_can_link_to_another_node(self):
        """Nodes can be linked via the next_node attribute."""
        n1 = Node(1)
        n2 = Node(2)
        n1.next_node = n2
        assert n1.next_node is n2
        assert n1.next_node.data == 2


# ---------------------------------------------------------------------------
# LinkedList tests — construction and introspection
# ---------------------------------------------------------------------------

class TestLinkedListConstruction:
    """Tests for LinkedList.__init__, is_empty, __len__, __repr__, __iter__."""

    def test_new_list_is_empty(self):
        """A freshly created LinkedList should be empty."""
        ll = LinkedList()
        assert ll.is_empty() is True
        assert len(ll) == 0
        assert ll.head is None

    def test_repr_empty_list(self):
        """__repr__ on an empty list returns an empty string."""
        ll = LinkedList()
        assert repr(ll) == ""
        assert str(ll) == ""

    def test_iter_empty_list(self):
        """Iterating an empty list yields no items."""
        ll = LinkedList()
        assert list(ll) == []

    def test_len_after_inserts(self):
        """__len__ should reflect the number of inserted nodes."""
        ll = LinkedList()
        for i in range(5):
            ll.insert_tail(i)
            assert len(ll) == i + 1

    def test_is_empty_after_inserts(self):
        """is_empty should return False after elements are inserted."""
        ll = LinkedList()
        ll.insert_head(1)
        assert ll.is_empty() is False


# ---------------------------------------------------------------------------
# LinkedList tests — insert operations
# ---------------------------------------------------------------------------

class TestLinkedListInsert:
    """Tests for insert_head, insert_tail, insert_nth."""

    def test_insert_head_on_empty_list(self):
        """insert_head on an empty list sets head correctly."""
        ll = LinkedList()
        ll.insert_head("a")
        assert ll.head.data == "a"
        assert ll.head.next_node is None

    def test_insert_head_multiple(self):
        """Multiple insert_head calls prepend in reverse order."""
        ll = LinkedList()
        ll.insert_head(1)
        ll.insert_head(2)
        ll.insert_head(3)
        assert list(ll) == [3, 2, 1]

    def test_insert_tail_on_empty_list(self):
        """insert_tail on an empty list sets head correctly."""
        ll = LinkedList()
        ll.insert_tail("z")
        assert ll.head.data == "z"
        assert ll.head.next_node is None

    def test_insert_tail_multiple(self):
        """Multiple insert_tail calls append in order."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_tail(2)
        ll.insert_tail(3)
        assert list(ll) == [1, 2, 3]

    def test_insert_nth_at_zero(self):
        """insert_nth at index 0 is equivalent to insert_head."""
        ll = LinkedList()
        ll.insert_tail(2)
        ll.insert_nth(0, 1)
        assert list(ll) == [1, 2]

    def test_insert_nth_at_end(self):
        """insert_nth at len(self) is equivalent to insert_tail."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_nth(len(ll), 2)
        assert list(ll) == [1, 2]

    def test_insert_nth_in_middle(self):
        """insert_nth at an interior index places the node correctly."""
        ll = LinkedList()
        for v in [1, 2, 4, 5]:
            ll.insert_tail(v)
        ll.insert_nth(2, 3)
        assert list(ll) == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize("bad_index", [-1, 2, 100])
    def test_insert_nth_out_of_range_raises_index_error(self, bad_index):
        """insert_nth with an out-of-range index raises IndexError."""
        ll = LinkedList()
        ll.insert_tail(1)
        with pytest.raises(IndexError, match="list index out of range"):
            ll.insert_nth(bad_index, 99)


# ---------------------------------------------------------------------------
# LinkedList tests — delete operations
# ---------------------------------------------------------------------------

class TestLinkedListDelete:
    """Tests for delete_head, delete_tail, delete_nth."""

    def test_delete_head_returns_data(self):
        """delete_head returns the data of the removed head node."""
        ll = LinkedList()
        ll.insert_tail("first")
        ll.insert_tail("second")
        assert ll.delete_head() == "first"
        assert list(ll) == ["second"]

    def test_delete_head_single_element(self):
        """delete_head on a single-element list leaves it empty."""
        ll = LinkedList()
        ll.insert_tail("only")
        assert ll.delete_head() == "only"
        assert ll.is_empty()

    def test_delete_head_empty_raises_index_error(self):
        """delete_head on an empty list raises IndexError."""
        ll = LinkedList()
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_head()

    def test_delete_tail_returns_data(self):
        """delete_tail returns the data of the removed tail node."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_tail(2)
        ll.insert_tail(3)
        assert ll.delete_tail() == 3
        assert list(ll) == [1, 2]

    def test_delete_tail_single_element(self):
        """delete_tail on a single-element list leaves it empty."""
        ll = LinkedList()
        ll.insert_head("only")
        assert ll.delete_tail() == "only"
        assert ll.is_empty()

    def test_delete_tail_empty_raises_index_error(self):
        """delete_tail on an empty list raises IndexError."""
        ll = LinkedList()
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_tail()

    def test_delete_nth_head(self):
        """delete_nth(0) is equivalent to delete_head."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        assert ll.delete_nth(0) == 1
        assert list(ll) == [2, 3]

    def test_delete_nth_tail(self):
        """delete_nth(len-1) is equivalent to delete_tail."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        assert ll.delete_nth(len(ll) - 1) == 3
        assert list(ll) == [1, 2]

    def test_delete_nth_middle(self):
        """delete_nth at an interior index removes the correct node."""
        ll = LinkedList()
        for v in [1, 2, 3, 4]:
            ll.insert_tail(v)
        assert ll.delete_nth(2) == 3
        assert list(ll) == [1, 2, 4]

    @pytest.mark.parametrize("bad_index", [-1, 5, 100])
    def test_delete_nth_out_of_range_raises_index_error(self, bad_index):
        """delete_nth with an out-of-range index raises IndexError."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        with pytest.raises(IndexError, match="List index out of range."):
            ll.delete_nth(bad_index)


# ---------------------------------------------------------------------------
# LinkedList tests — indexing (get / set)
# ---------------------------------------------------------------------------

class TestLinkedListIndexing:
    """Tests for __getitem__ and __setitem__."""

    def test_getitem_positive_indices(self):
        """__getitem__ with valid positive indices returns correct data."""
        ll = LinkedList()
        for i in range(5):
            ll.insert_tail(i * 10)
        for i in range(5):
            assert ll[i] == i * 10

    def test_getitem_first_and_last(self):
        """__getitem__ works for first (0) and last (len-1) indices."""
        ll = LinkedList()
        ll.insert_tail("a")
        ll.insert_tail("b")
        ll.insert_tail("c")
        assert ll[0] == "a"
        assert ll[2] == "c"

    @pytest.mark.parametrize("bad_index", [-1, -10, 3, 100])
    def test_getitem_out_of_range_raises_value_error(self, bad_index):
        """__getitem__ with out-of-range index raises ValueError."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        with pytest.raises(ValueError, match="list index out of range."):
            _ = ll[bad_index]

    def test_setitem_changes_data(self):
        """__setitem__ updates the data at the given index."""
        ll = LinkedList()
        for i in range(5):
            ll.insert_tail(i)
        ll[2] = 999
        assert ll[2] == 999
        assert list(ll) == [0, 1, 999, 3, 4]

    def test_setitem_first_and_last(self):
        """__setitem__ works on first and last elements."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        ll[0] = "start"
        ll[2] = "end"
        assert list(ll) == ["start", 2, "end"]

    @pytest.mark.parametrize("bad_index", [-1, 3, 100])
    def test_setitem_out_of_range_raises_value_error(self, bad_index):
        """__setitem__ with out-of-range index raises ValueError."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        with pytest.raises(ValueError, match="list index out of range."):
            ll[bad_index] = 0


# ---------------------------------------------------------------------------
# LinkedList tests — reverse
# ---------------------------------------------------------------------------

class TestLinkedListReverse:
    """Tests for the reverse method."""

    def test_reverse_multiple_elements(self):
        """reverse should invert the order of nodes."""
        ll = LinkedList()
        for v in [1, 2, 3, 4, 5]:
            ll.insert_tail(v)
        ll.reverse()
        assert list(ll) == [5, 4, 3, 2, 1]

    def test_reverse_single_element(self):
        """Reversing a single-element list is a no-op."""
        ll = LinkedList()
        ll.insert_tail(42)
        ll.reverse()
        assert list(ll) == [42]
        assert ll.head.data == 42
        assert ll.head.next_node is None

    def test_reverse_empty_list(self):
        """Reversing an empty list should not raise an error."""
        ll = LinkedList()
        ll.reverse()
        assert ll.is_empty()
        assert ll.head is None

    def test_reverse_twice_restores_original(self):
        """Reversing twice yields the original order."""
        ll = LinkedList()
        for v in [1, 2, 3]:
            ll.insert_tail(v)
        ll.reverse()
        ll.reverse()
        assert list(ll) == [1, 2, 3]


# ---------------------------------------------------------------------------
# LinkedList tests — heterogeneous data types
# ---------------------------------------------------------------------------

class TestLinkedListHeterogeneous:
    """Tests with mixed data types including None and custom objects."""

    def test_mixed_types_including_none(self):
        """LinkedList should handle None and mixed types correctly."""
        ll = LinkedList()
        values = [None, 0, "hello", 3.14, Node(99), None]
        for v in values:
            ll.insert_tail(v)
        assert list(ll) == values

    def test_repr_with_none_and_node(self):
        """__repr__ should include None and Node representations."""
        ll = LinkedList()
        ll.insert_tail(None)
        ll.insert_tail(Node(7))
        assert repr(ll) == "None -> Node(7)"

    def test_delete_nth_with_none_data(self):
        """Deleting a node whose data is None returns None."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_tail(None)
        ll.insert_tail(3)
        result = ll.delete_nth(1)
        assert result is None
        assert list(ll) == [1, 3]


# ---------------------------------------------------------------------------
# LinkedList tests — iteration and string representation
# ---------------------------------------------------------------------------

class TestLinkedListIterAndRepr:
    """Tests for __iter__ and __repr__ edge cases."""

    def test_iter_yields_data_in_order(self):
        """__iter__ yields node data in insertion order."""
        ll = LinkedList()
        for v in ["a", "b", "c"]:
            ll.insert_tail(v)
        collected = []
        for item in ll:
            collected.append(item)
        assert collected == ["a", "b", "c"]

    def test_repr_join_format(self):
        """__repr__ joins data with ' -> '."""
        ll = LinkedList()
        ll.insert_tail(1)
        ll.insert_tail(2)
        ll.insert_tail(3)
        assert repr(ll) == "1 -> 2 -> 3"
        assert str(ll) == "1 -> 2 -> 3"
        assert f"{ll}" == "1 -> 2 -> 3"


# ---------------------------------------------------------------------------
# Integration / scenario tests
# ---------------------------------------------------------------------------

class TestLinkedListIntegration:
    """End-to-end scenarios combining multiple operations."""

    def test_full_lifecycle(self):
        """Insert, delete, reverse, index, and check length in sequence."""
        ll = LinkedList()
        assert ll.is_empty()

        # Build list: 0..9
        for i in range(10):
            ll.insert_nth(i, i + 1)
        assert len(ll) == 10
        assert str(ll) == " -> ".join(str(i) for i in range(1, 11))

        # Prepend and append
        ll.insert_head(0)
        ll.insert_tail(11)
        assert str(ll) == " -> ".join(str(i) for i in range(12))

        # Delete head, middle, tail
        assert ll.delete_head() == 0
        assert ll.delete_nth(9) == 10
        assert ll.delete_tail() == 11
        assert len(ll) == 9
        assert str(ll) == " -> ".join(str(i) for i in range(1, 10))

        # Indexing
        assert all(ll[i] == i + 1 for i in range(9))

        # Mutation via setitem
        for i in range(9):
            ll[i] = -i
        assert all(ll[i] == -i for i in range(9))

        # Reverse
        ll.reverse()
        assert str(ll) == " -> ".join(str(i) for i in range(-8, 1))

    def test_alternating_head_tail_inserts(self):
        """Alternating insert_head and insert_tail builds correct order."""
        ll = LinkedList()
        ll.insert_head(2)
        ll.insert_tail(3)
        ll.insert_head(1)
        ll.insert_tail(4)
        assert list(ll) == [1, 2, 3, 4]

    def test_delete_all_until_empty(self):
        """Deleting all elements one by one leaves the list empty."""
        ll = LinkedList()
        for v in [10, 20, 30]:
            ll.insert_tail(v)
        while not ll.is_empty():
            ll.delete_head()
        assert len(ll) == 0
        assert ll.is_empty()
        assert list(ll) == []