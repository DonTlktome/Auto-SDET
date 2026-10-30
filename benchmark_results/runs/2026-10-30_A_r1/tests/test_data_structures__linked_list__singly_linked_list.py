import pytest

from singly_linked_list import LinkedList, Node


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------
class TestNode:
    """Tests for the Node dataclass."""

    def test_node_creation_default_next(self):
        """A Node created with only data should have next_node set to None."""
        node = Node(20)
        assert node.data == 20
        assert node.next_node is None

    def test_node_creation_with_next(self):
        """A Node can be linked to another Node via next_node."""
        tail = Node(2)
        head = Node(1, tail)
        assert head.next_node is tail

    @pytest.mark.parametrize(
        "data, expected_repr",
        [
            (20, "Node(20)"),
            ("Hello, world!", "Node(Hello, world!)"),
            (None, "Node(None)"),
            (True, "Node(True)"),
        ],
    )
    def test_node_repr(self, data, expected_repr):
        """repr() should produce the expected string for various data types."""
        assert repr(Node(data)) == expected_repr

    def test_node_str(self):
        """str() should match repr() for a Node."""
        assert str(Node(10)) == "Node(10)"


# ---------------------------------------------------------------------------
# LinkedList tests
# ---------------------------------------------------------------------------
class TestLinkedListBasics:
    """Tests for construction, iteration, length, and representation."""

    def test_init_empty(self):
        """A new LinkedList should have head set to None."""
        linked_list = LinkedList()
        assert linked_list.head is None

    def test_is_empty_true(self):
        """is_empty should return True for a fresh list."""
        assert LinkedList().is_empty() is True

    def test_is_empty_false(self):
        """is_empty should return False after inserting a node."""
        linked_list = LinkedList()
        linked_list.insert_head("first")
        assert linked_list.is_empty() is False

    def test_len_empty(self):
        """len() should be 0 for an empty list."""
        assert len(LinkedList()) == 0

    def test_len_after_inserts(self):
        """len() should reflect the number of inserted nodes."""
        linked_list = LinkedList()
        linked_list.insert_tail("a")
        linked_list.insert_head("b")
        linked_list.insert_tail("c")
        assert len(linked_list) == 3

    def test_repr_empty(self):
        """repr() of an empty list should be an empty string."""
        assert repr(LinkedList()) == ""

    def test_repr_multiple(self):
        """repr() should join node data with ' -> '."""
        linked_list = LinkedList()
        linked_list.insert_tail(1)
        linked_list.insert_tail(3)
        linked_list.insert_tail(5)
        assert repr(linked_list) == "1 -> 3 -> 5"
        assert str(linked_list) == "1 -> 3 -> 5"

    def test_iter_empty(self):
        """Iterating an empty list should yield nothing."""
        assert list(LinkedList()) == []

    def test_iter_yields_data(self):
        """Iteration should yield node data in order."""
        linked_list = LinkedList()
        for value in ("a", "b", "c"):
            linked_list.insert_tail(value)
        assert list(linked_list) == ["a", "b", "c"]


class TestInsertion:
    """Tests for insert_tail, insert_head, and insert_nth."""

    def test_insert_tail_empty(self):
        """insert_tail on an empty list should create the head."""
        linked_list = LinkedList()
        linked_list.insert_tail("tail")
        assert list(linked_list) == ["tail"]

    def test_insert_tail_multiple(self):
        """insert_tail should append nodes in order."""
        linked_list = LinkedList()
        linked_list.insert_tail("a")
        linked_list.insert_tail("b")
        linked_list.insert_tail("c")
        assert list(linked_list) == ["a", "b", "c"]

    def test_insert_head_empty(self):
        """insert_head on an empty list should create the head."""
        linked_list = LinkedList()
        linked_list.insert_head("head")
        assert list(linked_list) == ["head"]

    def test_insert_head_multiple(self):
        """insert_head should prepend nodes in reverse insertion order."""
        linked_list = LinkedList()
        linked_list.insert_head("a")
        linked_list.insert_head("b")
        linked_list.insert_head("c")
        assert list(linked_list) == ["c", "b", "a"]

    def test_insert_nth_middle(self):
        """insert_nth should insert at the correct middle position."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        linked_list.insert_nth(1, "fourth")
        assert list(linked_list) == ["first", "fourth", "second", "third"]

    def test_insert_nth_at_zero(self):
        """insert_nth(0, ...) should behave like insert_head."""
        linked_list = LinkedList()
        linked_list.insert_tail("b")
        linked_list.insert_nth(0, "a")
        assert list(linked_list) == ["a", "b"]

    def test_insert_nth_at_end(self):
        """insert_nth(len, ...) should behave like insert_tail."""
        linked_list = LinkedList()
        linked_list.insert_tail("a")
        linked_list.insert_nth(1, "b")
        assert list(linked_list) == ["a", "b"]

    @pytest.mark.parametrize("bad_index", [-1, 2])
    def test_insert_nth_out_of_range(self, bad_index):
        """insert_nth should raise IndexError for out-of-range indices."""
        linked_list = LinkedList()
        linked_list.insert_tail("a")
        with pytest.raises(IndexError):
            linked_list.insert_nth(bad_index, "x")


class TestDeletion:
    """Tests for delete_head, delete_tail, and delete_nth."""

    def test_delete_head_single(self):
        """delete_head should remove and return the only node's data."""
        linked_list = LinkedList()
        linked_list.insert_tail("first")
        assert linked_list.delete_head() == "first"
        assert linked_list.is_empty() is True

    def test_delete_head_multiple(self):
        """delete_head should remove the first node and return its data."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        assert linked_list.delete_head() == "first"
        assert list(linked_list) == ["second", "third"]

    def test_delete_head_empty(self):
        """delete_head on an empty list should raise IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_head()

    def test_delete_tail_single(self):
        """delete_tail should remove and return the only node's data."""
        linked_list = LinkedList()
        linked_list.insert_tail("first")
        assert linked_list.delete_tail() == "first"
        assert linked_list.is_empty() is True

    def test_delete_tail_multiple(self):
        """delete_tail should remove the last node and return its data."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        assert linked_list.delete_tail() == "third"
        assert list(linked_list) == ["first", "second"]

    def test_delete_tail_empty(self):
        """delete_tail on an empty list should raise IndexError."""
        with pytest.raises(IndexError):
            LinkedList().delete_tail()

    def test_delete_nth_middle(self):
        """delete_nth should remove and return the middle node's data."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        assert linked_list.delete_nth(1) == "second"
        assert list(linked_list) == ["first", "third"]

    def test_delete_nth_default_index(self):
        """delete_nth with no argument should delete the head."""
        linked_list = LinkedList()
        linked_list.insert_tail("a")
        linked_list.insert_tail("b")
        assert linked_list.delete_nth() == "a"
        assert list(linked_list) == ["b"]

    @pytest.mark.parametrize("bad_index", [-1, 5])
    def test_delete_nth_out_of_range(self, bad_index):
        """delete_nth should raise IndexError for invalid indices."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        with pytest.raises(IndexError):
            linked_list.delete_nth(bad_index)


class TestIndexing:
    """Tests for __getitem__ and __setitem__."""

    def test_getitem_valid(self):
        """__getitem__ should return data at a valid index."""
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        assert linked_list[0] == 0
        assert linked_list[4] == 4
        assert linked_list[2] == 2

    @pytest.mark.parametrize("bad_index", [-1, 5])
    def test_getitem_out_of_range(self, bad_index):
        """__getitem__ should raise ValueError for invalid indices."""
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        with pytest.raises(ValueError):
            linked_list[bad_index]

    def test_setitem_valid(self):
        """__setitem__ should update data at a valid index."""
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        linked_list[0] = 100
        linked_list[4] = 200
        assert linked_list[0] == 100
        assert linked_list[4] == 200

    @pytest.mark.parametrize("bad_index", [-1, 5])
    def test_setitem_out_of_range(self, bad_index):
        """__setitem__ should raise ValueError for invalid indices."""
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        with pytest.raises(ValueError):
            linked_list[bad_index] = 0


class TestReverse:
    """Tests for the reverse method."""

    def test_reverse_empty(self):
        """Reversing an empty list should leave it empty."""
        linked_list = LinkedList()
        linked_list.reverse()
        assert linked_list.is_empty() is True

    def test_reverse_single(self):
        """Reversing a single-node list should leave it unchanged."""
        linked_list = LinkedList()
        linked_list.insert_tail("only")
        linked_list.reverse()
        assert list(linked_list) == ["only"]

    def test_reverse_multiple(self):
        """Reversing should invert the order of nodes."""
        linked_list = LinkedList()
        for value in ("first", "second", "third"):
            linked_list.insert_tail(value)
        linked_list.reverse()
        assert list(linked_list) == ["third", "second", "first"]


class TestPrintList:
    """Tests for the print_list method."""

    def test_print_list(self, capsys):
        """print_list should print the string representation of the list."""
        linked_list = LinkedList()
        linked_list.insert_tail("first")
        linked_list.insert_tail("second")
        linked_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second\n"

    def test_print_list_empty(self, capsys):
        """print_list on an empty list should print an empty line."""
        LinkedList().print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"


class TestMixedDataTypes:
    """Tests using heterogeneous data types, mirroring the doctest suite."""

    def test_mixed_types_roundtrip(self):
        """Inserting and deleting mixed data types should preserve values."""
        test_input = [
            -9,
            100,
            Node(77345112),
            "dlrow olleH",
            7,
            5555,
            0,
            -192.55555,
            "Hello, world!",
            77.9,
            Node(10),
            None,
            None,
            12.20,
        ]
        linked_list = LinkedList()
        for item in test_input:
            linked_list.insert_tail(item)

        assert linked_list.is_empty() is False
        assert linked_list.delete_head() == -9
        assert linked_list.delete_tail() == 12.2
        assert linked_list.delete_nth(10) is None
        assert len(linked_list) == 11