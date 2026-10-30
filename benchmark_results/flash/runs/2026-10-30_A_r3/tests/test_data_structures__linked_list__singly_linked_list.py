"""Pytest test suite for the singly linked list implementation.

The module under test is imported by file path so that the test file works
regardless of the package layout of the project.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by absolute path.
# ---------------------------------------------------------------------------
_SOURCE_PATH = Path(
    r"D:\Documents\Study\Projects\algorithms-bench\data_structures\linked_list\singly_linked_list.py"
)

if not _SOURCE_PATH.exists():  # pragma: no cover - fallback for relocated files
    _SOURCE_PATH = Path(__file__).resolve().parent / "singly_linked_list.py"

_spec = importlib.util.spec_from_file_location("singly_linked_list_under_test", _SOURCE_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

Node = _module.Node
LinkedList = _module.LinkedList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_list(*items):
    """Build a LinkedList containing ``items`` in the given order."""
    linked_list = LinkedList()
    for item in items:
        linked_list.insert_tail(item)
    return linked_list


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class TestNode:
    """Tests for the ``Node`` dataclass."""

    def test_node_default_next_is_none(self):
        """A freshly created node has no successor."""
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
    def test_node_repr(self, data, expected):
        """``repr``/``str`` render as ``Node(<data>)`` for various types."""
        node = Node(data)
        assert repr(node) == expected
        assert str(node) == expected
        assert node.__repr__() == expected

    def test_node_links_to_next_node(self):
        """A node can be explicitly linked to a successor node."""
        tail = Node("tail")
        head = Node("head", tail)
        assert head.next_node is tail
        assert head.next_node.data == "tail"


# ---------------------------------------------------------------------------
# Construction / emptiness / length
# ---------------------------------------------------------------------------
class TestConstruction:
    """Tests for ``__init__``, ``is_empty`` and ``__len__``."""

    def test_new_list_is_empty(self):
        """A new list has ``head is None`` and reports empty."""
        linked_list = LinkedList()
        assert linked_list.head is None
        assert linked_list.is_empty() is True
        assert len(linked_list) == 0

    def test_is_empty_false_after_insert(self):
        """Inserting a single element makes the list non-empty."""
        linked_list = LinkedList()
        linked_list.insert_head("first")
        assert linked_list.is_empty() is False
        assert len(linked_list) == 1

    def test_is_empty_true_after_removing_all(self):
        """Removing the only element restores the empty state."""
        linked_list = make_list(1)
        assert linked_list.delete_head() == 1
        assert linked_list.is_empty() is True
        assert linked_list.head is None

    @pytest.mark.parametrize("count", [0, 1, 2, 5, 50])
    def test_len_matches_number_of_insertions(self, count):
        """``len`` equals the number of inserted elements."""
        linked_list = make_list(*range(count))
        assert len(linked_list) == count


# ---------------------------------------------------------------------------
# Iteration and representation
# ---------------------------------------------------------------------------
class TestIterationAndRepr:
    """Tests for ``__iter__`` and ``__repr__``."""

    def test_iter_empty_list_yields_nothing(self):
        """Iterating an empty list produces no values."""
        assert list(LinkedList()) == []

    def test_iter_yields_data_in_order(self):
        """Iteration yields node data in insertion order."""
        linked_list = make_list("tail", "tail_1", "tail_2")
        assert list(linked_list) == ["tail", "tail_1", "tail_2"]

    def test_iter_is_lazy_generator(self):
        """``__iter__`` returns a generator, not a materialised list."""
        linked_list = make_list(1, 2)
        iterator = iter(linked_list)
        assert iter(iterator) is iterator
        assert next(iterator) == 1
        assert next(iterator) == 2
        with pytest.raises(StopIteration):
            next(iterator)

    def test_repr_empty_list(self):
        """An empty list renders as an empty string."""
        assert repr(LinkedList()) == ""
        assert str(LinkedList()) == ""

    def test_repr_multiple_items(self):
        """Items are joined with `` -> ``."""
        linked_list = make_list(1, 3, 5)
        assert repr(linked_list) == "1 -> 3 -> 5"
        assert str(linked_list) == "1 -> 3 -> 5"
        assert f"{linked_list}" == "1 -> 3 -> 5"

    def test_repr_single_item(self):
        """A single element renders without separators."""
        assert repr(make_list("only")) == "only"

    def test_repr_uses_str_of_data(self):
        """Non-string data is converted with ``str``."""
        linked_list = make_list(None, True, 2.5, Node(7))
        assert repr(linked_list) == "None -> True -> 2.5 -> Node(7)"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
class TestGetItem:
    """Tests for ``__getitem__``."""

    def test_getitem_returns_data(self):
        """Indexing returns the data stored at that position."""
        linked_list = make_list(*range(10))
        assert all(linked_list[i] == i for i in range(10))

    @pytest.mark.parametrize("index", [-1, -10, 10, 100])
    def test_getitem_out_of_range_raises_value_error(self, index):
        """Out-of-range indices raise ``ValueError``."""
        linked_list = make_list(*range(10))
        with pytest.raises(ValueError, match="list index out of range."):
            linked_list[index]

    def test_getitem_on_empty_list_raises(self):
        """Indexing an empty list raises ``ValueError``."""
        with pytest.raises(ValueError):
            LinkedList()[0]

    def test_getitem_last_valid_index(self):
        """The last valid index is ``len - 1``."""
        linked_list = make_list("a", "b", "c")
        assert linked_list[len(linked_list) - 1] == "c"


class TestSetItem:
    """Tests for ``__setitem__``."""

    def test_setitem_updates_data(self):
        """Assigning to an index replaces the stored data."""
        linked_list = make_list(*range(10))
        linked_list[0] = 666
        linked_list[5] = -666
        assert linked_list[0] == 666
        assert linked_list[5] == -666
        assert len(linked_list) == 10

    def test_setitem_does_not_change_structure(self):
        """Assignment keeps the list length and ordering intact."""
        linked_list = make_list(1, 2, 3)
        linked_list[1] = "two"
        assert list(linked_list) == [1, "two", 3]

    @pytest.mark.parametrize("index", [-1, -10, 3, 10])
    def test_setitem_out_of_range_raises_value_error(self, index):
        """Out-of-range assignment raises ``ValueError``."""
        linked_list = make_list(1, 2, 3)
        with pytest.raises(ValueError, match="list index out of range."):
            linked_list[index] = 0

    def test_setitem_on_empty_list_raises(self):
        """Assigning into an empty list raises ``ValueError``."""
        with pytest.raises(ValueError):
            LinkedList()[0] = 1


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------
class TestInsertion:
    """Tests for ``insert_head``, ``insert_tail`` and ``insert_nth``."""

    def test_insert_tail_appends(self):
        """``insert_tail`` appends to the end."""
        linked_list = LinkedList()
        linked_list.insert_tail("tail")
        assert repr(linked_list) == "tail"
        linked_list.insert_tail("tail_2")
        linked_list.insert_tail("tail_3")
        assert repr(linked_list) == "tail -> tail_2 -> tail_3"

    def test_insert_head_prepends(self):
        """``insert_head`` prepends to the front."""
        linked_list = LinkedList()
        linked_list.insert_head("head")
        linked_list.insert_head("head_2")
        linked_list.insert_head("head_3")
        assert repr(linked_list) == "head_3 -> head_2 -> head"

    def test_insert_nth_middle(self):
        """``insert_nth`` inserts between existing nodes."""
        linked_list = make_list("first", "second", "third")
        linked_list.insert_nth(1, "fourth")
        assert repr(linked_list) == "first -> fourth -> second -> third"
        linked_list.insert_nth(3, "fifth")
        assert repr(linked_list) == "first -> fourth -> second -> fifth -> third"

    def test_insert_nth_at_zero_equals_insert_head(self):
        """Index 0 behaves like ``insert_head``."""
        linked_list = make_list(2, 3)
        linked_list.insert_nth(0, 1)
        assert list(linked_list) == [1, 2, 3]

    def test_insert_nth_at_len_equals_insert_tail(self):
        """Index ``len`` behaves like ``insert_tail``."""
        linked_list = make_list(1, 2)
        linked_list.insert_nth(len(linked_list), 3)
        assert list(linked_list) == [1, 2, 3]

    def test_insert_nth_into_empty_list(self):
        """Inserting at index 0 into an empty list sets the head."""
        linked_list = LinkedList()
        linked_list.insert_nth(0, "only")
        assert linked_list.head is not None
        assert linked_list.head.data == "only"
        assert linked_list.head.next_node is None

    @pytest.mark.parametrize("index", [-1, -5, 4, 100])
    def test_insert_nth_out_of_range_raises_index_error(self, index):
        """Invalid insertion indices raise ``IndexError``."""
        linked_list = make_list(1, 2, 3)
        with pytest.raises(IndexError, match="list index out of range"):
            linked_list.insert_nth(index, "x")

    def test_insert_nth_negative_on_empty_list_raises(self):
        """A negative index on an empty list raises ``IndexError``."""
        with pytest.raises(IndexError):
            LinkedList().insert_nth(-1, "x")

    def test_insert_keeps_tail_link_terminated(self):
        """The last node always has ``next_node is None``."""
        linked_list = make_list(1, 2, 3)
        node = linked_list.head
        while node.next_node is not None:
            node = node.next_node
        assert node.data == 3
        assert node.next_node is None


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
class TestDeletion:
    """Tests for ``delete_head``, ``delete_tail`` and ``delete_nth``."""

    def test_delete_head_returns_and_removes_first(self):
        """``delete_head`` returns the first element and shrinks the list."""
        linked_list = make_list("first", "second", "third")
        assert linked_list.delete_head() == "first"
        assert repr(linked_list) == "second -> third"
        assert linked_list.delete_head() == "second"
        assert repr(linked_list) == "third"
        assert linked_list.delete_head() == "third"
        assert linked_list.is_empty() is True

    def test_delete_tail_returns_and_removes_last(self):
        """``delete_tail`` returns the last element and shrinks the list."""
        linked_list = make_list("first", "second", "third")
        assert linked_list.delete_tail() == "third"
        assert repr(linked_list) == "first -> second"
        assert linked_list.delete_tail() == "second"
        assert repr(linked_list) == "first"
        assert linked_list.delete_tail() == "first"
        assert linked_list.is_empty() is True

    def test_delete_nth_middle(self):
        """``delete_nth`` removes and returns the element at the index."""
        linked_list = make_list("first", "second", "third")
        assert linked_list.delete_nth(1) == "second"
        assert repr(linked_list) == "first -> third"

    def test_delete_nth_default_index_is_zero(self):
        """``delete_nth`` defaults to index 0."""
        linked_list = make_list(1, 2, 3)
        assert linked_list.delete_nth() == 1
        assert list(linked_list) == [2, 3]

    def test_delete_nth_last_index(self):
        """Deleting at ``len - 1`` removes the tail."""
        linked_list = make_list(1, 2, 3)
        assert linked_list.delete_nth(2) == 3
        assert list(linked_list) == [1, 2]

    @pytest.mark.parametrize("index", [-1, -10, 3, 5])
    def test_delete_nth_out_of_range_raises_index_error(self, index):
        """Invalid deletion indices raise ``IndexError``."""
        linked_list = make_list("first", "second", "third")
        with pytest.raises(IndexError, match="List index out of range."):
            linked_list.delete_nth(index)

    def test_delete_head_on_empty_list_raises(self):
        """Deleting the head of an empty list raises ``IndexError``."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_head()

    def test_delete_tail_on_empty_list_raises(self):
        """Deleting the tail of an empty list raises ``IndexError``."""
        with pytest.raises(IndexError, match="List index out of range."):
            LinkedList().delete_tail()

    def test_delete_nth_on_empty_list_raises(self):
        """Deleting any index of an empty list raises ``IndexError``."""
        with pytest.raises(IndexError):
            LinkedList().delete_nth(0)

    def test_delete_returns_none_data(self):
        """``None`` stored as data is returned as ``None``."""
        linked_list = make_list(None, 1)
        assert linked_list.delete_head() is None
        assert list(linked_list) == [1]


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------
class TestReverse:
    """Tests for ``reverse``."""

    def test_reverse_multiple_items(self):
        """Reversing reorders the list in place."""
        linked_list = make_list("first", "second", "third")
        linked_list.reverse()
        assert repr(linked_list) == "third -> second -> first"
        assert len(linked_list) == 3

    def test_reverse_empty_list(self):
        """Reversing an empty list is a no-op."""
        linked_list = LinkedList()
        linked_list.reverse()
        assert linked_list.head is None
        assert linked_list.is_empty() is True

    def test_reverse_single_item(self):
        """Reversing a one-element list leaves it unchanged."""
        linked_list = make_list(42)
        linked_list.reverse()
        assert list(linked_list) == [42]
        assert linked_list.head.next_node is None

    def test_reverse_twice_restores_original(self):
        """Reversing twice yields the original order."""
        linked_list = make_list(1, 2, 3, 4)
        linked_list.reverse()
        linked_list.reverse()
        assert list(linked_list) == [1, 2, 3, 4]

    def test_reverse_keeps_tail_terminated(self):
        """After reversing, the new tail still terminates the chain."""
        linked_list = make_list(1, 2, 3)
        linked_list.reverse()
        node = linked_list.head
        while node.next_node is not None:
            node = node.next_node
        assert node.data == 1
        assert node.next_node is None


# ---------------------------------------------------------------------------
# print_list
# ---------------------------------------------------------------------------
class TestPrintList:
    """Tests for ``print_list``."""

    def test_print_list_outputs_repr(self, capsys):
        """``print_list`` writes the string representation to stdout."""
        linked_list = make_list("first", "second", "third")
        linked_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == "first -> second -> third\n"

    def test_print_list_empty(self, capsys):
        """Printing an empty list emits a single newline."""
        LinkedList().print_list()
        captured = capsys.readouterr()
        assert captured.out == "\n"


# ---------------------------------------------------------------------------
# Integration-style scenarios
# ---------------------------------------------------------------------------
class TestIntegration:
    """End-to-end scenarios combining several operations."""

    def test_mixed_operations_sequence(self):
        """A long sequence of operations keeps the list consistent."""
        linked_list = LinkedList()
        assert linked_list.is_empty() is True
        assert str(linked_list) == ""

        for i in range(10):
            assert len(linked_list) == i
            linked_list.insert_nth(i, i + 1)
        assert str(linked_list) == " -> ".join(str(i) for i in range(1, 11))

        linked_list.insert_head(0)
        linked_list.insert_tail(11)
        assert str(linked_list) == " -> ".join(str(i) for i in range(12))

        assert linked_list.delete_head() == 0
        assert linked_list.delete_nth(9) == 10
        assert linked_list.delete_tail() == 11
        assert len(linked_list) == 9
        assert str(linked_list) == " -> ".join(str(i) for i in range(1, 10))

        assert all(linked_list[i] == i + 1 for i in range(9)) is True

        for i in range(9):
            linked_list[i] = -i
        assert all(linked_list[i] == -i for i in range(9)) is True

        linked_list.reverse()
        assert str(linked_list) == " -> ".join(str(i) for i in range(-8, 1))

    def test_mixed_data_types(self):
        """Heterogeneous data types are stored and rendered correctly."""
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
        assert (
            str(linked_list)
            == "-9 -> 100 -> Node(77345112) -> dlrow olleH -> 7 -> 5555 -> "
            "0 -> -192.55555 -> Hello, world! -> 77.9 -> Node(10) -> None -> None -> 12.2"
        )

        assert linked_list.delete_head() == -9
        assert linked_list.delete_tail() == 12.2
        assert linked_list.delete_nth(10) is None

        linked_list.insert_head(Node("Hello again, world!"))
        linked_list.insert_tail(None)
        linked_list.reverse()
        assert (
            str(linked_list)
            == "None -> None -> Node(10) -> 77.9 -> Hello, world! -> -192.55555 -> 0 -> "
            "5555 -> 7 -> dlrow olleH -> Node(77345112) -> 100 -> Node(Hello again, world!)"
        )

    def test_module_level_test_functions_pass(self):
        """The module's own self-test functions still pass."""
        _module.test_singly_linked_list()
        _module.test_singly_linked_list_2()

    def test_doctests_pass(self):
        """All doctests embedded in the module pass."""
        import doctest

        results = doctest.testmod(_module, verbose=False)
        assert results.failed == 0