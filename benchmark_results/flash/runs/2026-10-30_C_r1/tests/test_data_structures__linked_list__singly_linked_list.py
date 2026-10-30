"""Pytest test suite for data_structures/linked_list/singly_linked_list.py.

The module under test is imported by file path so that the test file is
self-contained and does not depend on package layout / sys.path tweaks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Module loading
# --------------------------------------------------------------------------- #

_MODULE_FILENAME = "singly_linked_list.py"

_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "venv",
}


def _find_source() -> Path:
    """Locate ``singly_linked_list.py`` without hard-coding an absolute path."""
    here = Path(__file__).resolve().parent
    parent = here.parent

    direct_candidates = [
        here / _MODULE_FILENAME,
        here / "data_structures" / "linked_list" / _MODULE_FILENAME,
        here / "linked_list" / _MODULE_FILENAME,
        parent / "data_structures" / "linked_list" / _MODULE_FILENAME,
        parent / "linked_list" / _MODULE_FILENAME,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    # Fall back to a bounded breadth-first search of the surrounding project.
    for root in [here, *here.parents]:
        if len(root.parts) <= 1:  # never crawl the filesystem root
            continue
        matches: list[Path] = []
        queue: list[tuple[Path, int]] = [(root, 0)]
        visited = 0
        while queue:
            directory, depth = queue.pop(0)
            visited += 1
            if visited > 5000:
                break
            try:
                entries = sorted(directory.iterdir())
            except (OSError, PermissionError):
                continue
            for entry in entries:
                if entry.name in _IGNORED_DIRS:
                    continue
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    continue
                if is_dir:
                    if depth < 8:
                        queue.append((entry, depth + 1))
                elif entry.name == _MODULE_FILENAME:
                    matches.append(entry)
        if matches:
            matches.sort(
                key=lambda path: (
                    0 if "linked_list" in path.parts else 1,
                    len(path.parts),
                    str(path),
                )
            )
            return matches[0]

    raise FileNotFoundError(
        "Could not locate " + repr(_MODULE_FILENAME) + " starting from " + str(here)
    )


def _load_module():
    """Load the singly linked list module from the discovered file path."""
    source_path = _find_source()
    spec = importlib.util.spec_from_file_location(
        "singly_linked_list_under_test", source_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_module()
Node = _module.Node
LinkedList = _module.LinkedList


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_list(*values):
    """Build a LinkedList containing ``values`` in order."""
    linked_list = LinkedList()
    for value in values:
        linked_list.insert_tail(value)
    return linked_list


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


def test_node_default_next_node_is_none():
    """A freshly created Node must have next_node set to None."""
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
def test_node_repr(data, expected):
    """repr()/str() of a Node renders as ``Node(<data>)``."""
    node = Node(data)
    assert repr(node) == expected
    assert str(node) == expected
    assert node.__repr__() == expected


def test_node_links_to_next_node():
    """A Node can be linked to another Node via next_node."""
    tail = Node("tail")
    head = Node("head", tail)
    assert head.next_node is tail
    assert head.next_node.data == "tail"


def test_node_equality_is_dataclass_based():
    """Dataclass equality compares data and next_node."""
    assert Node(1) == Node(1)
    assert Node(1) != Node(2)
    assert Node(1, Node(2)) == Node(1, Node(2))


# --------------------------------------------------------------------------- #
# LinkedList.__init__ / is_empty / __len__
# --------------------------------------------------------------------------- #


def test_new_list_is_empty():
    """A new LinkedList has head None, is empty and has length 0."""
    linked_list = LinkedList()
    assert linked_list.head is None
    assert linked_list.is_empty() is True
    assert len(linked_list) == 0


def test_is_empty_false_after_insert():
    """is_empty returns False once at least one element is inserted."""
    linked_list = LinkedList()
    linked_list.insert_head("first")
    assert linked_list.is_empty() is False


def test_is_empty_true_again_after_removing_all():
    """Removing the only element makes the list empty again."""
    linked_list = make_list(1)
    assert linked_list.delete_head() == 1
    assert linked_list.is_empty() is True
    assert len(linked_list) == 0


@pytest.mark.parametrize("count", [0, 1, 2, 5, 10])
def test_len_matches_number_of_inserted_items(count):
    """len() equals the number of inserted elements."""
    linked_list = make_list(*range(count))
    assert len(linked_list) == count


# --------------------------------------------------------------------------- #
# __iter__ / __repr__
# --------------------------------------------------------------------------- #


def test_iter_yields_data_in_order():
    """Iteration yields node data from head to tail."""
    linked_list = make_list("tail", "tail_1", "tail_2")
    assert list(linked_list) == ["tail", "tail_1", "tail_2"]


def test_iter_on_empty_list_yields_nothing():
    """Iterating an empty list produces no items."""
    assert list(LinkedList()) == []


@pytest.mark.parametrize(
    "values, expected",
    [
        ((), ""),
        ((1,), "1"),
        ((1, 3), "1 -> 3"),
        ((1, 3, 5), "1 -> 3 -> 5"),
        (("a", "b"), "a -> b"),
    ],
)
def test_repr(values, expected):
    """repr()/str()/f-string join the data with ' -> '."""
    linked_list = make_list(*values)
    assert repr(linked_list) == expected
    assert str(linked_list) == expected
    assert f"{linked_list}" == expected
    assert linked_list.__repr__() == expected


# --------------------------------------------------------------------------- #
# insert_nth / insert_head / insert_tail
# --------------------------------------------------------------------------- #


def test_insert_nth_into_empty_list():
    """Inserting at index 0 of an empty list sets the head."""
    linked_list = LinkedList()
    linked_list.insert_nth(0, "only")
    assert linked_list.head is not None
    assert linked_list.head.data == "only"
    assert str(linked_list) == "only"


def test_insert_nth_at_head_middle_and_tail():
    """insert_nth places data at the requested position."""
    linked_list = make_list("first", "second", "third")
    linked_list.insert_nth(1, "fourth")
    assert str(linked_list) == "first -> fourth -> second -> third"
    linked_list.insert_nth(3, "fifth")
    assert str(linked_list) == "first -> fourth -> second -> fifth -> third"
    linked_list.insert_nth(0, "zeroth")
    assert str(linked_list) == "zeroth -> first -> fourth -> second -> fifth -> third"
    linked_list.insert_nth(len(linked_list), "last")
    assert str(linked_list).endswith("-> last")


@pytest.mark.parametrize("index", [-1, -10, 4, 5])
def test_insert_nth_invalid_index_raises_index_error(index):
    """insert_nth raises IndexError for out-of-range indices."""
    linked_list = make_list(1, 2, 3)
    assert len(linked_list) == 3  # valid indices are 0..3
    with pytest.raises(IndexError):
        linked_list.insert_nth(index, "x")
    # The list must be unchanged after a failed insert.
    assert str(linked_list) == "1 -> 2 -> 3"


def test_insert_head_prepends():
    """insert_head adds elements at the front, reversing insertion order."""
    linked_list = LinkedList()
    linked_list.insert_head("head")
    assert str(linked_list) == "head"
    linked_list.insert_head("head_2")
    assert str(linked_list) == "head_2 -> head"
    linked_list.insert_head("head_3")
    assert str(linked_list) == "head_3 -> head_2 -> head"


def test_insert_tail_appends():
    """insert_tail adds elements at the end, preserving insertion order."""
    linked_list = LinkedList()
    linked_list.insert_tail("tail")
    assert str(linked_list) == "tail"
    linked_list.insert_tail("tail_2")
    assert str(linked_list) == "tail -> tail_2"
    linked_list.insert_tail("tail_3")
    assert str(linked_list) == "tail -> tail_2 -> tail_3"


def test_insert_tail_accepts_none_and_node_objects():
    """Arbitrary objects (including None and Node) can be stored."""
    linked_list = LinkedList()
    linked_list.insert_tail(None)
    linked_list.insert_tail(Node(10))
    assert str(linked_list) == "None -> Node(10)"
    assert len(linked_list) == 2


# --------------------------------------------------------------------------- #
# __getitem__ / __setitem__
# --------------------------------------------------------------------------- #


def test_getitem_returns_data_at_index():
    """Indexing returns the data stored at that position."""
    linked_list = make_list(*range(10))
    assert all(linked_list[i] == i for i in range(10))


@pytest.mark.parametrize("index", [-1, -10, 10, 100])
def test_getitem_out_of_range_raises_value_error(index):
    """Indexing outside [0, len) raises ValueError."""
    linked_list = make_list(*range(10))
    with pytest.raises(ValueError, match="list index out of range."):
        _ = linked_list[index]


def test_getitem_on_empty_list_raises_value_error():
    """Indexing an empty list raises ValueError."""
    with pytest.raises(ValueError):
        _ = LinkedList()[0]


def test_setitem_updates_data():
    """Assigning to an index replaces the data of that node."""
    linked_list = make_list(*range(10))
    linked_list[0] = 666
    linked_list[5] = -666
    assert linked_list[0] == 666
    assert linked_list[5] == -666
    assert len(linked_list) == 10


@pytest.mark.parametrize("index", [-10, -1, 10])
def test_setitem_out_of_range_raises_value_error(index):
    """Assigning outside [0, len) raises ValueError and leaves data intact."""
    linked_list = make_list(*range(10))
    with pytest.raises(ValueError, match="list index out of range."):
        linked_list[index] = 666
    assert list(linked_list) == list(range(10))


# --------------------------------------------------------------------------- #
# delete_head / delete_tail / delete_nth
# --------------------------------------------------------------------------- #


def test_delete_head_returns_and_removes_first():
    """delete_head returns the first element and shortens the list."""
    linked_list = make_list("first", "second", "third")
    assert linked_list.delete_head() == "first"
    assert str(linked_list) == "second -> third"
    assert linked_list.delete_head() == "second"
    assert str(linked_list) == "third"
    assert linked_list.delete_head() == "third"
    assert linked_list.is_empty() is True


def test_delete_head_on_empty_list_raises_index_error():
    """delete_head on an empty list raises IndexError."""
    with pytest.raises(IndexError):
        LinkedList().delete_head()


def test_delete_tail_returns_and_removes_last():
    """delete_tail returns the last element and shortens the list."""
    linked_list = make_list("first", "second", "third")
    assert linked_list.delete_tail() == "third"
    assert str(linked_list) == "first -> second"
    assert linked_list.delete_tail() == "second"
    assert str(linked_list) == "first"
    assert linked_list.delete_tail() == "first"
    assert linked_list.is_empty() is True


def test_delete_tail_on_empty_list_raises_index_error():
    """delete_tail on an empty list raises IndexError."""
    with pytest.raises(IndexError):
        LinkedList().delete_tail()


def test_delete_nth_middle():
    """delete_nth removes and returns the element at the given index."""
    linked_list = make_list("first", "second", "third")
    assert linked_list.delete_nth(1) == "second"
    assert str(linked_list) == "first -> third"


def test_delete_nth_default_index_is_zero():
    """delete_nth defaults to index 0 (same as delete_head)."""
    linked_list = make_list("a", "b")
    assert linked_list.delete_nth() == "a"
    assert str(linked_list) == "b"


@pytest.mark.parametrize("index", [-1, 5, 100])
def test_delete_nth_invalid_index_raises_index_error(index):
    """delete_nth raises IndexError for out-of-range indices."""
    linked_list = make_list("first", "second", "third")
    with pytest.raises(IndexError, match="List index out of range."):
        linked_list.delete_nth(index)
    assert str(linked_list) == "first -> second -> third"


def test_delete_nth_on_empty_list_raises_index_error():
    """delete_nth on an empty list raises IndexError."""
    with pytest.raises(IndexError):
        LinkedList().delete_nth(0)


def test_delete_nth_returns_none_when_data_is_none():
    """A stored None value is returned as None (not confused with failure)."""
    linked_list = make_list(1, None, 3)
    assert linked_list.delete_nth(1) is None
    assert str(linked_list) == "1 -> 3"


# --------------------------------------------------------------------------- #
# reverse
# --------------------------------------------------------------------------- #


def test_reverse_reverses_order():
    """reverse() reverses the list in place."""
    linked_list = make_list("first", "second", "third")
    linked_list.reverse()
    assert str(linked_list) == "third -> second -> first"
    assert list(linked_list) == ["third", "second", "first"]


@pytest.mark.parametrize("values", [(), (1,), (1, 2), (1, 2, 3, 4)])
def test_reverse_edge_cases(values):
    """reverse() handles empty, single-element and even-length lists."""
    linked_list = make_list(*values)
    linked_list.reverse()
    assert list(linked_list) == list(reversed(values))
    assert len(linked_list) == len(values)


def test_reverse_twice_restores_original():
    """Reversing twice yields the original order."""
    linked_list = make_list(1, 2, 3, 4, 5)
    linked_list.reverse()
    linked_list.reverse()
    assert list(linked_list) == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# print_list
# --------------------------------------------------------------------------- #


def test_print_list_writes_repr_to_stdout(capsys):
    """print_list prints the string representation of the list."""
    linked_list = make_list("first", "second", "third")
    linked_list.print_list()
    captured = capsys.readouterr()
    assert captured.out == "first -> second -> third\n"


def test_print_list_on_empty_list_prints_blank_line(capsys):
    """print_list on an empty list prints an empty line."""
    LinkedList().print_list()
    captured = capsys.readouterr()
    assert captured.out == "\n"


# --------------------------------------------------------------------------- #
# Integration / mixed-type behaviour
# --------------------------------------------------------------------------- #


def test_mixed_types_round_trip():
    """A list holding mixed types keeps order through all operations."""
    values = [-9, 100, Node(77345112), "dlrow olleH", 7, 5555, 0, -192.55555, None]
    linked_list = make_list(*values)
    assert linked_list.is_empty() is False
    assert len(linked_list) == len(values)
    assert str(linked_list) == (
        "-9 -> 100 -> Node(77345112) -> dlrow olleH -> 7 -> 5555 -> 0 -> "
        "-192.55555 -> None"
    )
    assert linked_list.delete_head() == -9
    assert linked_list.delete_tail() is None
    linked_list.reverse()
    assert list(linked_list) == list(reversed(values[1:-1]))


def test_full_lifecycle_sequence():
    """Exercise a long sequence of operations and verify the final state."""
    linked_list = LinkedList()
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

    for i in range(9):
        linked_list[i] = -i
    assert all(linked_list[i] == -i for i in range(9))

    linked_list.reverse()
    assert str(linked_list) == " -> ".join(str(i) for i in range(-8, 1))


def test_module_doctests_pass():
    """The module's own doctests must pass (no network / stdin required)."""
    import doctest

    results = doctest.testmod(_module, verbose=False)
    assert results.failed == 0
