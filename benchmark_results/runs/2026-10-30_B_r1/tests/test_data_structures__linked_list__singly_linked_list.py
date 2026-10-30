import pytest

from singly_linked_list import LinkedList, Node


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------
class TestNode:
    '''Tests for the Node dataclass.'''

    def test_node_creation_default_next(self):
        '''A Node created with only data should have next_node set to None.'''
        node = Node(20)
        assert node.data == 20
        assert node.next_node is None

    def test_node_creation_with_next(self):
        '''A Node can be linked to another Node via next_node.'''
        tail = Node(2)
        head = Node(1, tail)
        assert head.next_node is tail

    @pytest.mark.parametrize(
        'data,expected',
        [
            (20, 'Node(20)'),
            ('Hello, world!', 'Node(Hello, world!)'),
            (None, 'Node(None)'),
            (True, 'Node(True)'),
        ],
    )
    def test_node_repr(self, data, expected):
        '''repr() should produce the expected string for various data types.'''
        assert repr(Node(data)) == expected


# ---------------------------------------------------------------------------
# LinkedList construction / basic state
# ---------------------------------------------------------------------------
class TestLinkedListBasics:
    '''Tests for construction, emptiness, and length.'''

    def test_new_list_is_empty(self):
        '''A freshly created LinkedList should have a None head and length 0.'''
        linked_list = LinkedList()
        assert linked_list.head is None
        assert linked_list.is_empty() is True
        assert len(linked_list) == 0

    def test_is_empty_after_insert(self):
        '''is_empty should return False after inserting an element.'''
        linked_list = LinkedList()
        linked_list.insert_head('first')
        assert linked_list.is_empty() is False

    def test_len_reflects_insertions(self):
        '''len() should track the number of inserted nodes.'''
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
            assert len(linked_list) == i + 1


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------
class TestInsertion:
    '''Tests for insert_tail, insert_head, and insert_nth.'''

    def test_insert_tail_single(self):
        '''Inserting a single tail element should make it the head.'''
        linked_list = LinkedList()
        linked_list.insert_tail('tail')
        assert str(linked_list) == 'tail'

    def test_insert_tail_multiple(self):
        '''Multiple tail insertions should preserve insertion order.'''
        linked_list = LinkedList()
        linked_list.insert_tail('a')
        linked_list.insert_tail('b')
        linked_list.insert_tail('c')
        assert str(linked_list) == 'a -> b -> c'

    def test_insert_head_single(self):
        '''Inserting a single head element should make it the head.'''
        linked_list = LinkedList()
        linked_list.insert_head('head')
        assert str(linked_list) == 'head'

    def test_insert_head_multiple(self):
        '''Head insertions should prepend, reversing insertion order.'''
        linked_list = LinkedList()
        linked_list.insert_head('a')
        linked_list.insert_head('b')
        linked_list.insert_head('c')
        assert str(linked_list) == 'c -> b -> a'

    def test_insert_nth_middle(self):
        '''insert_nth should place an element at the given index.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        linked_list.insert_nth(1, 'fourth')
        assert str(linked_list) == 'first -> fourth -> second -> third'

    def test_insert_nth_at_zero(self):
        '''insert_nth(0, ...) should behave like insert_head.'''
        linked_list = LinkedList()
        linked_list.insert_tail('b')
        linked_list.insert_nth(0, 'a')
        assert str(linked_list) == 'a -> b'

    def test_insert_nth_at_end(self):
        '''insert_nth(len, ...) should behave like insert_tail.'''
        linked_list = LinkedList()
        linked_list.insert_tail('a')
        linked_list.insert_nth(1, 'b')
        assert str(linked_list) == 'a -> b'

    def test_insert_nth_into_empty(self):
        '''insert_nth(0, ...) on an empty list should create the head.'''
        linked_list = LinkedList()
        linked_list.insert_nth(0, 'only')
        assert str(linked_list) == 'only'

    @pytest.mark.parametrize('index', [-1, 2])
    def test_insert_nth_out_of_range(self, index):
        '''insert_nth with an invalid index should raise IndexError.'''
        linked_list = LinkedList()
        linked_list.insert_tail('a')
        with pytest.raises(IndexError):
            linked_list.insert_nth(index, 'bad')


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
class TestDeletion:
    '''Tests for delete_head, delete_tail, and delete_nth.'''

    def test_delete_head_returns_data(self):
        '''delete_head should return the removed node's data.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        assert linked_list.delete_head() == 'first'
        assert str(linked_list) == 'second -> third'

    def test_delete_tail_returns_data(self):
        '''delete_tail should return the removed node's data.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        assert linked_list.delete_tail() == 'third'
        assert str(linked_list) == 'first -> second'

    def test_delete_nth_middle(self):
        '''delete_nth should remove and return the middle element.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        assert linked_list.delete_nth(1) == 'second'
        assert str(linked_list) == 'first -> third'

    def test_delete_nth_default_index(self):
        '''delete_nth with no argument should delete the head.'''
        linked_list = LinkedList()
        linked_list.insert_tail('a')
        linked_list.insert_tail('b')
        assert linked_list.delete_nth() == 'a'
        assert str(linked_list) == 'b'

    def test_delete_head_empty(self):
        '''delete_head on an empty list should raise IndexError.'''
        linked_list = LinkedList()
        with pytest.raises(IndexError):
            linked_list.delete_head()

    def test_delete_tail_empty(self):
        '''delete_tail on an empty list should raise IndexError.'''
        linked_list = LinkedList()
        with pytest.raises(IndexError):
            linked_list.delete_tail()

    @pytest.mark.parametrize('index', [-1, 5])
    def test_delete_nth_out_of_range(self, index):
        '''delete_nth with an invalid index should raise IndexError.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        with pytest.raises(IndexError):
            linked_list.delete_nth(index)

    def test_delete_single_element(self):
        '''Deleting the only element should leave an empty list.'''
        linked_list = LinkedList()
        linked_list.insert_tail('only')
        assert linked_list.delete_head() == 'only'
        assert linked_list.is_empty() is True
        assert len(linked_list) == 0


# ---------------------------------------------------------------------------
# Iteration / indexing
# ---------------------------------------------------------------------------
class TestIterationAndIndexing:
    '''Tests for __iter__, __getitem__, and __setitem__.'''

    def test_iter_yields_data_in_order(self):
        '''Iteration should yield node data in list order.'''
        linked_list = LinkedList()
        for value in ('a', 'b', 'c'):
            linked_list.insert_tail(value)
        assert list(linked_list) == ['a', 'b', 'c']

    def test_iter_empty(self):
        '''Iterating an empty list should yield nothing.'''
        assert list(LinkedList()) == []

    def test_getitem(self):
        '''__getitem__ should return data at the given index.'''
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        assert linked_list[0] == 0
        assert linked_list[4] == 4

    @pytest.mark.parametrize('index', [-1, 5])
    def test_getitem_out_of_range(self, index):
        '''__getitem__ with an invalid index should raise ValueError.'''
        linked_list = LinkedList()
        for i in range(5):
            linked_list.insert_tail(i)
        with pytest.raises(ValueError):
            linked_list[index]

    def test_setitem(self):
        '''__setitem__ should update the data at the given index.'''
        linked_list = LinkedList()
        for i in range(3):
            linked_list.insert_tail(i)
        linked_list[1] = 99
        assert linked_list[1] == 99
        assert str(linked_list) == '0 -> 99 -> 2'

    @pytest.mark.parametrize('index', [-1, 3])
    def test_setitem_out_of_range(self, index):
        '''__setitem__ with an invalid index should raise ValueError.'''
        linked_list = LinkedList()
        for i in range(3):
            linked_list.insert_tail(i)
        with pytest.raises(ValueError):
            linked_list[index] = 0


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
class TestRepresentation:
    '''Tests for __repr__ / __str__.'''

    def test_repr_empty(self):
        '''An empty list should have an empty string representation.'''
        assert str(LinkedList()) == ''

    def test_repr_multiple(self):
        '''repr should join node data with ' -> '.'''
        linked_list = LinkedList()
        for value in (1, 3, 5):
            linked_list.insert_tail(value)
        assert repr(linked_list) == '1 -> 3 -> 5'
        assert str(linked_list) == '1 -> 3 -> 5'


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------
class TestReverse:
    '''Tests for the reverse method.'''

    def test_reverse_multiple(self):
        '''reverse should invert the order of the list.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        linked_list.reverse()
        assert str(linked_list) == 'third -> second -> first'

    def test_reverse_single(self):
        '''Reversing a single-element list should leave it unchanged.'''
        linked_list = LinkedList()
        linked_list.insert_tail('only')
        linked_list.reverse()
        assert str(linked_list) == 'only'

    def test_reverse_empty(self):
        '''Reversing an empty list should not raise and remain empty.'''
        linked_list = LinkedList()
        linked_list.reverse()
        assert linked_list.is_empty() is True


# ---------------------------------------------------------------------------
# print_list
# ---------------------------------------------------------------------------
class TestPrintList:
    '''Tests for the print_list method.'''

    def test_print_list(self, capsys):
        '''print_list should print the string representation to stdout.'''
        linked_list = LinkedList()
        for value in ('first', 'second', 'third'):
            linked_list.insert_tail(value)
        linked_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == 'first -> second -> third\n'

    def test_print_list_empty(self, capsys):
        '''print_list on an empty list should print an empty line.'''
        linked_list = LinkedList()
        linked_list.print_list()
        captured = capsys.readouterr()
        assert captured.out == '\n'


# ---------------------------------------------------------------------------
# Mixed data types
# ---------------------------------------------------------------------------
class TestMixedDataTypes:
    '''Tests using heterogeneous data types.'''

    def test_mixed_types_roundtrip(self):
        '''The list should preserve heterogeneous data types.'''
        values = [-9, 100, Node(77345112), 'dlrow olleH', 7, None, 12.20]
        linked_list = LinkedList()
        for value in values:
            linked_list.insert_tail(value)

        assert linked_list.is_empty() is False
        assert linked_list.delete_head() == -9
        assert linked_list.delete_tail() == 12.2
        assert linked_list.delete_nth(4) is None
        assert len(linked_list) == 4
