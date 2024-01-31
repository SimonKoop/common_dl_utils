""" 
A module for working with trees.
Here, a tree is understood to be a composition of standard container types in Python: dicts, lists, and tuples (and UserDicts) containing leaves (anything else).

For example: the nested config dictionaries in config_realization and config_creation.
"""
from typing import Union, Any, get_args
from collections.abc import Callable, Mapping

__all__ = [
    "NonLeafTree",
    "Tree",
    "GeneralIndex",
    "get_from_index",
    "has_index",
    "tree_map",
    "get_leaves",
    "get_first_leaf",
    "tree_any",
    "tree_all",
    "repeated_tree_map",
    "get_items_of_interest",
]

NonLeafTree = Union[tuple, list, Mapping]
Tree = Union[NonLeafTree, Any]
GeneralIndex = Union[int, str, list[Union[int, str]], tuple[Union[int, str]]]
NonLeafTree.__doc__ = """
A Tree that is not a leaf. I.e. a composition of lists, tuples, and dicts (or other Mappings) containing leaves (anything else).
Example {1: [2, 3, 4], 5: {6: 7, 8: 9}} is a NonLeafTree
"""
Tree.__doc__ = """
Either a leaf (anything) or a composition of lists, tuples, and dicts (or other Mappings) containing leaves.
"""
GeneralIndex.__doc__ = """
A (multi) index into a NonLeafTree.
A regular index is assumed to be an int or str
A multi-index is assumed to be a list or tuple of regular indices
E.g. 1, 'a', [1, 'a'], and (1, 'a') may all be valid GeneralIndices
E.g. in the tree [1, {'a':2, 'b':(1, 2, 3)}], the index (1, 'b', 2) points to the leaf 3
"""

def get_from_index(tree:NonLeafTree, index: GeneralIndex):
    """ 
    Get the value at index from tree
    :param tree: a composition of tuples, lists, and Mappings
    :param index: 
        either an index of the root of the tree, 
        or a tuple/list of indices (i0, i1, ...)
            where i0 is an index of the root of the tree, 
            i1 is an index of tree[i0], etc.
    NB this function may fail without warning if the tree is a leaf such as a string
    """
    try:
        if isinstance(index, (tuple, list)):
            if len(index) == 1:
                return tree[index[0]]
            return get_from_index(tree[index[0]], index[1:])
        return tree[index]
    except IndexError:
        raise IndexError(f"{index} not in {tree}")
    
def has_index(tree:NonLeafTree, index: GeneralIndex):
    """ 
    Check if index is a valid path in tree
    NB this function may fail without warning if the tree is a leaf such as a string
    or if tree is e.g. a UserList with different indexing conventions from list 
    """
    # first maybe split index into first index (i0) and rest of index (ir)
    if isinstance(index, (tuple, list)):
        if not index:
            raise ValueError("empty multi-index")
        if len(index) == 1:
            i0 = index[0]
            ir = None 
        else:
            i0, ir = index[0], index[1:]
    else:
        i0 = index 
        ir = None 

    # first logic for mapping type tree
    if isinstance(tree, Mapping):
        if i0 not in tree:
            return False 
        elif ir is None:
            return True 
        else:
            return has_index(tree[i0], ir)
    # assume tree is like tuple and list
    if not isinstance(i0, int):
        raise TypeError(f"index {i0} is not an int while tree {tree} is not a mapping")
    if i0 < 0 or i0 >= len(tree):
        return False 
    elif ir is None:
        return True 
    else:
        return has_index(tree[i0], ir)
    
def tree_map(tree:Tree, func: Callable, is_leaf: Callable=lambda x: False):
    """ 
    Apply func to every leaf of tree
    :param tree: a composition of Sequences and Mappings
    :param func: function to be mapped over the leaves of tree
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves. 

    example use case:
    >>> scores = {'group_1': {'Maud': [10, 12, 11], 'Puck': [11, 14, 9, 8, 10], 'Astrid': [9, 8, 13, 14]}, 'group_2': {'Ilze': [12, 12, 13], 'Suzan': [10, 11]}}
    >>> num_attempts = tree_map(scores, len, is_leaf=lambda x: isinstance(x, list)) 
    num_attempts now is {'group_1': {'Maud': 3, 'Puck': 5, 'Astrid': 4}, 'group_2': {'Ilze': 3, 'Suzan': 2}}
    """
    if is_leaf(tree):
        # allows for manual override of what is supposed to be a leaf
        # e.g. if you want lists to be considered leaves
        return func(tree)
    # elif isinstance(tree, UserDict):
    #     return type(tree)(tree_map(tree.data, func, is_leaf=is_leaf))
    # elif isinstance(tree, dict):
    #     return {
    #         key: tree_map(sub_tree, func, is_leaf=is_leaf)
    #         for key, sub_tree in tree.items()
    #     }
    elif isinstance(tree, Mapping):
        return type(tree)(
            {
                key: tree_map(sub_tree, func, is_leaf=is_leaf)
                for key, sub_tree in tree.items()
            }
        )
    elif isinstance(tree, (tuple, list)):
        return type(tree)(tree_map(sub_tree, func, is_leaf=is_leaf) for sub_tree in tree)
    else:
        # we assume that if tree is none of the above, it is a leaf.
        return func(tree)
    
def get_leaves(tree: Tree, is_leaf: Callable=lambda x: False):
    """ 
    Get a generator of all leaves in tree
    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves. 
    """
    if is_leaf(tree):
        yield tree
    elif isinstance(tree, Mapping):
        for sub_tree in tree.values():
            yield from get_leaves(sub_tree, is_leaf=is_leaf)
    elif isinstance(tree, (tuple, list)):
        for sub_tree in tree:
            yield from get_leaves(sub_tree, is_leaf=is_leaf)
    else:
        # assume tree is a leaf
        yield tree


def get_first_leaf(tree: Tree, is_leaf: Callable=lambda x: False):
    """ 
    Get the first leaf in tree
    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves.
    
    NB if the tree contains mappings with unordered keys, the result of this function may be non-deterministic
    """
    if is_leaf(tree):
        return tree
    elif isinstance(tree, Mapping):
        return get_first_leaf(
            tree[next(iter(tree.keys()))],
            is_leaf=is_leaf
        )
    elif isinstance(tree, (tuple, list)):
        return get_first_leaf(tree[0], is_leaf=is_leaf)
    else:
        return tree
    
def tree_any(tree: Tree, func: Callable=bool, is_leaf: Callable=lambda x: False)-> bool:
    """ 
    See if func applied to the leaves of tree returns any True values

    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param func: function to be mapped over the leaves of tree
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves. 
    """
    return any(map(func, get_leaves(tree, is_leaf=is_leaf)))

def tree_all(tree: Tree, func: Callable=bool, is_leaf: Callable=lambda x: False)-> bool:
    """ 
    See if func applied to the leaves of tree returns only True values

    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param func: function to be mapped over the leaves of tree
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves. 
    """

    return all(map(func, get_leaves(tree, is_leaf=is_leaf)))

def repeated_tree_map(
    tree: Tree,
    func: Callable,
    requires_repetition: Callable,
    is_leaf: Callable=(lambda x: False),
    )-> Tree:
    """ 
    Repeatedly apply func to the values of tree until the termination condition set by requires_repetition is met
    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param func: function to be mapped over the leaves of tree
    :param requires_repetition: function to be mapped over the leaves of the result of tree_map.
        if evaluated to a truthy value for any of the leaves, tree map is applied again
    :param is_leaf: callable that can manually indicate whether something is a leaf. 
        E.G. used to indicate that all lists in tree should be considered leaves.

    For example, in config_creation, the values stored in _VariableToken objects may be subtrees that themselves contain _VariableToken objects
    To get a realization of a config containing `_VariableToken`s, we must repeatedly replace the `_VariableToken`s with their values 
    until we have a tree containing no `_VariableToken`s
    """
    ret_val = tree_map(
        tree,
        func,
        is_leaf=is_leaf
    )
    while tree_any(ret_val, func=requires_repetition):
        ret_val = tree_map(
            ret_val,
            func,
            is_leaf=is_leaf
        )
    return ret_val

def get_items_of_interest(tree:NonLeafTree, is_value_of_interest: Callable):
    """ 
    Get all (multi)indices and values of subtrees of tree for which is_value_of_interest returns True
    :param tree: a composition of lists, tuples, dicts, and UserDicts
    :param is_value_of_interest: callable indicating whether or not the index of the current subtree should be returned
    :yields: tuples of path, value where path is a (multi)index in tree
    """
    key_value_iterator = tree.items() if isinstance(tree, Mapping) else enumerate(tree)
    for key, value in key_value_iterator:
        if is_value_of_interest(value):
            yield (key,), value
        if isinstance(value, get_args(NonLeafTree)):
            for path, value in get_items_of_interest(tree=value, is_value_of_interest=is_value_of_interest):
                yield (key,) + path, value
