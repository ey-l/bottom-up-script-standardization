import ast

import astpretty
import graphviz
from dataclasses import dataclass, field
from typing import List, Union, Any, Tuple, Dict
from collections import namedtuple
import copy

"""
Limitations:
2. No function definition
3. No scoping
5. Each Expr/Stmt don't cross multiple lines of code
"""

"""
Assumptions:
1. SubscriptAssign: dict[key] = value ---> refreshes the dict
2. All modifications to an object is done through Assignment. No in-place operators like in-place drop.
3. The length of the target of assignment is at most 1.
"""

"""
Questions:
1. a.b.func() ==> resolved
2. a = b + 1
   c = a + 1  ==> resolved
3. a.b()[0] = 1
4. a[1][2] = 3  ==> resolved
5. c = (a[1]=2)  ???
6. a, b[key] = c （let's not bother for now)
"""


class StateVarNode:
    """
    Special Nodes that do not have a corresponding region of code.
    """
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return self.name

    def set_id(self, id):
        self.id = f"{self.name}_{id}"


class DataNode:

    attrs = ('ast_node', 'value', 'lineno', 'col_offset', 'end_lineno', 'end_col_offset', 'astdag', 'count_id', 'id', 'body_idx')
    
    def __init__(self, ast_node=None, value=None, lineno=None, col_offset=None, astdag=None, body_idx=None):
        assert(astdag is not None)
        assert(body_idx is not None)

        self.body_idx = body_idx
        self.ast_node = ast_node
        self.value = value
        self.lineno = lineno
        self.col_offset = col_offset
        self.astdag = astdag
        self.get_pos_in_script()
        self.count_id = self.astdag.node_count_id
        self.astdag.node_count_id += 1        
        self.astdag.count_id_to_node[self.count_id] = self
        self.id = str(self.id_tuple) + "_" + str(self.count_id)

        self.astdag.DAG_nodes.add(self)

    def __repr__(self):
        """The label to be shown in visualization."""
        postfix = f'_{self.id}' if self.id is not None else ''
        if isinstance(getattr(self, 'ast_node', None), ast.Name):
            return self.ast_node.id + postfix
        elif isinstance(getattr(self, 'ast_node', None), ast.Constant):
            return str(self.ast_node.value) + postfix
        elif isinstance(getattr(self, 'ast_node', None), ast.Load) or \
            isinstance(getattr(self, 'ast_node', None), ast.Store) or \
            isinstance(getattr(self, 'ast_node', None), ast.Del):
            return self.ast_node.__class__.__name__ + postfix
        elif isinstance(self, VarNode):
            if self.id is not None:
                return str(self.id)
            else:
                return str(self.value)
        elif isinstance(self, ConstNode):
            return f"{self.value}" + postfix
    
    def clean(self):
        """Make itself the base form."""
        self.lineno = None
        self.col_offset = None
        self.end_lineno = None
        self.end_col_offset = None
        self.astdag = None
        self.count_id = None
        self.id = None
        self.body_idx = None
        if self.ast_node is not None:
            for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
                if hasattr(self.ast_node, attr):
                    setattr(self.ast_node, attr, None)
                    # del self.ast_node[attr]
    
    @staticmethod
    def _eq_ast_node(n1, n2):
        if type(n1) is not type(n2):
            return False
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if getattr(n1, attr, None) != getattr(n2, attr, None):
                return False
            return True
    
    @staticmethod
    def _hash_ast_node(n):
        rst = [type(n)]
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            rst.append(getattr(n, attr, None))
        return hash(tuple(rst))

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        """
        print('lineno', self.lineno == other.lineno)
        print('col_offset', self.col_offset == other.col_offset)
        print('eq_ast_node', self._eq_ast_node(self.ast_node, other.ast_node))
        print('astdag', self.astdag == other.astdag)
        print('count_id', self.count_id, other.count_id, self.count_id == other.count_id)
        print('id', self.id , other.id, self.id == other.id)
        print('value', self.value == other.value)
        print('body_idx', self.body_idx == other.body_idx)
        """
        return self.lineno == other.lineno and self.col_offset == other.col_offset \
            and self.end_lineno == other.end_lineno and self.end_col_offset == other.end_col_offset \
            and self._eq_ast_node(self.ast_node, other.ast_node) \
            and self.count_id == other.count_id \
            and self.id == other.id \
            and self.value == other.value \
            and self.body_idx == other.body_idx
            #and self.astdag == other.astdag
    
    def __hash__(self):
        rst = []
        for attr in self.attrs:
            if attr == 'ast_node':
                rst.append(self._hash_ast_node(getattr(self, attr, None)))
            else:
                rst.append(hash(getattr(self, attr, None)))
        return hash(tuple(rst))
    
    @property
    def id_tuple(self):
        return self.get_pos_in_script()
    
    def get_pos_in_script(self):
        self.lineno = self.lineno if self.lineno is not None and self.lineno != -1 else (self.ast_node.lineno if (hasattr(self.ast_node, 'lineno') and self.ast_node is not None) else -1)
        self.end_lineno = self.lineno  # assume no AST node corosses multiple lines
        self.col_offset = self.col_offset if self.col_offset is not None else (self.ast_node.col_offset if (hasattr(self.ast_node, 'col_offset') and self.ast_node is not None) else -1)
        self.end_col_offset = self.ast_node.end_col_offset if (hasattr(self.ast_node, 'end_col_offset') and self.ast_node is not None) else -1
        return (self.lineno, self.col_offset, self.end_col_offset)

    def label(self):
        return repr(self).split('_')[0]


class VarNode(DataNode):
    def __init__(self, astdag, ast_node=None, value=None, lineno=None, col_offset=None, body_idx=None):
        """
        'ast_node' should present when representing a variable.
        'value' should only be used when capturing variables introduced in Import-related constructs.
        """
        if ast_node is None and value is None:
            raise ValueError("Either ast_node or value has to be NOT None.")
        if ast_node is not None and not isinstance(ast_node, ast.Name):
            raise ValueError(f'VarNode need a Name node, but got {ast_node.__class__.__name__}.')
        if value is not None and not isinstance(value, str):
            raise ValueError(f"'value' has to be a string but got {type(value)}")
        
        super().__init__(ast_node=ast_node, astdag=astdag, value=value, lineno=lineno, col_offset=col_offset, body_idx=body_idx)

        self.ctx = None  # one of None, ast.Load, ast.Store, or ast.Del

        if isinstance(ast_node, ast.Name) and value is None:
            name = ast_node.id
            self.ctx = ast_node.ctx
            if isinstance(ast_node.ctx, ast.Load):
                # built-in namse have no cresponding prev, e.g., print
                prev = self.astdag.sym_table.get(name, None)
                if prev is not None:  
                    self.astdag.add_DataCoref_edge(prev[0], self)
            if isinstance(ast_node.ctx, ast.Del):
                raise ValueError("Del not supported yet!")
        elif value is not None:
            # alias.name
            # self.id = f"{value}_{self.lineno}"
            self.id = f"{value}_<{self.count_id}>"
        else:
            raise ValueError('Impossible case in VarNode')
    
    def get_ast_node(self):
        if self.ast_node is None:
            raise NotImplementedError("alias.name")
        return self.ast_node


class ConstNode(DataNode):
    def __init__(self, astdag, ast_node=None, value=None, lineno=None, col_offset=None, body_idx=None):
        """
        'value' is used to represent constants in the script that don't have a corresponding Constant AST node.
        E.g., Attribute.attr, ImportFrom.level
        """
        if ast_node is None and value is None:
            raise ValueError("Either ast_node or value has to be NOT None.")
        # if ast_node is not None and not isinstance(ast_node, ast.Constant):
        #     raise ValueError(f'ConstNode need a Constant AST node, but got {ast_node.__class__.__name__}.')
    
        super().__init__(ast_node=ast_node, value=value, lineno=lineno, col_offset=col_offset, astdag=astdag, body_idx=body_idx)

    def get_ast_node(self):
        return self.value if self.value is not None else self.ast_node


class SyntaxConstructInvNode(DataNode):
    def __init__(self, ast_node, astdag, order: int, top_level : bool, body_idx=None):
        super().__init__(ast_node=ast_node, astdag=astdag, body_idx=body_idx)
        self.order = order
        self.top_level = top_level
        self.op = self.astdag.extract_from_node(self.ast_node, 'operator')
        # self.id = f"{self.ast_node.lineno, self.ast_node.col_offset, self.ast_node.end_col_offset}<{self.count_id}>"

    def __repr__(self):     
        postfix = f'_{self.id}' if self.id is not None else ''   
        return f'{self.op}' + postfix
    
    def get_ast_node(self):
        return self.ast_node


Indexing = lambda x: x.__getitem__
DotAcc = lambda x: x.__getattribute__


@dataclass
class AccessStep:
    method: Union[Indexing, DotAcc]
    key: Union[str, int]

    @property
    def label(self):
        if self.method == DotAcc:
            return f".{self.key}"
        else:
            return f"[{self.key}]"
    
    def __eq__(self, other):
        if not isinstance(other, AccessStep):
            return False
        return self.label == other.label


@dataclass
class AccessSteps:
    steps: list[AccessStep] = field(default_factory=list)

    def delete(self, other):
        """
        'other' is the node that receiving the inputs
        """
        other = getattr(other, 'ast_node', other)
        return self._delete_from_AST(other)
    
    def _delete_from_AST(self, ast_node):
        """
        Delete the element that is accessed by self from ast_node.
        """
        val = ast_node
        for step in self.steps[0:-1]:
            val = step.method(val)(step.key) #step.key
        last_step = self.steps[-1]
        if last_step.method == Indexing:
            # Handle the case where val is a list
            last_key = last_step.key
            while (last_key > 0):
                try:
                    del val[last_step.key]
                    return last_key
                except:
                    last_key -= 1
        else:
            object.__setattr__(val, last_step.key, None)
        return last_step.key
    
    def insert(self, other, to_insert):
        """
        Insert 'to_insert' into 'other' as the element that is accessed by self.
        """
        other = getattr(other, 'ast_node', other)
        to_insert = getattr(to_insert, 'ast_node', to_insert)
        self._insert_to_AST(other, to_insert)
    
    def _insert_to_AST(self, ast_node, ast_to_insert):
        val = ast_node
        for step in self.steps[:-1]:
            val = step.method(val)(step.key) #step.key
        last_step = self.steps[-1]
        if last_step.method == Indexing:
            #print("val:", val)
            #print('last_step.key:', last_step.key)
            if len(val) == 0:
                val = [ast_to_insert]
            else: 
                val[last_step.key] = ast_to_insert
        else:
            object.__setattr__(val, last_step.key, ast_to_insert)

    @property
    def label(self):
        return "".join([i.label for i in self.steps])
    
    def __eq__(self, other):
        if isinstance(other, AccessSteps):
            return self.steps == other.steps
        return False

    def __hash__(self):
        return hash(self.label)


@dataclass
class UsedAsInAST:
    val: Union[ast.AST, DataNode]
    access: AccessStep


class Edge():
    def __init__(self, _s, _e):
        self._s = _s
        self._e = _e

    def __repr__(self):
        return f"({self.__class__.__name__}) {repr(self._s)} -> {repr(self._e)}"
    
    @property
    def label(self):
        assert(0)
    
    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self._s == other._s and self._e == other._e and getattr(self, 'access', None) == getattr(other, 'access', None)
    
    def __hash__(self):
        return hash((self._s, self._e, getattr(self, 'access', None)))

    def clean(self):
        self._s.clean()
        self._e.clean()


class DataCorefEdge(Edge):
    def __init__(self, old, new):
        super().__init__(old, new)
    
    def __repr__(self):
        return f'(Coref) {repr(self._s)} -- {repr(self._e)}'
    
    @property
    def label(self):
        return 'coref'


class DataFlowEdge(Edge):
    def __init__(self, starting_node:DataNode, ending_node:DataNode, access:AccessSteps):
        super().__init__(starting_node, ending_node)
        self.access = access

    @property
    def label(self):
        return self.access.label


class DefineEdge(Edge):
    @property
    def label(self):
        return 'defines'


class AtomFlowEdge(DataFlowEdge):
    """
    An edge between two atoms. Must be representing a DataFlow relation.
    """
    def __init__(self, starting_node:DataNode, ending_node:DataNode, access:AccessSteps):
        super().__init__(starting_node, ending_node, access)
    
    def __eq__(self, other):
        if isinstance(other, AtomFlowEdge):
            return self._s == other._s and self._e == other._e and self.access == other.access
        return False

    def __hash__(self):
        return hash( (self._s, self._e, self.access) )

    def __repr__(self):
        return f'AtomFlowEdge({self._s} -{self.access.label}-> {self._e})'

    @property
    def base(self):
        return AtomFlowEdgeBase(self)


class AtomFlowEdgeBase:
    """
    A base class for AtomFlowEdge. This is what we store in our AtomCollection.
    """
    def __init__(self, edge):
        self._s = edge._s.base
        self._e = edge._e.base
        self.access = edge.access

    def __eq__(self, other):
        if isinstance(other, AtomFlowEdgeBase):
            return self._s == other._s and self._e == other._e and self.access == other.access
        return False
    
    def __hash__(self):
        return hash( (self._s, self._e, self.access) )

    def __repr__(self):
        return f'AtomFlowEdgeBase({self._s} -{self.access.label}-> {self._e})'

class Atom:
    """
    syntax_const_nodes: Tuple[SyntaxConstructInvNode]. ordered SyntaxConstructInvNodes that are involved in the atom.
    border_edges: Tuple[Tuple[Edge]]. border_edges[i] are edges that linked from an external non-syntax-construct node to syntax_const_nodes[i].
    internal_edges: Tuple[Edge]. internal_edges[i] is the edge connecting syntax_const_nodes[i] and syntax_const_nodes[i+1].
    external_nodes: Tuple[Tuple[DataNode]]. external_nodes[i] are external non-syntax-construct nodes that are linked to syntax_const_nodes[i].
    edges_to_remove: Tuple[Tuple[Edge]]. edges_to_remove[i] are edges to remove from the AST node in syntax_const_nodes[i] when cleaning the atom.
    """

    def __init__(self, syntax_const_nodes, border_edges, internal_edges, external_nodes, edges_to_remove):
        self.syntax_const_nodes = syntax_const_nodes
        self.border_edges = list(border_edges)
        self.internal_edges = internal_edges
        self.external_nodes = list(external_nodes)
        self.edges_to_remove = edges_to_remove
        self.lineno, self.col_offset = syntax_const_nodes[0].lineno, syntax_const_nodes[0].col_offset
        # assert(all([isinstance(e, DataFlowEdge) for edges in border_edges for e in edges]))
        # assert(all([isinstance(e, DataFlowEdge) for e in internal_edges]))
        self.atom_base = None

        # sort the external edges based on the AccessSteps
        for idx in range(len(self.syntax_const_nodes)):
            border_edge_set = self.border_edges[idx]
            external_node_set = self.external_nodes[idx]
            tmp = [(i, j) for i, j in zip(border_edge_set, external_node_set)]
            tmp = sorted(tmp, key=lambda x: x[0].access.label)
            if len(tmp) == 0:
                self.border_edges[idx], self.external_nodes[idx] = (), ()
            else:
                self.border_edges[idx], self.external_nodes[idx] = zip(*tmp)
        
        self.border_edges = tuple(self.border_edges)
        self.external_nodes = tuple(self.external_nodes)
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return self.syntax_const_nodes == other.syntax_const_nodes and self.border_edges == other.border_edges and self.internal_edges == other.internal_edges and self.external_nodes == other.external_nodes \
             and self.edges_to_remove == other.edges_to_remove and self.lineno == other.lineno and self.col_offset == other.col_offset

    def __hash__(self):
        return hash( (self.syntax_const_nodes, self.border_edges, self.internal_edges, self.external_nodes, self.lineno, self.col_offset) )

    def __repr__(self):
        return f"Atom({self.lineno}): {'_'.join([n.op for n in self.syntax_const_nodes])}"

    @property
    def base(self):
        if self.atom_base is None:
            self.atom_base = AtomBase(self)
        return self.atom_base


class AtomBase:
    """
    'Base form' of an Atom. This is what we store in AtomCollection.
    """
    def __init__(self, atom):
        atom = copy.deepcopy(atom)
        self.syntax_const_nodes = atom.syntax_const_nodes
        self.border_edges = atom.border_edges
        self.internal_edges = atom.internal_edges
        self.external_nodes = atom.external_nodes

        # this is kinda like detaching the atom from the graph
        for node in self.syntax_const_nodes:
            node.clean()
        for edge_set in self.border_edges:
            for edge in edge_set:
                edge.clean()
        for edge in self.internal_edges:
            edge.clean()
        for node_set in self.external_nodes:
            for node in node_set:
                node.clean()

        # remove edges that connect to external SyntaxConstructInvNode
        for syntax_const_node, edges_to_remove in zip(self.syntax_const_nodes, atom.edges_to_remove):
            for edge in edges_to_remove:
                edge.access.delete(syntax_const_node)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.syntax_const_nodes == other.syntax_const_nodes and self.border_edges == other.border_edges and self.internal_edges == other.internal_edges and self.external_nodes == other.external_nodes

    def __hash__(self):
        return hash( (self.syntax_const_nodes, self.border_edges, self.internal_edges, self.external_nodes) )
    
    def __repr__(self):
        return f"AtomBase({'_'.join([n.op for n in self.syntax_const_nodes])+'_'+'_'.join([repr(n) for n in self.external_nodes])})"



class ASTDAG:
    def __init__(self, root=None, DAG_nodes=None, DAG_edges=None, filename=None):
        self.root = root
        self.DAG_nodes = DAG_nodes
        self.DAG_edges = DAG_edges
        self.filename = filename

    def gen_AST(self):
        if self.DAG_nodes is None or self.DAG_edges is None:
            raise ValueError("DAG_nodes and DAG_edges have to be set before generating AST.")
        
        statements = sorted([n for n in self.DAG_nodes if isinstance(n, SyntaxConstructInvNode) and n.top_level], key=lambda x: x.order)
        self.DAG_nodes_ids = {n.id for n in self.DAG_nodes}

        root = ast.Module(body=[], type_ignores=[])
        for node in statements:
            if self._is_expression(node):
                # root.body.append(ast.Expr(value=self.gen_AST_from_SyntaxConstrct(node), lineno=node.lineno, col_offset=node.col_offset, end_col_offset=node.end_col_offset, end_lineno=node.end_lineno))
                root.body.append(ast.Expr(value=node.get_ast_node()))
            else:
                root.body.append(node.get_ast_node())

        ast.fix_missing_locations(root)
        return root
    
    def _is_expression(self, node):
        # TODO
        stms = [
            'Assign', 'AugAssign', 'AnnAssign', 'Delete', 'For', 'AsyncFor', 'While', 'If', 'With', 'AsyncWith', 'Raise', 'Try', 'Assert', 'Import', 'ImportFrom', 'Global', 'Nonlocal', 'Expr', 'Pass', 'Break', 'Continue', 'Return'
        ]
        return not node.ast_node.__class__.__name__ in stms
    
    @staticmethod
    def wrap_up_for_correctness_measures(import_name, func_name, tree=None, file=None):
        """
        import_name: the module we implemented for the correctness measures
        func_name: the function we implemented for saving DFs at the end of a script
        """
        if tree is None is None and file is None:
            raise ValueError("Either tree, code or file must be provided!")
        if tree is not None and file is not None:
            raise ValueError("Only one of tree and file can be provided!")
        if tree is None:
            with open(file, encoding='utf-8') as f:
                tree = ast.parse(f.read())
        
        import_node = ast.Import(  # import import_name
                                    names=[ast.alias(name=import_name, asname=None)],
                                    level=0)
        
        save_df_node = ast.Call(func=ast.Attribute(value=ast.Name(id=import_name, ctx=ast.Load()), attr=func_name, ctx=ast.Load()), args=[
                                    ast.Name(id='__file__', ctx=ast.Load()), ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])], keywords=[])

        tree.body.insert(0, import_node)
        tree.body.append(ast.Expr(save_df_node))

        ast.fix_missing_locations(tree)
        return tree
    
    @staticmethod
    def insert_lines(lines: List[str], tree=None, file=None):
        if tree is None and file is None:
            raise ValueError("Either tree, code or file must be provided!")
        if tree is not None and file is not None:
            raise ValueError("Only one of tree and file can be provided!")
        if tree is None:
            with open(file, encoding='utf-8') as f:
                tree = ast.parse(f.read())
        
        for line in list(reversed(lines)):
            line_tree = ast.parse(line)
            line_node = line_tree.body[0]
            tree.body.insert(0, line_node)
        
        ast.fix_missing_locations(tree)
        return tree
    
    @staticmethod
    def append_lines(lines: List[str], tree=None, file=None):
        if tree is None and file is None:
            raise ValueError("Either tree, code or file must be provided!")
        if tree is not None and file is not None:
            raise ValueError("Only one of tree and file can be provided!")
        if tree is None:
            with open(file, encoding='utf-8') as f:
                tree = ast.parse(f.read())
        
        for line in lines:
            line_tree = ast.parse(line)
            line_node = line_tree.body[0]
            tree.body.append(line_node)
        
        ast.fix_missing_locations(tree)
        return tree

    def gen_DAG(self, root=None):
        self.root = root if root is not None else self.root
        self.syn_construct_counts = 0
        self.sym_table = {}  # assumes no scoping, Dict[var_name, VarNode]
        self.DAG_nodes = set()
        self.DAG_edges = set()
        self.node_count_id = 0
        self.count_id_to_node = {}

        if not isinstance(self.root, ast.Module):
            raise ValueError('The root is not a Module.')
        for idx, node in enumerate(self.root.body):
            if isinstance(node, ast.Expr):
                node = node.value
            self.gen_sub_DAG(node, top_level=True, body_idx=idx)
    
    def gen_sub_DAG(self, node, top_level, body_idx):
        """
        Return: a SyntaxConstructInvNode or VarNode, or ConsNode
        """

        if isinstance(node, DataNode):
            return node
        elif isinstance(node, ast.Constant):
            return ConstNode(ast_node=node, astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.Name):
            return VarNode(ast_node=node, astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.Eq) or isinstance(node, ast.NotEq) or isinstance(node, ast.Lt) or isinstance(node, ast.LtE) or isinstance(node, ast.Gt) or isinstance(node, ast.GtE) or isinstance(node, ast.Is) or isinstance(node, ast.IsNot) or isinstance(node, ast.NotIn) or isinstance(node, ast.In):
            # Comparison operator tokens
            return ConstNode(ast_node=node, astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.Load) or isinstance(node, ast.Store) or isinstance(node, ast.Del):
            return ConstNode(ast_node=node, astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.arguments):
            # For Lambda, a corse way to handle
            return ConstNode(ast_node=node, astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.keyword):
            invoc = SyntaxConstructInvNode(ast_node=node, astdag=self, order=self.syn_construct_counts, top_level=False, body_idx=body_idx)
            self.syn_construct_counts += 1
            inputs = [UsedAsInAST(val=ConstNode(value=node.arg, astdag=self, body_idx=body_idx), access=AccessSteps([AccessStep(DotAcc, 'arg')])),
                UsedAsInAST(val=self.gen_sub_DAG(node.value, top_level=False, body_idx=body_idx), access=AccessSteps([AccessStep(DotAcc, 'value')]))
            ]
            self.update_DAG(inputs, invoc, [])
            return invoc
        elif node is None:
            assert(0)
        elif isinstance(node, tuple):
            # Attribute.attr is a string
            return ConstNode(value=node[0], lineno=node[1], col_offset=node[2], astdag=self, body_idx=body_idx)
        elif isinstance(node, ast.Import):
            invoc = SyntaxConstructInvNode(ast_node=node, astdag=self, order=self.syn_construct_counts, top_level=True, body_idx=body_idx)
            self.syn_construct_counts += 1

            inputs, defines = [], self.get_define_vars(node)
            for idx, name in enumerate(node.names):
                if name.asname is not None:
                    tmp = ConstNode(value=name.asname, astdag=self, lineno=node.lineno, body_idx=body_idx)
                    inputs.append(UsedAsInAST(val=ConstNode(value=name.name, lineno=node.lineno, astdag=self, body_idx=body_idx), access=AccessSteps([
                        AccessStep(DotAcc, 'names'), AccessStep(Indexing, idx), AccessStep(DotAcc, 'name')
                    ])))
                else:
                    tmp = ConstNode(value=name.name, astdag=self, lineno=node.lineno, body_idx=body_idx)

                inputs.append(UsedAsInAST(val=tmp, access=AccessSteps([
                        AccessStep(DotAcc, 'names'), AccessStep(Indexing, idx), AccessStep(DotAcc, 'asname')
                    ])))
            self.update_DAG(inputs, invoc, defines)
            return invoc
        elif isinstance(node, ast.ImportFrom):
            invoc = SyntaxConstructInvNode(ast_node=node, astdag=self, order=self.syn_construct_counts, top_level=True, body_idx=body_idx)
            self.syn_construct_counts += 1

            module = node.module
            level = node.level
            inputs, defines = [], self.get_define_vars(node)
            inputs.append(UsedAsInAST(val=ConstNode(value=module, astdag=self, lineno=node.lineno, col_offset=-1, body_idx=body_idx), access=AccessSteps([AccessStep(DotAcc, 'module')])))

            for idx, name in enumerate(node.names):
                if name.asname is not None:
                    tmp = ConstNode(value=name.asname, astdag=self, lineno=node.lineno, body_idx=body_idx)
                    inputs.append(UsedAsInAST(val=ConstNode(value=name.name, lineno=node.lineno, astdag=self, body_idx=body_idx), access=AccessSteps([
                        AccessStep(DotAcc, 'names'), AccessStep(Indexing, idx), AccessStep(DotAcc, 'name')
                    ])))
                else:
                    tmp = ConstNode(value=name.name, astdag=self, lineno=node.lineno, body_idx=body_idx)

                inputs.append(UsedAsInAST(val=tmp, access=AccessSteps([
                        AccessStep(DotAcc, 'names'), AccessStep(Indexing, idx), AccessStep(DotAcc, 'asname')
                    ])))
    
            self.update_DAG(inputs, invoc, defines)
            return invoc

        # SyntaxConstructInvNode  / Statement
        if node.lineno != node.end_lineno and node.lineno is not None and node.end_lineno is not None:
            raise NotImplementedError('We do not support constructs in multiple lines of code yet!')

        invoc = SyntaxConstructInvNode(ast_node=node, astdag=self, order=self.syn_construct_counts, top_level=top_level, body_idx=body_idx)
        self.syn_construct_counts += 1

        inputs = self.extract_from_node(node, 'inputs')  # A list of AST nodes        
        inputs = [UsedAsInAST(val=self.gen_sub_DAG(n.val, top_level=False, body_idx=body_idx), access=n.access) for n in inputs]  # A sequence of sequence

        defines = self.get_define_vars(node)

        self.update_DAG(inputs, invoc, defines)

        return invoc
    
    def get_define_vars(self, ast_node):
        if isinstance(ast_node, ast.Assign):
            defines = self.get_assign_outputs(ast_node.targets)
        elif isinstance(ast_node, ast.Import) or isinstance(ast_node, ast.ImportFrom):
            defines = []
            for name in ast_node.names:
                if name.asname is not None:
                    defines.append(StateVarNode(name.asname))
                else:
                    defines.append(StateVarNode(name.name))
        else:
            defines = []
        
        return defines
    
    def get_value_vars(self, node, cur_rst):
        """
        cur_rst: a list of AST Name nodes.
        Put all Name nodes in the value field in the tree starting from 'node' ino cur_rst. 
        """
        if isinstance(node, ast.Name):
            cur_rst.append(StateVarNode(node.id))
            return
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Subscript):
            self.get_value_vars(node.value, cur_rst)

    def get_assign_outputs(self, targets) -> list[StateVarNode]:
        defines = []
        for target in targets:
            if isinstance(target, ast.Name):
                new_var = StateVarNode(target.id)
                defines.append(new_var)
            elif isinstance(target, ast.Attribute) or isinstance(target, ast.Subscript):
                assert(isinstance(target.ctx, ast.Store))
                target = target.value
                other_redefines = []
                self.get_value_vars(target, other_redefines)
                defines = defines + other_redefines

        return defines

    def add_DataCoref_edge(self, old, new):
        self.DAG_edges.add(DataCorefEdge(old, new))

    def update_DAG(self, inputs: List[UsedAsInAST], syntax_invoc, defines: List[StateVarNode]):
        self.DAG_nodes.update([i.val for i in inputs])
        # self.DAG_nodes.update(defines)
        for node in defines:
            self.sym_table.setdefault(node.name, [node, -1])
            self.sym_table[node.name][0] = node
            self.sym_table[node.name][1] += 1
            node.set_id(self.sym_table[node.name][1])
        self.DAG_nodes.add(syntax_invoc)

        for node in inputs:
            self.DAG_edges.add(DataFlowEdge(node.val, syntax_invoc, access=node.access))
        for node in defines:
            self.DAG_edges.add(DefineEdge(syntax_invoc, node))

    def extract_from_node(self, ast_node, info:str):
        """
        The return value is either 
        1. a list of AST nodes or None
        3. a string representing the operator / an AST node representing the operator

        """
        assert(info in ['inputs', 'operator'])
        if isinstance(ast_node, ast.Call):
            return {
                'operator': 'Call',
                # TODO: add support for keyword arguments
                'inputs': [UsedAsInAST(val=ast_node.func, access=AccessSteps([AccessStep(DotAcc, 'func')]))] +\
                     [UsedAsInAST(val=arg, access=AccessSteps([AccessStep(DotAcc, 'args'), AccessStep(Indexing, idx)])) for idx, arg in enumerate(ast_node.args)] +\
                         [UsedAsInAST(val=keyword, access=AccessSteps([AccessStep(DotAcc, 'keywords'), AccessStep(Indexing, idx)])) for idx, keyword in enumerate(ast_node.keywords)]
            }.get(info, None)
        if isinstance(ast_node, ast.Compare):
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': [UsedAsInAST(val=ast_node.left, access=AccessSteps([AccessStep(DotAcc, 'left')]))] +\
                     [UsedAsInAST(val=comparator, access=AccessSteps([AccessStep(DotAcc, 'comparators'), AccessStep(Indexing, idx)])) for idx, comparator in enumerate(ast_node.comparators)] +\
                         [UsedAsInAST(val=op, access=AccessSteps([AccessStep(DotAcc, 'ops'), AccessStep(Indexing, idx)])) for idx, op in enumerate(ast_node.ops)]
            }.get(info, None)
        if isinstance(ast_node, ast.Lambda):
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': [UsedAsInAST(val=ast_node.args, access=AccessSteps([AccessStep(DotAcc, 'args')]))] +\
                    [UsedAsInAST(val=ast_node.body, access=AccessSteps([AccessStep(DotAcc, 'body')]))]
            }.get(info, None)
        elif isinstance(ast_node, ast.Assign):
            return {
                'operator': 'Assign',
                'inputs': [UsedAsInAST(val=ast_node.value, access=AccessSteps([AccessStep(DotAcc, 'value')]))] +
                [UsedAsInAST(val=target, access=AccessSteps([AccessStep(DotAcc, 'targets'), AccessStep(Indexing, idx)])) for idx, target in enumerate(ast_node.targets)]
            }.get(info, None)
        elif isinstance(ast_node, ast.Attribute):
            return {
                'operator': 'Attribute',
                'inputs': (UsedAsInAST(val=ast_node.value, access=AccessSteps([AccessStep(DotAcc, 'value')])), 
                UsedAsInAST(val=(ast_node.attr, ast_node.lineno, ast_node.value.end_col_offset+1, ast_node.value.end_col_offset+1+len(ast_node.attr)), access=AccessSteps([AccessStep(DotAcc, 'attr')])))
            }.get(info, None)
        elif isinstance(ast_node, ast.Subscript):
            if isinstance(ast_node.ctx, ast.Load) or isinstance(ast_node.ctx, ast.Store):
                return {
                    'operator': "Subscript",
                    'inputs': (UsedAsInAST(val=ast_node.value, access=AccessSteps([AccessStep(DotAcc, 'value')])), UsedAsInAST(val=ast_node.slice, access=AccessSteps([AccessStep(DotAcc, 'slice')])))
                }.get(info, None)
            else:
                raise NotImplementedError(f'Subscript {ast_node.ctx} is not implemented yet!')
        elif isinstance(ast_node, ast.Tuple) or isinstance(ast_node, ast.List):
            if isinstance(ast_node.ctx, ast.Store):
                #raise NotImplementedError(f'Ctx Store for {ast_node.__class__.__name__} is not implemented!')
                pass
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': [UsedAsInAST(val=elt, access=AccessSteps([AccessStep(DotAcc, 'elts'), AccessStep(Indexing, idx)])) for idx, elt in enumerate(ast_node.elts)]
            }.get(info, None)
        elif isinstance(ast_node, ast.Set):
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': [UsedAsInAST(val=elt, access=AccessSteps([AccessStep(DotAcc, 'elts'), AccessStep(Indexing, idx)])) for idx, elt in enumerate(ast_node.elts)]
            }.get(info, None)
        elif isinstance(ast_node, ast.Dict):
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': [UsedAsInAST(val=key, access=AccessSteps([AccessStep(DotAcc, 'keys'), AccessStep(Indexing, idx)])) for idx, key in enumerate(ast_node.keys)] +\
                    [UsedAsInAST(val=value, access=AccessSteps([AccessStep(DotAcc, 'values'), AccessStep(Indexing, idx)])) for idx, value in enumerate(ast_node.values)]
            }.get(info, None)
        elif isinstance(ast_node, ast.Slice):
            return {
                'operator': 'Slice',
                'inputs': ([UsedAsInAST(val=ast_node.lower, access=AccessSteps([AccessStep(DotAcc, 'lower')]))] if ast_node.lower is not None else []) + 
                ([UsedAsInAST(val=ast_node.upper, access=AccessSteps([AccessStep(DotAcc, 'upper')]))] if ast_node.upper is not None else []) + 
                ([UsedAsInAST(val=ast_node.step, access=AccessSteps([AccessStep(DotAcc, 'step')]))] if ast_node.step is not None else [])
            }.get(info, None)
        elif isinstance(ast_node, ast.BinOp):
            return {
                'operator': ast_node.op.__class__.__name__,
                'inputs': (UsedAsInAST(val=ast_node.left, access=AccessSteps([AccessStep(DotAcc, 'left')])), \
                     UsedAsInAST(val=ast_node.right, access=AccessSteps([AccessStep(DotAcc, 'right')])))
            }.get(info, None)
        elif isinstance(ast_node, ast.UnaryOp):
            return {
                'operator': ast_node.op.__class__.__name__,
                'inputs': (UsedAsInAST(val=ast_node.operand, access=AccessSteps([AccessStep(DotAcc, 'operand')])),)
            }.get(info, None)
        elif isinstance(ast_node, ast.IfExp):
            return {
                'operator': ast_node.__class__.__name__,
                'inputs': (UsedAsInAST(val=ast_node.test, access=AccessSteps([AccessStep(DotAcc, 'test')])), \
                    UsedAsInAST(val=ast_node.body, access=AccessSteps([AccessStep(DotAcc, 'body')])), \
                     UsedAsInAST(val=ast_node.orelse, access=AccessSteps([AccessStep(DotAcc, 'orelse')])))
            }.get(info, None)
        elif isinstance(ast_node, ast.Import):
            return {
                'operator': 'Import'
            }.get(info, None)
        elif isinstance(ast_node, ast.ImportFrom):
            return {
                'operator': 'ImportFrom'
            }.get(info, None)
        elif isinstance(ast_node, ast.keyword):
            return {
                'operator': 'keyword'
            }.get(info, None)
        elif isinstance(ast_node, ast.arguments):
            return {
                'operator': 'arguments'
            }.get(info, None)
        elif isinstance(ast_node, ast.arg):
            return {
                'operator': 'arg'
            }.get(info, None)
        elif isinstance(ast_node, ast.Expr) or isinstance(ast_node, ast.alias):
            raise ValueError("Impossible to get Expr or alias here")
        else:
            raise NotImplementedError(f"{type(ast_node)} not implemented yet!")

    def print_raw_DAG(self):
        print("<<< nodes")
        print(self.DAG_nodes)
        print("<<< edges")
        # print(self.DAG_edges)
        for edge in self.DAG_edges:
            print(edge.label)
        print(len(self.DAG_edges))
    
    def _add_DAG_node(self, g, node):
        if isinstance(node, SyntaxConstructInvNode):
            g.node(node.id, repr(node), shape='box')
        elif isinstance(node, VarNode):
            g.node(node.id, label=repr(node), shape='diamond', style='filled')
        elif isinstance(node, ConstNode):
            g.node(node.id, label=repr(node))
        elif isinstance(node, StateVarNode):
            g.node(node.id, label=repr(node), shape='diamond')
        else:
            raise ValueError(f"Impossible value in DAG_nodes: {type(node)}")

    def DAG_to_Graphviz(self, edges_to_vis=None):
        g = graphviz.Digraph('DAG')
        g.attr(rankdir='LR')
        edges_to_vis = self.DAG_edges if edges_to_vis is None else edges_to_vis
        for edge in edges_to_vis:
            self._add_DAG_node(g, edge._s)
            self._add_DAG_node(g, edge._e)
            g.edge(edge._s.id, edge._e.id, label=edge.label)
        g.view()
    
    def identify_atoms(self):
        """
        Identify atoms and atom_flow_edges in the DAG.
            function call atom: Attribute-Call
            single-construct atom: syntax-construct node + parent non syntax-construct nodes
        So after, DAG consists of atoms and atom_flow_edges.
        """
        atoms = []

        # get Attribute-Call atoms
        long_atoms = self._get_len_two_parttern_from_DAG(['Attribute', 'Call'])
        atoms += long_atoms

        # filter out SyntaxConstructInvNode that are already in Attribute-Call atoms
        included_sytax_const = set([n for atom in long_atoms for n in atom.syntax_const_nodes])

        single_syntax_const = [n for n in self.DAG_nodes if isinstance(n, SyntaxConstructInvNode) and not n in included_sytax_const]
        for n in single_syntax_const:
            # get single-length patterned atoms
            nodes, edges, edges_to_remove = self._get_attached(n)
            atoms.append(Atom(syntax_const_nodes=(n, ), border_edges=(edges, ), external_nodes=(nodes, ), internal_edges=(), edges_to_remove=(edges_to_remove, )))

        # TODO: we could optimize this 3-for loop code in the future if needed.
        atom_flow_edges = []
        for edge in self.DAG_edges:
            for start_atom in atoms:
                if edge._s == start_atom.syntax_const_nodes[-1]:
                    if not isinstance(edge, DataFlowEdge):
                        # purposely leaving out edges related to StateVarNode. 
                        # Not sure if we'd need them in the future.
                        continue
                    for end_atom in atoms:
                        if edge._e == end_atom.syntax_const_nodes[0]:
                            atom_flow_edges.append(AtomFlowEdge(start_atom, end_atom, edge.access))
                            
        return atoms, atom_flow_edges

    def _get_len_two_parttern_from_DAG(self, pattern: List[str]):
        """
        Returns the identified nodes, with any direct inputs attached to them.
        """
        if len(self.DAG_nodes) == 0 or len(self.DAG_nodes) == 0:
            raise ValueError('No nodes or edges in the DAG!')
        
        if len(pattern) != 2:
            raise NotImplementedError(f'Pattern of length {len(pattern)} has not been implemented yet!')
    
        rst: List[Atom] = []
        for edge in self.DAG_edges:
            if not isinstance(edge, DataFlowEdge): continue
            if not isinstance(edge._s, SyntaxConstructInvNode) or not isinstance(edge._e, SyntaxConstructInvNode): continue
            if edge._s.op != pattern[0] or edge._e.op != pattern[1]: continue
            n1, e1, edges_to_remove1 = self._get_attached(edge._s)
            n2, e2, edges_to_remove2 = self._get_attached(edge._e, allowed_edge=edge)
            rst.append(Atom(syntax_const_nodes=(edge._s, edge._e), border_edges=(e1, e2), internal_edges=(edge, ), external_nodes=(n1, n2), edges_to_remove=(edges_to_remove1, edges_to_remove2)))
        return rst
    
    def get_pattern_from_DAG(self, pattern: List[str]) -> List[Atom]:
        """
        Returns the identified atoms.
        Can be used for n-grams.
        TODO: there seems to be bugs when the pattern is longer than 2. Need fix in the future.
        """

        raise ValueError("This function is not working properly. Please use _get_len_two_parttern_from_DAG instead.")
        if len(self.DAG_nodes) == 0 or len(self.DAG_nodes) == 0:
            raise ValueError('No nodes or edges in the DAG!')
        
        if len(pattern) < 2:
            raise NotImplementedError(f'Pattern of length {len(pattern)} has not been implemented yet!')

        rst_nodes: List[List[SyntaxConstructInvNode]] = []
        rst_edges: List[List[DataFlowEdge]] = []
        atom_nodes: List[List[DataNode]] = []
        for i in range(len(pattern)-1):
            p = pattern[i:i+2]
            next_rst_nodes, next_rst_edges, next_atom_nodes = self._get_len_two_parttern_from_DAG(p)
            if i == 0:
                rst_nodes = next_rst_nodes
                rst_edges = next_rst_edges
                atom_nodes = next_atom_nodes
            else:
                for k, a in enumerate(rst_nodes):
                    for j, b in enumerate(next_rst_nodes):
                        if a[-1] == b[0]:
                            rst_nodes[k].append(b[1])
                            rst_edges[k] = list(set(rst_edges[k] + next_rst_edges[j]))
                            atom_nodes[k] = list(set(atom_nodes[k] + next_atom_nodes[j]))
        
        atoms: List[Atom] = [Atom(root=None, atom_nodes=atom_nodes[i], atom_edges=rst_edges[i]) for i in range(len(rst_nodes))]
        
        return atoms

    def _get_attached(self, node: SyntaxConstructInvNode, allowed_edge=None):
        """
        Input:
        allowed_edge: used when finding an atom that has more than one SyntaxConstructInvNode nodes

        Output:
        nodes: get all non-SyntaxConstructInvNode nodes attached to the target node
        edges: get all edges connecting nodes and the target node
        edges_to_remove: get all edges that should be removed from the DAG when cleaning
        """
        nodes, edges, edges_to_remove = [], [], []

        for e in self.DAG_edges:
            if e._e != node:
                continue

            assert(not isinstance(e._s, StateVarNode))

            if not isinstance(e._s, SyntaxConstructInvNode):
                nodes.append(e._s)
                edges.append(e)
            elif e != allowed_edge:
                edges_to_remove.append(e)

        return tuple(nodes), tuple(edges), tuple(edges_to_remove)

    def delete_node_by_pos_id(self, id):
        node_to_rmv = None
        for node in self.DAG_nodes:
            if node.id_tuple == id:
                node_to_rmv = node
        if node_to_rmv is None:
            raise ValueError("<<< The ID doesn't cresponde to any existing node!")
        edges_to_remove = self._delete_node(node_to_rmv)
        self.DAG_nodes.remove(node_to_rmv)
        self.DAG_edges.difference_update(edges_to_remove)
    
    def delete_node_by_count_id(self, count_id):
        if count_id > self.node_count_id or count_id <= 0:
            raise ValueError("<<< The ID doesn't cresponde to any existing node!")
        node_to_rmv = self.count_id_to_node[count_id]
        edges_to_remove = self._delete_node(node_to_rmv)
        self.DAG_nodes.remove(node_to_rmv)
        self.DAG_edges.difference_update(edges_to_remove)

    def _delete_node(self, node_to_rmv):
        edges_to_remove = []
        for edge in self.DAG_edges:
            if edge._e == node_to_rmv and isinstance(edge._s, SyntaxConstructInvNode):
                syntax_cons_node = edge._s
            elif edge._s == node_to_rmv and isinstance(edge._e, SyntaxConstructInvNode):
                syntax_cons_node = edge._e
            else:
                continue

            edges_to_remove.append(edge)
            if isinstance(edge, DataFlowEdge):
                access = edge.access
                access.delete(syntax_cons_node)

        #self.DAG_edges.difference_update(edges_to_remove)
        return edges_to_remove


if __name__ == "__main__":
    import sys
    sys.path.insert(0, sys.path[0] + '/..')

    """
    NOTE: Moved all the test cases to test_luciddag.py in scripts/python/
    """