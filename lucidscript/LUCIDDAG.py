from tabnanny import verbose
from turtle import position
from unicodedata import name
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, namedtuple, OrderedDict, defaultdict
from itertools import combinations, permutations, product, chain
from scipy.stats import entropy
import copy
import csv
import os
import glob
import math
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Any, Tuple
import ast
import astpretty
from pathlib import Path
import time

import sys
sys.path.append(sys.path[0] + '/..')
from lucidscript.ASTDAG import *
from lucidscript.utils import *

PROJ_ROOT = Path(sys.path[0] + '/..')

NOOP_TOKEN = 'NOOP'
RESULT_PATH = PROJ_ROOT / "scratch" / "result.csv"
IMPORT_LINES = [
    "import sys",
    "import copy",
    "import pandas as pd",
    f"sys.path.append('{sys.path[0] + '/../lucidscript'}')",
    f"sys.path.append('{sys.path[0] + '/..'}')",
    "from correctness import compute_corr_measures"
]
CORR_LINES = [
    "updated_dataframes = []",
    "l = locals().copy()",
    """
for i in l.items():
    # Check if the object is a DataFrame
    if isinstance(i[1], pd.DataFrame):
        # Add the object (name, DataFrame) to the list
        updated_dataframes.append(copy.deepcopy(i))
    """
]


class LUCIDDAG:
    """
    A layer of abstraction for the LUCID algorithm.

    astdag: ASTDAG,
    atoms,
    atom_flow_edges

    NOTE: Another thing that might work and is faster is that for a LUCIDDAG, we keep an id for each Atom and AtomFlowEdge 
    so that when we make a copy, we don’t have to iterate all nodes and edges to find the match but just grab the one with the same ID.
    """
    def __init__(self, ASTDAG) -> None:
        self.astdag = ASTDAG
        self.atoms = None
        self.atom_flow_edges = None
        self.last_step_lineno = 0 # Keep track of the lineno of the last step
        self.x = None
        self.Px = None
        self.update_LUCID_graph()

    def init_Px(self, atom_collection):
        base_edges = [e.base for e in self.atom_flow_edges]
        #self.csvlogger.info(["get_edge_base", time.time()-base_start, "", "", self.script_i])
        self.x = count_occurrences(base_edges, atom_collection.unique_edges)
        self.Px = compute_prob(self.x)
    
    def get_edges(self) -> List:
        if len(self.atoms) == 0:
            return [NOOP_TOKEN]
        return self.atom_flow_edges

    def get_edge(self, other_edge: AtomFlowEdge) -> AtomFlowEdge:
        """
        Get the edge of this luciddag, given an edge of another luciddag.
        """
        for edge in self.atom_flow_edges:
            if edge == other_edge:
                return edge
        return None

    def get_edge_between_atoms(self, atom1: Atom, atom2: Atom) -> AtomFlowEdge:
        """
        Get the edge between two atoms.
        """
        for edge in self.atom_flow_edges:
            if edge._s == atom1 and edge._e == atom2:
                return edge
        return None
    
    def get_atom_incoming_edges(self, atom: Atom) -> List:
        """
        Get the incoming edges of an atom.
        """
        if not atom in self.atoms:
            raise ValueError("Atom not in LUCID graph.")

        ins = []
        for edge in self.atom_flow_edges:
            if edge._e == atom:
                ins.append(edge)
        return ins

    def get_atom_outgoing_edges(self, atom: Atom) -> List:
        """
        Get the outgoing edges of an atom.
        """
        if not atom in self.atoms:
            raise ValueError("Atom not in LUCID graph.")

        outs = []
        for edge in self.atom_flow_edges:
            if edge._s == atom:
                outs.append(edge)
        return outs

    def get_atom(self, other_atom: Atom) -> Atom:
        """
        Get the edge of this luciddag, given an edge of another luciddag.
        """
        for atom in self.atoms:
            if atom == other_atom:
                return atom
        return None

    def get_nodes(self) -> List:
        return self.atoms

    def update_LUCID_graph(self):
        """
        NOTE: This function is called after each ASTDAG update.
        We don't use networkx to build the LUCID graph anymore because it's easier to
        directly work with our AtomFlowEdge objects.
        
        TODO: Eugenie needs to update Lucid Class to reflect this change.
        """
        self.atoms, self.atom_flow_edges = self.astdag.identify_atoms()

    def update_LUCID_and_ASTDAG_graph(self):
        """
        Build the LUCID graph according to the current ASTDAG.
        Called after each ASTDAG update and in the init function.
        """
        #astpretty.pprint(self.astdag.gen_AST())
        self.astdag.gen_AST()
        self.astdag.gen_DAG()
        #self.astdag.DAG_to_Graphviz()
        self.update_LUCID_graph()

    def save_processed_dataframe(self, dirpath: str):
        """
        Save all the dataframes in the script to dirpath; return filenames.
        :return filenames: List[str]
        """
        # Remove the files in dir before start
        files = glob.glob(dirpath+'/*')
        for f in files:
            os.remove(f)
        
        tree = self.astdag.gen_AST()
        tree = ASTDAG.append_lines(
            [ "l = locals().copy()",
            f"""
for i in l.items():
    # Check if the object is a DataFrame
    if isinstance(i[1], pd.DataFrame):
        i[1].to_csv(\"{dirpath}\"+"/"+i[0]+".csv", index=False)
            """], 
            tree=tree)
        try:
            exe = compile(tree, filename="", mode="exec")
            try:
                exec(exe)
            except Exception as err:
                print("ExecutionError: ", err)
                return None
        except Exception as terr:
            print("TreeError: ", terr)
            return None
        
        # Get filenames
        filenames = next(os.walk(dirpath), (None, None, []))[2]  # [] if no file
        filenames = [fn[:-4] for fn in filenames if fn.split('.')[-1] == 'csv']
        return filenames
    
    def delete_one_atom_and_incoming_edge(self, edge_to_delete: AtomFlowEdge, outgoing_edge: AtomFlowEdge):
        """
        Delete an atom and its incoming edge in the LUCID graph.
        - TransformationBase: delete A_1 -(e_1')-> A'
        - S* before: A_1 -(e_1')-> A' -(e_'2)-> A_2
        - S* after: A_1 -(e_'2)-> A_2
        """
        atom1 = edge_to_delete._s
        atom2 = outgoing_edge._e
        
        outgoing_edge.access.insert(atom2.syntax_const_nodes[0], atom1.syntax_const_nodes[-1])
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom1.lineno
        if self.last_step_lineno is None:
            self.last_step_lineno = atom2.lineno

    def delete_one_atom_and_outgoing_edge(self, edge_to_delete: AtomFlowEdge, incoming_edge: AtomFlowEdge):
        """
        Delete an atom and its outgoing edge in the LUCID graph.
        - TransformationBase: delete A' -(e_'2)-> A_2
        - S* before: A_1 -(e_1')-> A' -(e_'2)-> A_2
        - S* after: A_1 -(e_1')-> A_2
        """
        atom1 = incoming_edge._s
        atom2 = edge_to_delete._e
        edge_to_delete.access.delete(atom2.syntax_const_nodes[0])
        incoming_edge.access.insert(atom2.syntax_const_nodes[0], atom1.syntax_const_nodes[-1])
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom2.lineno
        if self.last_step_lineno is None:
            self.last_step_lineno = atom1.lineno

    def delete_one_atom_at_the_end(self, edge_to_delete: AtomFlowEdge):
        """
        Delete the atom at the end in the LUCID graph.
        - TransformationBase: delete A_1 -(e_1')-> A'
        - S* before: A_1 -(e_1')-> A'
        - S* after: A_1
        """
        atom = edge_to_delete._s
        #print(edge_to_delete.access.label)
        body_idx = atom.syntax_const_nodes[0].body_idx
        #if not isinstance(self.astdag.root.body[body_idx], ast.Expr):
        #    raise NotImplementedError()  # such as ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef
        #astpretty.pprint(self.astdag.root.body[body_idx])
        
        # Check if A_1 would become the last atom after deletion
        if ('value' in edge_to_delete.access.label and self.astdag.root.body[body_idx].value == atom.syntax_const_nodes[-1].ast_node) \
        or ('target' in edge_to_delete.access.label and self.astdag.root.body[body_idx].targets[0] == atom.syntax_const_nodes[-1].ast_node):
            # Assume only 1 target
            self.astdag.root.body[body_idx] = ast.Expr(value=atom.syntax_const_nodes[-1].ast_node)
        else: self.astdag.root.body[body_idx].value = atom.syntax_const_nodes[-1].ast_node
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom.lineno
        if self.last_step_lineno is None:
            self.last_step_lineno = edge_to_delete._e.lineno

    def delete_one_atom_at_the_start(self, edge_to_delete: AtomFlowEdge):
        """
        Delete the atom at the start in the LUCID graph.
        - TransformationBase: delete A' -(e_'2)-> A_2
        - S* before: A' -(e_'2)-> A_2
        - S* after: A_2
        """
        atom2 = edge_to_delete._e
        edge_to_delete.access.delete(atom2.syntax_const_nodes[0])
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom2.lineno
        if self.last_step_lineno is None:
            self.last_step_lineno = edge_to_delete._s.lineno

    def insert_a_line(self, code: str, lineno: int):
        """
        TODO
        """
        body = self.astdag.root.body
        code_in_ast = ast.parse(code).body[0] # There is only one line
        body.insert(lineno, code_in_ast)
        # Set to the new body
        self.astdag.root.body = body
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = lineno
    
    def insert_one_atom_and_incoming_edge(self, atom1: Atom, atom2: Atom, edge_base: AtomFlowEdgeBase):
        """
        Insert a new 'edge' and 'atom' between atom1 and atom2 in the LUCID graph.
        - TransformationBase: insert A_1 -(e_1')-> A'
        - S* before: A_1 -(e_12)-> A_2
        - S* after: A_1 -(e_1')-> A' -(e_12)-> A_2
        """
        new_atom_base = copy.deepcopy(edge_base._e)
        existing_edge = self.get_edge_between_atoms(atom1, atom2)
        if existing_edge is None:
            raise ValueError("No edge between atoms.")
        #atom1 = existing_edge._s
        #atom2 = existing_edge._e
        edge_base.access.insert(new_atom_base.syntax_const_nodes[0], atom1.syntax_const_nodes[-1])
        existing_edge.access.insert(atom2.syntax_const_nodes[0], new_atom_base.syntax_const_nodes[-1])
        #astpretty.pprint(atom2.syntax_const_nodes[0].ast_node)
        #print("----")
        #body_idx = atom2.syntax_const_nodes[0].body_idx
        #astpretty.pprint(self.astdag.root.body[body_idx])
        #self.astdag.root.body[body_idx] = atom2.syntax_const_nodes[0].ast_node

        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom1.lineno

    def insert_one_atom_and_outgoing_edge(self, atom1: Atom, atom2: Atom, edge_base: AtomFlowEdgeBase):
        """
        Insert a new 'edge' and atom between atom1 and atom2 in the LUCID graph.
        - TransformationBase: insert A' -(e_'2)-> A2
        - S* before: A_1 -(e_12)-> A_2
        - S* after: A_1 -(e_12)-> A' -(e_'2)-> A_2
        """
        new_atom_base = copy.deepcopy(edge_base._s)
        existing_edge = self.get_edge_between_atoms(atom1, atom2)
        if existing_edge is None:
            raise ValueError("No edge between atoms.")
        
        #astpretty.pprint(new_atom_base.syntax_const_nodes[0].ast_node)
        #astpretty.pprint(atom1.syntax_const_nodes[-1].ast_node)
        existing_edge.access.insert(new_atom_base.syntax_const_nodes[0], atom1.syntax_const_nodes[-1])
        edge_base.access.insert(atom2.syntax_const_nodes[0], new_atom_base.syntax_const_nodes[-1])

        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom1.lineno
    
    def insert_new_edge_and_atom(self, atom: Atom, edge_base: AtomFlowEdgeBase):
        """
        Insert a new 'edge' and atom before the 'atom' in the LUCID graph.
        - TransformationBase: insert A' -(e_'2)-> A2
        - S* before: A_2
        - S* after: A' -(e_'2)-> A_2
        """
        new_atom_base = copy.deepcopy(edge_base._s)

        # A' is the nested layer of A_2
        edge_base.access.insert(atom.syntax_const_nodes[0], new_atom_base.syntax_const_nodes[-1])
        body_idx = atom.syntax_const_nodes[0].body_idx
        if not isinstance(self.astdag.root.body[body_idx], ast.Expr):
            raise NotImplementedError()  # such as ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef
        
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom.lineno

    def append_new_edge_and_atom(self, atom: Atom, edge_base: AtomFlowEdgeBase):
        """
        Append a new 'edge' and atom to the 'atom' in the LUCID graph.
        - TransformationBase: insert A_1 -(e_1')-> A'
        - S* before: A_1 
        - S* after: A_1 -(e_1')-> A'

        NOTE:
        This could be problematic because this requires each transformation leads to a valid Python script so that
        we could gen_DAG. However, the optimzer might make a transformation that is invalid on its own but actually
        valid with another transformation together.

        Actually, is this really a issue? Isn't the optimizer supposed to make valid transformations?
        """
        new_atom_base = copy.deepcopy(edge_base._e)

        #astpretty.pprint(new_atom_base.syntax_const_nodes[0].ast_node)
        #astpretty.pprint(atom.syntax_const_nodes[-1].ast_node)
        edge_base.access.insert(new_atom_base.syntax_const_nodes[0], atom.syntax_const_nodes[-1])
        body_idx = atom.syntax_const_nodes[0].body_idx
        if not isinstance(self.astdag.root.body[body_idx], ast.Expr):
            raise NotImplementedError()  # such as ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef
        self.astdag.root.body[body_idx].value = new_atom_base.syntax_const_nodes[-1].ast_node
        
        self.update_LUCID_and_ASTDAG_graph()
        self.last_step_lineno = atom.lineno
        
class AtomCollection:
    """
    Attributes:
        incoming_edges_vocab: Dict[AtomBase, Dict[AtomFlowEdgeBase, count_of_occurrence]]
        outgoing_edges_vocab: Dict[AtomBase, Dict[AtomFlowEdgeBase, count_of_occurrence]]
        line_vocab: Dict[str, count_of_occurrence]
    """
    def __init__(self, luciddags: List[LUCIDDAG], tune=1):
        self.luciddags = luciddags
        self.tune = tune
        self._build_vocabs()
    
    def _build_vocabs(self):
        self.incoming_edges_vocab = defaultdict(Counter)
        self.outgoing_edges_vocab = defaultdict(Counter)
        self.edge_vocab = Counter()
        self.line_vocab = Counter() # Add line vocab
        self.line_rel_pos = {} # For greedy
        self.line_edge_mapping = {} # For Px computation

        total = 0
        for idx, luciddag in enumerate(self.luciddags):
            t = time.time()
            for atom_flow_edge in luciddag.atom_flow_edges:
                atom_flow_edge_base = atom_flow_edge.base

                self.outgoing_edges_vocab[atom_flow_edge_base._s][atom_flow_edge_base] += 1
                self.incoming_edges_vocab[atom_flow_edge_base._e][atom_flow_edge_base] += 1
                self.edge_vocab[atom_flow_edge_base] += 1

            body = luciddag.astdag.root.body
            length = len(body)
            for idx, line_in_ast in enumerate(body):
                # Use AST because it standardizes syntax
                # Unparse to strings because we don't have to compare AST objects
                line = ast.unparse(line_in_ast) # This shouldn't fail
                self.line_vocab[line] += 1
                # For naive-greedy SearchStepPosition
                self.line_rel_pos.setdefault(line, [])
                self.line_rel_pos[line].append((idx+1)/length)

                if self.line_vocab[line] == 1:
                    tree = ast.parse(line)
                    d = ASTDAG(root=tree)
                    try:
                        d.gen_DAG()
                        lld = LUCIDDAG(d)
                        self.line_edge_mapping.setdefault(line, [])
                        self.line_edge_mapping[line] = [e.base for e in lld.atom_flow_edges]
                        #print(os.path.join(path, name))
                    except Exception as e:
                        #print(e)
                        pass
            t = time.time() - t
            total += t

        self.unique_edges = list(self.edge_vocab.keys()) + [NOOP_TOKEN]
        # Tuning
        #self.edge_vocab = {key:val for key, val in self.edge_vocab.items() if val > self.tune}
        self.line_vocab = {key:val for key, val in self.line_vocab.items() if val > self.tune}
        self.incoming_edges_vocab = {k: {k2: v2 for k2, v2 in v.items() if v2 > self.tune} for k, v in self.incoming_edges_vocab.items()}
        self.outgoing_edges_vocab = {k: {k2: v2 for k2, v2 in v.items() if v2 > self.tune} for k, v in self.outgoing_edges_vocab.items()}

    def report(self):
        print(f"There are {len(self.edge_vocab)} unique edges.")
        print(f"There are {len(self.line_vocab)} unique lines of code.")
        print(f"{sum(len(v) for v in self.incoming_edges_vocab.values())} unique incoming edges are considered in next steps.")
        print(f"{sum(len(v) for v in self.outgoing_edges_vocab.values())} unique outgoing edges are considered in next steps.")
        #print(self.incoming_edges_vocab)

class Transformation:
    """
    Store the transformation information.

    :type (str): ...

    """
    def __init__(self, x=None, type: str='', atom1: Atom=None, atom2: Atom=None, edge_base: AtomFlowEdgeBase=None, edge_to_delete: AtomFlowEdge=None, incoming_edge: AtomFlowEdge=None, outgoing_edge: AtomFlowEdge=None, line: str=None, lineno: int=None):
        self.x = x
        self.Px = None
        self.luciddag = None
        self.type = type
        self.atom1 = atom1
        self.atom2 = atom2
        self.edge_base = edge_base
        self.edge_to_delete = edge_to_delete
        self.incoming_edge = incoming_edge
        self.outgoing_edge = outgoing_edge
        self.line = line
        self.lineno = lineno

        assert(type in ['', 'append_new_edge_and_atom', 'insert_new_edge_and_atom', 
        'insert_one_atom_and_outgoing_edge', 'insert_one_atom_and_incoming_edge',
        'delete_one_atom_and_outgoing_edge', 'delete_one_atom_and_incoming_edge', 
        'delete_one_atom_at_the_end', 'delete_one_atom_at_the_start',
        'insert_a_line'])
        
        if type == 'append_new_edge_and_atom':
            assert(x is not None and edge_base is not None and atom1 is not None and atom2 is None and edge_to_delete is None and incoming_edge is None and outgoing_edge is None and line is None and lineno is None)
        elif type == 'insert_new_edge_and_atom':
            assert(x is not None and edge_base is not None and atom2 is not None and atom1 is None and edge_to_delete is None and incoming_edge is None and outgoing_edge is None and line is None and lineno is None)
        elif type == 'insert_one_atom_and_outgoing_edge' or type == 'insert_one_atom_and_incoming_edge':
            assert(x is not None and edge_base is not None and atom1 is not None and atom2 is not None and edge_to_delete is None and incoming_edge is None and outgoing_edge is None and line is None and lineno is None)
        elif type == 'delete_one_atom_and_outgoing_edge' or type == 'delete_one_atom_and_incoming_edge':
            assert(x is not None and atom1 is None and atom2 is None and edge_base is None and edge_to_delete is not None and line is None and lineno is None)
            assert(incoming_edge is not None or outgoing_edge is not None)
        elif type == 'delete_one_atom_at_the_end' or type == 'delete_one_atom_at_the_start':
            assert(x is not None and atom1 is None and atom2 is None and edge_base is None and edge_to_delete is not None and incoming_edge is None and outgoing_edge is None and line is None and lineno is None)
        elif type == 'insert_a_line':
            assert(x is not None and atom1 is None and atom2 is None and edge_base is None and edge_to_delete is None and incoming_edge is None and outgoing_edge is None and line is not None and lineno is not None)
        else:
            assert(x is None and atom1 is None and atom2 is None and edge_base is None and edge_to_delete is None and incoming_edge is None and outgoing_edge is None and line is None and lineno is None)
        
    def __eq__(self, other):
        """
        Compare two transformations.
        Not considering lineno for now.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.type == 'append_new_edge_and_atom':
            return self.type == other.type and self.atom1 == other.atom1 and self.edge_base == other.edge_base
        elif self.type == 'insert_new_edge_and_atom':
            return self.type == other.type and self.atom2 == other.atom2 and self.edge_base == other.edge_base
        elif self.type == 'insert_one_atom_and_outgoing_edge' or self.type == 'insert_one_atom_and_incoming_edge':
            return self.type == other.type and self.atom1 == other.atom1 and self.atom2 == other.atom2 and self.edge_base == other.edge_base
        elif self.type == 'delete_one_atom_and_outgoing_edge' or self.type == 'delete_one_atom_and_incoming_edge' or self.type == 'delete_one_atom_at_the_end' or self.type == 'delete_one_atom_at_the_start':
            return self.type == other.type and self.edge_to_delete == other.edge_to_delete
        elif self.type == 'insert_a_line':
            return self.type == other.type and self.line == other.line
        return False

    def __repr__(self):
        if self.type == 'append_new_edge_and_atom':
            return f"Transformation(append_at_end): {self.atom1} -{self.edge_base}-> at lineno {self.atom1.lineno}"
        elif self.type == 'insert_new_edge_and_atom':
            return f"Transformation(insert_at_start): -{self.edge_base}-> {self.atom2} at lineno {self.atom2.lineno}"
        elif self.type == 'insert_one_atom_and_outgoing_edge' or self.type == 'insert_one_atom_and_incoming_edge':
            return f"Transformation(insert): {self.edge_base} between {self.atom1} and {self.atom2} at lineno {self.atom1.lineno}"
        elif self.type == 'delete_one_atom_and_outgoing_edge' or self.type == 'delete_one_atom_and_incoming_edge':
            return f"Transformation(delete): {self.edge_to_delete} at lineno {self.edge_to_delete._e.lineno}"
        elif self.type == 'delete_one_atom_at_the_end':
            return f"Transformation(delete_at_end): {self.edge_to_delete} at lineno {self.edge_to_delete._e.lineno}"
        elif self.type == 'delete_one_atom_at_the_start':
            return f"Transformation(delete_at_start): {self.edge_to_delete} at lineno {self.edge_to_delete._e.lineno}"
        elif self.type == 'insert_a_line':
            return f"Transformation(insert_a_line): {self.line} at lineno {self.lineno}"
        else:
            return f"Transformation()"
    
    def update_Px(self, atom_collection): 
        """
        Compute the updated Px without updating the DAG.
        """
        if self.type == 'append_new_edge_and_atom':
            self.insert_update_DAG_Px(atom_collection, self.edge_base)
        elif self.type == 'insert_new_edge_and_atom':
            self.insert_update_DAG_Px(atom_collection, self.edge_base)
        elif self.type == 'insert_one_atom_and_outgoing_edge':
            self.insert_update_DAG_Px(atom_collection, self.edge_base)
        elif self.type == 'insert_one_atom_and_incoming_edge':
            self.insert_update_DAG_Px(atom_collection, self.edge_base)
        elif self.type == 'delete_one_atom_and_outgoing_edge':
            self.delete_update_DAG_Px(atom_collection, self.edge_to_delete.base)
        elif self.type == 'delete_one_atom_and_incoming_edge':
            self.delete_update_DAG_Px(atom_collection, self.edge_to_delete.base)
        elif self.type == 'delete_one_atom_at_the_end':
            self.delete_update_DAG_Px(atom_collection, self.edge_to_delete.base)
        elif self.type == 'delete_one_atom_at_the_start':
            self.delete_update_DAG_Px(atom_collection, self.edge_to_delete.base)
        else:
            self.insert_update_DAG_Px(atom_collection, self.line)

        return self.x, self.Px
    
    def apply(self, luciddag: LUCIDDAG, atom_collection, verbose=False):
        """
        Apply transformation to the DAG accordingly.
        'AssertionError' object has no attribute 'Px' when Transformation causes invalid DAG.
        """
        # Make a copy of the DAG
        self.luciddag = copy.deepcopy(luciddag)  # get rid of this
        try: 
            if self.type == 'append_new_edge_and_atom':
                self.atom1 = self.luciddag.get_atom(self.atom1)
                self.luciddag.append_new_edge_and_atom(self.atom1, self.edge_base)
            elif self.type == 'insert_new_edge_and_atom':
                self.atom2 = self.luciddag.get_atom(self.atom2)
                self.luciddag.insert_new_edge_and_atom(self.atom2, self.edge_base)
            elif self.type == 'insert_one_atom_and_outgoing_edge':
                e = self.luciddag.get_edge_between_atoms(self.atom1, self.atom2)
                self.atom1 = e._s
                self.atom2 = e._e
                self.luciddag.insert_one_atom_and_outgoing_edge(self.atom1, self.atom2, self.edge_base)
            elif self.type == 'insert_one_atom_and_incoming_edge':
                e = self.luciddag.get_edge_between_atoms(self.atom1, self.atom2)
                self.atom1 = e._s
                self.atom2 = e._e
                self.luciddag.insert_one_atom_and_incoming_edge(self.atom1, self.atom2, self.edge_base)
            elif self.type == 'delete_one_atom_and_outgoing_edge':
                self.edge_to_delete = self.luciddag.get_edge(self.edge_to_delete)
                self.incoming_edge = self.luciddag.get_edge(self.incoming_edge)
                self.luciddag.delete_one_atom_and_outgoing_edge(self.edge_to_delete, self.incoming_edge)
            elif self.type == 'delete_one_atom_and_incoming_edge':
                self.edge_to_delete = self.luciddag.get_edge(self.edge_to_delete)
                self.incoming_edge = self.luciddag.get_edge(self.incoming_edge)
                self.luciddag.delete_one_atom_and_incoming_edge(self.edge_to_delete, self.outgoing_edge)
            elif self.type == 'delete_one_atom_at_the_end':
                self.edge_to_delete = self.luciddag.get_edge(self.edge_to_delete)
                self.luciddag.delete_one_atom_at_the_end(self.edge_to_delete)
            elif self.type == 'delete_one_atom_at_the_start':
                self.edge_to_delete = self.luciddag.get_edge(self.edge_to_delete)
                self.luciddag.delete_one_atom_at_the_start(self.edge_to_delete)
            elif self.type == 'insert_a_line':
                self.luciddag.insert_a_line(self.line, self.lineno)
            else:
                raise Exception("Unknown transformation type.")
            return self.luciddag
        except AssertionError as error:
            if verbose:
                print("TransformationError: Transformation caused invalid AST.")
            raise Exception(error)
    
    def delete_update_DAG_Px(self, atom_collection, key):
        index = atom_collection.unique_edges.index(key)
        val = self.x[index] - 1
        np.put(self.x, index, val)
        self.Px = compute_prob(self.x)
    
    def insert_update_DAG_Px(self, atom_collection, key):
        keys = [key]
        if isinstance(key, str):
            keys = atom_collection.line_edge_mapping[key]
        
        for key in keys:
            # New edges may not be in the unique_edges list.
            # Choose to not update the score.
            try:
                index = atom_collection.unique_edges.index(key)
                val = self.x[index] + 1
                np.put(self.x, index, val)
            except:
                continue
        self.Px = compute_prob(self.x)

def compute_prob(occs):
    occs = np.array(occs)
    prob = occs / np.sum(occs)
    prob[prob == 0] = 1e-50 # avoid 0, small chance of appearing
    return prob

def count_occurrences(x, uniques):
    """
    Count the frquency of each unique element in x.
    """
    occs = np.zeros(len(uniques))
    for i, a in enumerate(uniques):
        occs[i] = x.count(a)
    return occs

if __name__ == "__main__":
    pass