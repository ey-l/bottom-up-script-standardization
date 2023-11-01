from tabnanny import verbose
from turtle import position
from unicodedata import name
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, namedtuple, OrderedDict, defaultdict
from itertools import combinations, permutations, product, chain
from scipy.stats import entropy
from scipy.cluster.vq import whiten, kmeans2
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
from nltk import edit_distance
#from memory_profiler import profile

import sys
sys.path.append(sys.path[0] + '/..')
from lucidscript.ASTDAG import *
from lucidscript.LUCIDDAG import *
from lucidscript.utils import *

PROJ_ROOT = Path(sys.path[0] + '/..')

NOOP_TOKEN = 'NOOP'
RESULT_PATH = PROJ_ROOT / "scratch" / "result.csv"

class Lucid:
    def __init__(self, dataset: str, output_dir, AC: AtomCollection, max_seq_len: int=2, log_config=None, verbose=True):
        self.dataset = dataset
        self.AC = AC
        #self.scripts = scripts
        self.corpus = None
        self.input = None
        self.input_code = None
        self.script_i = None
        self.Qc = None
        self.budget = None
        self.kmeans = None
        self.tries = 1
        self.max_seq_len = max_seq_len
        self.output_dir = output_dir
        self.semantic_freq = 0
        self.verbose = verbose
        self.log_config = log_config
        self._init_logger()
    
    def _init_logger(self):
        # Creat logger with csv rotating handler
        if self.log_config is None:
            self.log_config = {"filepath": LOGFILE_DEFAULT_PATH, "header": LOG_HEADER}
        self.csvlogger = CsvLogger(filename=str(self.log_config["filepath"]), header=self.log_config["header"], 
                            level=logging.NOTSET, add_level_names=[],
                            delimiter=',', add_level_nums=None, fmt=LOG_FMT, datefmt=LOG_DATEFMT, 
                            max_size=LOG_MAX_SIZE, max_files=1)
    
    def _init_Qc(self):
        # Initialize Qc from atom collection
        c = np.array(list(self.AC.edge_vocab.values()) + [0]) # change Q to not include input
        self.Qc = compute_prob(c)

    def set_logger(self, log_config):
        self.log_config = log_config
        self._init_logger()

    def set_budget(self, budget):
        self.budget = budget
    
    def set_kmeans(self, k):
        self.kmeans = k
    
    def set_tries(self, tries):
        self.tries = tries

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def set_semantic_freq(self, freq):
        self.semantic_freq = freq
    
    def set_input(self, input):
        self.input = input
    
    def set_script_i(self, script_i):
        self.script_i = script_i
        
    def naive_exhaustive(self, i: int, current_dag: LUCIDDAG):
        """
        Wrapper for naive-exhaustive algorithm.
        """
        start = time.time()
        self.script_i = i
        self.input = current_dag
        self._init_Qc()
        correctness_filepath = self.init_data_correctness_eval()
        print(correctness_filepath)
        if len(correctness_filepath) == 0:
            print("==========================")
            return None
        
        # GetSteps() and ApplySteps()
        results = self.exaustive_apply_steps(current_dag)
        # GetPromissingTopK() and VerifyCorrectness()
        results = self.exhaustive_get_promissing(current_dag, results)
        results = prep_result_for_write(results, self.max_seq_len)
        df = pd.DataFrame.from_dict(results, orient='index')
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'LUCIDDAG'})
        filename = f"results_{i}.csv"
        df.to_csv(os.path.join(str(self.output_dir), filename), index=False)
        print(df.head())
        
        # Write results
        filename = f"final_{filename}"
        write_to = os.path.join(str(self.output_dir), filename)
        write_result(correctness_filepath, results, write_to)
        self.csvlogger.info(["naive", time.time()-start, "", "", self.script_i])

        if self.verbose:
            print("Done!")

    def exaustive_apply_steps(self, current_dag: LUCIDDAG):
        """
        GetSteps() and ApplySteps().
        """
        i = 0
        current_result = {}

        # Start off by evaluating the current dag
        quality_score = self.compute_quality(current_dag.Px)
        current_result.setdefault(current_dag, {})
        current_result[current_dag]['transformations'] = []
        current_result[current_dag]['quality_score'] = quality_score
        results = {} | current_result
        # For each current dag, generate all possible options
        while i < self.max_seq_len:
            current_dags = list(current_result.keys())
            current_result = {}
            start = time.time()
            for current_dag in current_dags:
                options = self.generate_options(current_dag)
                next_steps = self.rank_options(options)
                fails = 0
                for next_step in next_steps: 
                    try:
                        dag = copy.deepcopy(current_dag)
                        dag = self.apply_next_step(next_step, dag)
                        if dag is not None: 
                            current_result.setdefault(dag, {})
                            current_result[dag]['transformations'] = results[current_dag]['transformations']+[next_step["option"]]
                            current_result[dag]['quality_score'] = next_step["score"]
                    except Exception as err:
                        print("Error in apply_next_step:", err)
                        fails += 1

                self.csvlogger.info(["apply", 0, "total_count", len(options), self.script_i])
                self.csvlogger.info(["apply", 0, "fails_count", fails, self.script_i])
            end = time.time()-start
            self.csvlogger.info(["exhaustive_step", end, "seq_length", i, self.script_i])
            results |= current_result
            i += 1
        # Sort results on quality score
        results = OrderedDict(sorted(results.items(), key=lambda i:i[1]['quality_score']))
        return results
    
    def exhaustive_search_position(self, current_dag, steps):
        body = current_dag.astdag.root.body
        insert_line_steps = [step for step in steps if step.type == "insert_a_line"]
        return list(permutations(range(len(body)), len(insert_line_steps)))

    def exhaustive_get_promissing(self, current_dag: LUCIDDAG, results):
        start = time.time()
        tries = 0
        final_results = {}

        for dag in results.keys():
            semantic = self.check_semantic(dag)
            if semantic:
                final_results.setdefault(dag, {})
                final_results[dag]["transformations"] = results[dag]["transformations"]
                final_results[dag]['quality_score'] = results[dag]["quality_score"]
                final_results[dag]['i'] = tries

        self.csvlogger.info(["exhaustive_get_promissing", time.time()-start, "", "", self.script_i])
        return final_results
    
    def _next_step_wrapper(self, option):
        next_step = {}
        try: 
            x, Px = option.update_Px(atom_collection=self.AC)
            score = self.compute_quality(Px)
            next_step["option"] = option
            next_step["Px"] = Px
            next_step["x"] = x
            next_step["score"] = score
        except Exception as error:
            return None
        return next_step

    #@profile
    def greedy(self, i: int, current_dag: LUCIDDAG, get_promissing, step_function=None):
        """
        TODO: Change this to running one script end-to-end
        """
        start = time.time()
        # Save the code
        self.input = current_dag
        tree = current_dag.astdag.gen_AST()
        self.input_code = ast.unparse(tree)
        self.script_i = i
        self.input.init_Px(self.AC)
        self._init_Qc()
        correctness_filepath = self.init_data_correctness_eval()
        print(correctness_filepath)
        # If no correctness file, skip
        if len(correctness_filepath) == 0:
            print("==========================")
            return None

        results = get_promissing(self.input, step_function)
        results = prep_result_for_write(results, self.max_seq_len)
        df = pd.DataFrame.from_dict(results, orient='index')
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'LUCIDDAG'})
        filename = f"results_{i}.csv"
        df.to_csv(os.path.join(str(self.output_dir), filename), index=False)
        print(df.head())

        # Evaluate data correctness
        for tries, result in enumerate(results):
            try:
                self.eval_data_correctness(i, result, tries)
                results[result]["out"] = "SUCCESS"
                # Compute edit distance
                tree = result.astdag.gen_AST()
                code_to_test = ast.unparse(tree)
                edit_dis0 = edit_distance(self.input_code, '')
                edit_dis1 = edit_distance(self.input_code, code_to_test)
                results[result]["edit_dis0"] = edit_dis0
                results[result]["edit_dis1"] = edit_dis1
            except Exception as ex:
                results[result]["out"] = ex
                results[result]["edit_dis0"] = None
                results[result]["edit_dis1"] = None
                print(ex)
        
        # Write results
        filename = f"final_{filename}"
        print(filename)
        write_to = os.path.join(str(self.output_dir), filename)
        write_result(correctness_filepath, results, write_to)
        self.csvlogger.info(["greedy", time.time()-start, "", "", self.script_i])
        
        if self.verbose:
            print("Done!")
    
    def greedy_tie_get_promissing(self, current_dag: LUCIDDAG, step_function):
        """
        Retain ties at every step.
        """
        length = 0
        results = {}
        current_result = {}
        previous_result = {}

        # Start off by evaluating the current dag
        best_score = self.compute_quality(current_dag.Px)
        current_result.setdefault(current_dag, {})
        current_result[current_dag]['transformations'] = []
        current_result[current_dag]['quality_score'] = best_score
        results.update(current_result)

        # For each current dag, generate all possible options
        while length < self.max_seq_len:
            previous_result = copy.deepcopy(current_result)
            current_result = {}
            if self.semantic_freq > 0:
                check_semantic = (length + 1) % self.semantic_freq == 0
            else: check_semantic = False
            for dag in previous_result:
                next_dags = step_function(dag, check_semantic=check_semantic)
                # Add to the current_result if it's just as good
                for next_dag in next_dags:
                    score = next_dags[next_dag]['score']
                    if score <= best_score:
                        # Reshresh current_result if better score is found
                        if score < best_score: 
                            current_result = {}
                            best_score = score
                        current_result.setdefault(next_dag, {})
                        current_result[next_dag]['transformations'] = previous_result[dag]['transformations'] + [next_dags[next_dag]['option']]
                        current_result[next_dag]['quality_score'] = score
            # Check if current_result gets updated; break if no better score is found
            if len(current_result) > 0:
                results.update(current_result)
                length += 1
            else: break
        
        # Stop when max_length is reached; sort results on quality score
        results = OrderedDict(sorted(results.items(), key=lambda i:i[1]['quality_score']))
        return results

    def greedy_tie_step(self, current_dag: LUCIDDAG, check_semantic):
        best_score = np.inf
        next_dags = {}
        options = self.greedy_generate_options(current_dag)

        # Compute score for each option and put them in next_steps
        next_steps = self.rank_options(options)

        # Extend sequence by one step
        start = time.time()
        best_score = next_steps[0]["score"]
        for next_step in next_steps:
            if next_step["score"] > best_score: break
            # Perform transformation
            next_dag = self.apply_next_step(next_step, current_dag)
            if next_dag is not None: 
                next_dags.setdefault(next_dag, {})
                next_dags[next_dag]['option'] = next_step["option"]
                next_dags[next_dag]['score'] = next_step["score"]
        
        # Check semantic
        if check_semantic:
            candidate_dags = {}
            for i, dag in enumerate(next_dags):
                if self.check_semantic(dag):
                    candidate_dags.setdefault(dag, {})
                    candidate_dags[dag]['option'] = next_steps[i]["option"]
                    candidate_dags[dag]['score'] = next_steps[i]["score"]
            self.csvlogger.info(["greedy_tie_step", time.time()-start, "count", len(options), self.script_i])
            return candidate_dags
        
        self.csvlogger.info(["greedy_tie_step", time.time()-start, "count", len(options), self.script_i])
        return next_dags

    #@profile
    def greedy_k_get_promissing(self, current_dag: LUCIDDAG, step_function):
        """
        Greedy-one algorithm:
            - only optimize score when AddOneStep()
            - record correctness at every step 
            - return the sequence up to the step where correctness is still satisfied 
            - return the original S* if nothing satisfies correctness 
        TODO: Consider using beam search instead.
        NOTE: Beam search will be in a different algorithm.
        """
        length = 0
        results = {}
        current_result = {}
        previous_result = {}

        # Start off by evaluating the current_dag
        best_score = self.compute_quality(current_dag.Px)
        current_result.setdefault(current_dag, {})
        current_result[current_dag]['transformations'] = []
        current_result[current_dag]['quality_score'] = best_score
        results.update(current_result)
        
        # While length permits, try all next possible steps
        # If none of the additional steps decrease the score, stop
        while length < self.max_seq_len:
            previous_result = copy.deepcopy(current_result)
            worst_prev_score = max([previous_result[dag]['quality_score'] for dag in previous_result])
            current_result = {}
            if_updated = False
            if self.semantic_freq > 0:
                check_semantic = (length + 1) % self.semantic_freq == 0
            else: check_semantic = False
            for dag in previous_result:
                options = self.greedy_generate_options(dag)
                # Compute score for each option and put them in next_steps
                next_steps = self.rank_options(options)
                next_dags = step_function(dag, next_steps, check_semantic=check_semantic)
                # UpdateKBeams()
                for next_dag in next_dags:
                    # If budget allows, directly add to current_result
                    if len(current_result) < self.budget and next_dags[next_dag]['score'] < worst_prev_score:
                        current_result.setdefault(next_dag, {})
                        current_result[next_dag]['transformations'] = previous_result[dag]['transformations'] + [next_dags[next_dag]['option']]
                        current_result[next_dag]['quality_score'] = next_dags[next_dag]['score']
                        if_updated = True
                    elif next_dags[next_dag]['score'] < worst_prev_score:
                        # Swap out a worse score
                        current_result = OrderedDict(sorted(current_result.items(), key=lambda i:i[1]['quality_score'], reverse=True))
                        for curr_dag in current_result:
                            if next_dags[next_dag]['score'] < current_result[curr_dag]['quality_score']:
                                del current_result[curr_dag]
                                current_result.setdefault(next_dag, {})
                                current_result[next_dag]['transformations'] = previous_result[dag]['transformations'] + [next_dags[next_dag]['option']]
                                current_result[next_dag]['quality_score'] = next_dags[next_dag]['score']
                                if_updated = True
                                break
            # Early stopping: Check if current_result gets updated; break if no better score is found
            if if_updated:
                results.update(current_result)
                #print("current_result: ", current_result)
                length += 1
            else: break
        
        # Stop when max_length is reached; sort results on quality score
        results = OrderedDict(sorted(results.items(), key=lambda i:i[1]['quality_score']))
        return results
    
    def greedy_search_positions(self, current_dag, steps):
        pos_ls = []
        body = current_dag.astdag.root.body
        length = len(body)
        for step in steps:
            if step.type == "insert_a_line":
                line_pos = self.AC.line_rel_pos[step.line]
                line_pos = list({int(p * length) for p in line_pos})
                pos_ls.append(line_pos)
        return list(product(*pos_ls))
    
    #@profile
    def greedy_k_step(self, current_dag: LUCIDDAG, next_steps, check_semantic=False):
        "Run a step for the greedy algorithm."
        start = time.time()
        semantic = True
        next_dags = {}
        # Try each option in order, based on the budget
        tries = 0
        while tries < self.tries and len(next_dags) < self.budget and tries < len(next_steps):
            next_step = next_steps[tries]
            tries += 1
            # Perform transformation
            next_dag = self.apply_next_step(next_step, current_dag)
            if next_dag is not None: 
                # Check semantic correctness
                if check_semantic:
                    semantic = self.check_semantic(next_dag)
                if semantic:
                    next_dags.setdefault(next_dag, {})
                    next_dags[next_dag]['option'] = next_step["option"]
                    next_dags[next_dag]['score'] = next_step["score"]
        
        self.csvlogger.info(["greedy_k_step", time.time()-start, "count", tries, self.script_i])
        return next_dags
    
    def greedy_divstep_k_step(self, current_dag: LUCIDDAG, next_steps, check_semantic=False):
        """
        KMeans clustering on the next steps.
        """
        # If too small to cluster, run greedy_k_step
        if len(next_steps) < self.kmeans: return self.greedy_k_step(current_dag, next_steps, check_semantic=check_semantic)
        
        start = time.time()
        semantic = True
        next_dags = {}
        X = np.array([next_step["x"] for next_step in next_steps])
        X = whiten(X) # Normalize a group of observations on a per feature basis; rescale each feature dimension of the observation set by its standard deviation
        _, label = kmeans2(X, self.kmeans, minit='points')
        for i, next_step in enumerate(next_steps): next_step["label"] = label[i]
        for i in range(self.kmeans):
            for next_step in next_steps:
                if next_step["label"] == i:
                    # Perform transformation
                    next_dag = self.apply_next_step(next_step, current_dag)
                    if next_dag is not None: 
                        # Check semantic correctness
                        if check_semantic:
                            semantic = self.check_semantic(next_dag)
                        if semantic:
                            next_dags.setdefault(next_dag, {})
                            next_dags[next_dag]['option'] = next_step["option"]
                            next_dags[next_dag]['score'] = next_step["score"]
                            break
        #print("next_dags: ", next_dags)
        #print("len(next_dags): ", len(next_dags))
        self.csvlogger.info(["greedy_k_step", time.time()-start, "count", len(next_steps), self.script_i])
        return next_dags
    
    def greedy_divcon_k_step(self, current_dag: LUCIDDAG, next_steps, check_semantic=False):
        """
        3 groups: 
        """
        start = time.time()
        #semantic = True
        next_dags = {}
        label = [0, 1, 1] # groups
        for next_step in next_steps:
            # Perform transformation
            next_dag = self.apply_next_step(next_step, current_dag)
            if next_dag is not None: 
                # Check semantic correctness
                semantic = self.check_semantic(next_dag)
                if semantic and 1 in label:
                    # TODO: Check data correctness 
                    next_step["label"] = 1
                    label.remove(1)
                    next_dags.setdefault(next_dag, {})
                    next_dags[next_dag]['option'] = next_step["option"]
                    next_dags[next_dag]['score'] = next_step["score"]
                elif not semantic and 0 in label:
                    next_step["label"] = 0
                    label.remove(0)
                    next_dags.setdefault(next_dag, {})
                    next_dags[next_dag]['option'] = next_step["option"]
                    next_dags[next_dag]['score'] = next_step["score"]
            if len(label) == 0: break
        #print("next_dags: ", next_dags)
        #print("len(next_dags): ", len(next_dags))
        self.csvlogger.info(["greedy_k_step", time.time()-start, "count", len(next_steps), self.script_i])
        return next_dags
    
    def apply_next_step(self, next_step, current_dag):
        option = next_step["option"]
        try:
            next_dag = option.apply(luciddag=current_dag, atom_collection=self.AC)
        except Exception as error:
            #print("StepError:", error)
            return None
        next_dag.Px = next_step["Px"]
        next_dag.x = next_step["x"]
        return next_dag

    def rank_options(self, options):
        start = time.time()
        next_steps = []
        # Compute score for each option and put them in next_steps
        for option in options:
            next_step = {}
            try: 
                x, Px = option.update_Px(atom_collection=self.AC)
                score = self.compute_quality(Px)
                next_step["option"] = option
                next_step["Px"] = Px
                next_step["x"] = x
                next_step["score"] = score
                next_steps.append(next_step)
            except Exception as error:
                #print(error)
                continue
        # Sort next_steps by score
        next_steps = sorted(next_steps, key=lambda i:i["score"])
        self.csvlogger.info(["rank_options", time.time()-start, "count", len(options), self.script_i])
        return next_steps

    def generate_insert_line_options(self, current_dag: LUCIDDAG) -> List:
        """
        Generate all possible insert line options based on current state.
        Append options are of the form:
        * (line: str, lineno) - insert a line in the LUCID graph at lineno 
        """
        options = []
        # For each line in linecollection and absent in current_dag
        body = current_dag.astdag.root.body
        #current_lines = [ast.unparse(l) for l in body] # This shouldn't fail
        for line in self.AC.line_vocab:
            # For each body index of current_dag.astdag.root; +1 because we append at the end
            for i in range(len(body)+1):
                # Add an option to insert the line at this position
                options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_a_line', line=line, lineno=i))
            #options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_a_line', line=line, lineno=0))
        return options

    def generate_atom_options(self, current_dag: LUCIDDAG) -> List:
        """
        Generate all possible delete options based on current state.
        Delete options are of the form:
        * (edge_to_delete) - delete the edge A_1 -(e_1')-> A' in the LUCID graph.
        * (edge_to_delete) - delete the edge A' -(e_'2)-> A_2 in the LUCID graph.
        * (edge_to_delete, incoming_edge) - delete the edge in the LUCID graph and add the incoming edge.
        * (edge_to_delete, outgoing_edge) - delete the edge in the LUCID graph and add the outgoing edge.
        """
        options = []
        for atom in current_dag.atoms:
            # Only consider atoms in and after the current line
            if atom.lineno is not None:
                if atom.lineno >= current_dag.last_step_lineno:
                    # In current_dag, find atoms with no outgoing edges.
                    if len(current_dag.get_atom_outgoing_edges(atom)) == 0:
                        for edge in current_dag.get_atom_incoming_edges(atom):
                            # Make a copy of the current dag for each transformation option
                            #dag = copy.deepcopy(current_dag)
                            # Use the edge from the copy
                            # TODO: get rid of deep copy by calculating the RE at construction time and reversing it
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='delete_one_atom_at_the_end', edge_to_delete=edge))
                        
                        # For each of these atoms, find all possible outgoing edge_base in AtomCollection.
                        for edge_base in self.AC.outgoing_edges_vocab.get(atom.base, {}).keys():
                            #dag = copy.deepcopy(current_dag)
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='append_new_edge_and_atom', atom1=atom, edge_base=edge_base))

                    # In current_dag, find atoms with no incoming edges.
                    if len(current_dag.get_atom_incoming_edges(atom)) == 0:
                        for edge in current_dag.get_atom_outgoing_edges(atom):
                            # Make a copy of the current dag for each transformation option
                            #dag = copy.deepcopy(current_dag)
                            # Use the edge from the copy
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='delete_one_atom_at_the_start', edge_to_delete=edge))
                        
                        # For each of these atoms, find all possible incoming edge_base in AtomCollection.
                        for edge_base in self.AC.incoming_edges_vocab.get(atom.base, {}).keys():
                            #dag = copy.deepcopy(current_dag)
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_new_edge_and_atom', atom2=atom, edge_base=edge_base))
        
        # In current_dag, find all triplets of atoms A_1 -(e_1')-> A' -(e_'2)-> A_2
        for edge1 in current_dag.atom_flow_edges:
            atom1 = edge1._s
            atom_to_delete = edge1._e
            # Only consider atoms in and after the current line
            if atom1.lineno is not None and atom_to_delete.lineno is not None:
                if atom1.lineno >= current_dag.last_step_lineno and atom_to_delete.lineno >= current_dag.last_step_lineno:
                    for edge2 in current_dag.get_atom_outgoing_edges(atom_to_delete):
                        atom2 = edge2._e
                        for edge_base in self.AC.outgoing_edges_vocab.get(atom1.base, {}).keys():
                            if edge_base._e == atom2.base:
                                # In AtomCollection, check if A_1 -(e_1')-> A_2 is a valid edge.
                                if edge_base == edge1.base:
                                    # Make a copy of the current dag for each transformation option
                                    #dag = copy.deepcopy(current_dag)
                                    options.append(Transformation(x=copy.deepcopy(current_dag.x), type='delete_one_atom_and_outgoing_edge', edge_to_delete=edge2, incoming_edge=edge1))
                                # In AtomCollection, check if A_1 -(e_'2)-> A_2 is a valid edge.
                                if edge_base == edge2.base:
                                    # Make a copy of the current dag for each transformation option
                                    #dag = copy.deepcopy(current_dag)
                                    options.append(Transformation(x=copy.deepcopy(current_dag.x), type='delete_one_atom_and_incoming_edge', edge_to_delete=edge1, outgoing_edge=edge2))
            
            atom1 = edge1._s
            atom2 = edge1._e
            # Only consider atoms in and after the current line
            if atom1.lineno is not None and atom2.lineno is not None:
                if atom1.lineno >= current_dag.last_step_lineno and atom2.lineno >= current_dag.last_step_lineno:
                    # In AtomCollection, find an A' such that A_1 -(e_1')-> A' -(e_12)-> A_2
                    for edge_base in self.AC.outgoing_edges_vocab.get(atom1.base, {}).keys():
                        atom_candidate = edge_base._e
                        candidate_edges = self.AC.outgoing_edges_vocab.get(atom_candidate, {}).keys()
                        # Check again if edge (e_12) in A' -(e_12)-> A_2 is the same as edge (e_12) in A_1 -(e_12)-> A_2
                        if any([e._e == atom2.base and e.access == edge1.access for e in candidate_edges]):
                            # Make a copy of the current dag for each transformation option
                            #dag = copy.deepcopy(current_dag)
                            #dag_e = dag.get_edge_between_atoms(atom1, atom2) # Use the atoms from the copy
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_one_atom_and_incoming_edge', atom1=atom1, atom2=atom2, edge_base=edge_base))
                    # In AtomCollection, find an A' such that A_1 -(e_12)-> A' -(e_2')-> A_2
                    for edge_base in self.AC.incoming_edges_vocab.get(atom2.base, {}).keys():
                        atom_candidate = edge_base._s
                        candidate_edges = self.AC.incoming_edges_vocab.get(atom_candidate, {}).keys()
                        # Check again if edge (e_12) in A_1 -(e_12)-> A' is the same as edge (e_12) in A_1 -(e_12)-> A_2
                        if any([e._s == atom1.base and e.access == edge1.access for e in candidate_edges]):
                            # Make a copy of the current dag for each transformation option
                            #dag = copy.deepcopy(current_dag)
                            #dag_e = dag.get_edge_between_atoms(atom1, atom2) # Use the atoms from the copy
                            options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_one_atom_and_outgoing_edge', atom1=atom1, atom2=atom2, edge_base=edge_base))
        return options

    #@profile
    def greedy_generate_options(self, current_dag: LUCIDDAG) -> list:
        """
        Generate options according to current_dag.
        """
        options = []
        start = time.time()
        line_options = []
        # For each line in linecollection and absent in current_dag
        body = current_dag.astdag.root.body
        length = len(body)
        #current_lines = [ast.unparse(l) for l in body] # This shouldn't fail
        for line in self.AC.line_vocab:
            #if line not in current_lines:
            line_pos = self.AC.line_rel_pos[line]
            line_pos = list(set([int(p * length) for p in line_pos if p > current_dag.last_step_lineno/length])) # Only insert after lineno; added for monotonicity
            for p in line_pos:
                line_options.append(Transformation(x=copy.deepcopy(current_dag.x), type='insert_a_line', line=line, lineno=p))
        options.extend(line_options)
        self.csvlogger.info(["generate_line_options", time.time() - start, "count", len(line_options), self.script_i])

        start = time.time()
        atom_options = self.generate_atom_options(current_dag)
        options.extend(atom_options)
        self.csvlogger.info(["generate_atom_options", time.time() - start, "count", len(atom_options), self.script_i])

        if self.verbose:
            print(f"finished generating {len(options)} options")
        return options
    
    def generate_options(self, current_dag: LUCIDDAG) -> list:
        """
        Generate options according to current_dag.
        """
        options = []
        start = time.time()
        insert_lines = self.generate_insert_line_options(current_dag)
        options.extend(insert_lines)
        self.csvlogger.info(["generate_line_options", time.time() - start, "count", len(insert_lines), self.script_i])

        start = time.time()
        deletes = self.generate_atom_options(current_dag)
        options.extend(deletes)
        self.csvlogger.info(["generate_atom_options", time.time() - start, "count", len(deletes), self.script_i])

        if self.verbose:
            print(f"finished generating {len(options)} options")
        return options

    def compute_quality(self, Px) -> float:
        """
        Compute the relative entropy, doesn't include the input graph.
        """
        start = time.time()
        # Compute Qc from the corpus and input
        #uniques = list(self.AC.edge_vocab.keys()) + [NOOP_TOKEN]
        if self.Qc is None:
            c = np.array(list(self.AC.edge_vocab.values()) + [0]) # change Q to not include input
            self.Qc = compute_prob(c)

        # Compute P from the input
        #base_start = time.time()
        #self.csvlogger.info(["get_edge_base", time.time()-base_start, "", "", self.script_i])

        # Compute relative entropy
        score = np.log(Px/self.Qc) * Px #* length
        score[~np.isfinite(score)] = 0 # Remove NaNs and infs
        score = round(np.sum(score), 4)
        
        #self.csvlogger.info(["compute_quality", time.time()-start, "", "", self.script_i])
        return score
    
    def init_data_correctness_eval(self, input=None):
        if input is None:
            input = self.input
        # Create dirpaths
        output_dir = str(self.output_dir)
        input_dfs_dir = output_dir+f"/input_dfs_{self.script_i}/"
        if not os.path.exists(input_dfs_dir):
            os.makedirs(input_dfs_dir)
        corr_fp = output_dir+f'/correstness_{self.script_i}.csv'
        # Get result data from input
        input_filenames = self.input.save_processed_dataframe(input_dfs_dir)
        input_dfs_dir = "\"" + input_dfs_dir + "\""
        dataset = "\"" + self.dataset + "\""
        if input_filenames is None:
            return ""

        # Initialize correctness.csv
        if os.path.exists(corr_fp):
            os.remove(corr_fp)
        with open(corr_fp, 'a', newline='') as f:
            writer = csv.writer(f)
            columns = ["i"]
            for fn in input_filenames:
                columns.extend([fn+"_table_jaccard", fn+"_table_EMD", fn+"_model_performance"])
            writer.writerow(columns)
        corr_filepath = "\"" + str(corr_fp) + "\""

        # Add paths to correctness measure function
        func_line = f"compute_corr_measures({dataset}, {input_dfs_dir}, {input_filenames}, {corr_filepath}, idx, updated_dataframes)"
        self.corr_lines = copy.deepcopy(CORR_LINES)
        self.corr_lines.append(func_line)
        return corr_fp

    def check_semantic(self, dag: LUCIDDAG):
        """
        Only does the semantic correctness check -- attempts to run the program.
        """
        start = time.time()
        temp_scripts_dir = str(self.output_dir)+"/scripts/"
        try:
            tree = dag.astdag.gen_AST()
            code = ast.unparse(tree)
            filename = write_python_script(temp_scripts_dir, "result_temp", code)
            exec(open(filename).read())
            self.csvlogger.info(["check_semantic", time.time()-start, "", "", self.script_i])
            return True
        except Exception as err:
            self.csvlogger.info(["check_semantic", time.time()-start, "", "", self.script_i])
            return False

    def eval_data_correctness(self, i:int, dag: LUCIDDAG, tries: int):
        """
        Insert needed code to the transformed scripts to record data correctness.
        :param i: index of the script, used to join with saved transformation sequences.
        """
        # Evaluate correctness for each output_dag
        temp_scripts_dir = str(self.output_dir)+f"/scripts_{i}/"
        start = time.time()
        corr_lines = copy.deepcopy(self.corr_lines)
        corr_lines.insert(-2, f"idx = {tries}")
        tree = dag.astdag.gen_AST()
        tree = ASTDAG.insert_lines(IMPORT_LINES, tree=tree)
        tree = ASTDAG.append_lines(corr_lines, tree=tree)
        code = ast.unparse(tree)
        filename = write_python_script(temp_scripts_dir, "result_"+str(tries), code)
        
        # Try execute
        try:
            exec(open(filename).read())
        except Exception as err:
            self.csvlogger.info(["data_correctness", time.time()-start, "", "", self.script_i])
            raise Exception('SemanticError', err)
        self.csvlogger.info(["data_correctness", time.time()-start, "", "", self.script_i])

def write_result(correctness_filepath, result, write_to):
    """
    Join result with correctness. Write the merged result.
    """
    # Result to dataframe
    df = pd.DataFrame.from_dict(result, orient='index')
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'LUCIDDAG'})
    print(df.head())
    #df.index = range(len(result)) # If we don't want to save result dags

    # Load correctness
    correctness = pd.read_csv(correctness_filepath, index_col=0)
    print(correctness.head())
    try:
        df = df.join(correctness, how='left')
        df.to_csv(write_to, index=False)
    except Exception as ex:
        print(ex)

def prep_result_for_write(results, length):
    """
    Transform results to a more usable format
    """
    # Pad sequence
    for dag in results:
        results[dag]["step_no"] = len(results[dag]["transformations"])
        results[dag]["transformations"] = pad_to_length(results[dag]["transformations"], length, "")
    
    updated = copy.deepcopy(results)
    for dag in updated:
        seq = updated[dag]["transformations"]
        for i, step in enumerate(seq):
            updated[dag][f"step_{i}"] = step
    return updated

if __name__ == "__main__":
    pass