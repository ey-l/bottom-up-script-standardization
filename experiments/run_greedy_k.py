import ast
import contextlib
from typing import List, Union, Any, Tuple
import sys
import os
from itertools import combinations, permutations, product, chain
from collections import namedtuple, OrderedDict
import networkx as nx
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
#from memory_profiler import profile

sys.path.append(f'{sys.path[0]}/..')
#print(sys.path)
from lucidscript.ASTDAG import *
from lucidscript.LUCIDDAG import *
from lucidscript.LUCID import *

def js_reader(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

#@profile
def load_DAGs(root):
    # Get all file paths
    verbose = True
    luciddags = []
    #print(sys.path)
    for path, subdirs, files in os.walk(root):
        for name in files:
            # Load scripts
            if name.split('.')[-1] == 'py' and name.split('.')[0] != "codex":
                with open(os.path.join(path, name), encoding='utf-8') as f:
                    code = f.read()
                    code = code.replace("_data/input/", "data/input/")
                    code = code.replace("data/input/", f"{sys.path[-1]}/data/input/")
                with contextlib.suppress(Exception):
                    tree = ast.parse(code)
                    #print(code)
                    d = ASTDAG(root=tree, filename=name)
                    d.gen_DAG()
                    luciddags.append(LUCIDDAG(d))
    if verbose:
        print("Number of scripts:", len(luciddags))
    return luciddags

def run_gkdiv(dataset:str, ac: AtomCollection, scripts: List[LUCIDDAG], log_dict: dict, seq_len: int, budget: int, semantic_on: bool, kmeans:int, output_dir: Path):
    # Standardize one script end-to-end
    for i, dag in enumerate(scripts):
        lucid = Lucid(dataset=dataset, output_dir=output_dir, AC=ac)
        log_dict["filepath"] = output_dir / f'runtime_{i}.csv'
        if os.path.exists(log_dict["filepath"]):
            os.remove(log_dict["filepath"])
        lucid.set_max_seq_len(seq_len)
        lucid.set_semantic_freq(semantic_on)
        lucid.set_logger(log_dict)
        lucid.set_budget(budget)
        lucid.set_kmeans(kmeans)
        lucid.set_tries(np.inf)

        lucid.greedy(i, dag, lucid.greedy_k_get_promissing, lucid.greedy_divstep_k_step)
        del lucid
        print("============= Done with script =============")
        #gc.collect()

def run_gk(dataset: str, ac: AtomCollection, scripts: List[LUCIDDAG], log_dict: dict, seq_len: int, budget: int, semantic_on: bool, kmeans:int, output_dir: Path):
    # Standardize one script end-to-end
    for i, dag in enumerate(scripts):
        lucid = Lucid(dataset=dataset, output_dir=output_dir, AC=ac)
        log_dict["filepath"] = output_dir / f'runtime_{i}.csv'
        if os.path.exists(log_dict["filepath"]):
            os.remove(log_dict["filepath"])
        lucid.set_max_seq_len(seq_len)
        lucid.set_semantic_freq(semantic_on)
        lucid.set_logger(log_dict)
        lucid.set_budget(budget)
        lucid.set_kmeans(kmeans)
        lucid.set_tries(np.inf)
        
        lucid.greedy(i, dag, lucid.greedy_k_get_promissing, lucid.greedy_k_step)
        del lucid
        print("============= Done with script =============")

if __name__ == "__main__":
    config_file = sys.argv[1]
    config_dict = js_reader(config_file)

    print(config_dict)

    output_dir = PROJ_ROOT / 'scratch' / 'greedy_one'
    root = PROJ_ROOT / 'data' / 'lemmatized' / config_dict["dataset"]

    # Record time
    filepath = output_dir / 'stats.csv'
    columns = ["function", "time", "var", "val", "script_i"]
    log_dict = {"filepath": filepath, "header": LOG_HEADER + columns}
    scripts = load_DAGs(root)
    #scripts = scripts[:5]

    for seq_len in config_dict["seq_len"]:
        for tune in config_dict["tune"]:
            # AtomCollection init
            ac = AtomCollection(scripts, tune=tune)
            ac.report()

            for budget in config_dict["budget"]:
                for semantic_on in config_dict["semantic_on"]:
                    method_folder = config_dict["method"]
                    dataset_folder = config_dict["dataset"]
                    kmeans = config_dict["kmeans"]
                    output_dir = PROJ_ROOT / 'exp' / dataset_folder / method_folder / f'seq_{str(seq_len)}' / f'tune_{str(tune)}' / f'budget_{str(budget)}' / f'sem_{str(semantic_on)}'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    if config_dict["method"] == "lsd":
                        run_gkdiv(config_dict["dataset"], ac, scripts, log_dict, seq_len, budget, semantic_on, kmeans, output_dir)
                    elif config_dict["method"] == "lsv":
                        run_gk(config_dict["dataset"], ac, scripts, log_dict, seq_len, budget, semantic_on, kmeans, output_dir)
                    else:
                        print("Method not found.")