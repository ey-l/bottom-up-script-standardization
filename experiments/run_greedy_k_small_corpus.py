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
import random
#from memory_profiler import profile

sys.path.append(f'{sys.path[0]}/..')
#print(sys.path)
from lucidscript.ASTDAG import *
from lucidscript.LUCIDDAG import *
from lucidscript.LUCID import *
from run_greedy_k import *

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

    # Randomly select scripts
    for run in range(2, 10):
        
        scripts1 = random.sample(scripts, 10)
        scripts_to_improve = [s for s in scripts if s not in scripts1]
        scripts2 = scripts_to_improve

        # AtomCollection init
        ac = AtomCollection(scripts1, tune=1)
        ac.report()

        dataset_folder = config_dict["dataset"]
        output_dir = PROJ_ROOT / 'exp-addition' / 'small-corpus' / dataset_folder / f'{run}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        run_gkdiv(config_dict["dataset"], ac, scripts2, log_dict, 16, 3, 1, 3, output_dir)