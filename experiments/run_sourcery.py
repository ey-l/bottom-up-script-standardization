import ast
from typing import List, Union, Any, Tuple
import sys
import os
import astpretty
from itertools import combinations, permutations, product, chain
from collections import namedtuple, OrderedDict
import networkx as nx
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time

sys.path.append(f'{sys.path[0]}/..')
#print(sys.path)
from lucidscript.ASTDAG import *
from lucidscript.LUCIDDAG import *
from lucidscript.LUCID import *

"""
This file fixes the file paths in the sourcery data folder.
So we can run run_eval.py on it.
"""

if __name__ == "__main__":
    # Get all file paths
    method = "sourcery"
    dataset = 'house-prices-advanced-regression-techniques'
    verbose = True
    results = {}
    corpus_sets = {}
    scripts = {}
    root = PROJ_ROOT / 'scratch' / method / dataset
    pattern = "*.py"
    scriptpath = ""
    
    # Change the file names to sourcery.py
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.split('.')[-1] == 'py':
                scriptpath = os.path.join(path, name)
                newpath = os.path.join(path, 'sourcery.py')
                os.rename(scriptpath, newpath)

    root = PROJ_ROOT / 'data' / 'lemmatized' / dataset
    pattern = "*.py"
    scriptpath = ""
    
    for path, subdirs, files in os.walk(root):
        frompath = path
        topath = path.replace(f'data/lemmatized', 'scratch/sourcery')
        for name in files:
            if name.split('.')[-1] == 'py':
                scriptpath = os.path.join(frompath, name)
                copypath = os.path.join(topath, name)
                print(scriptpath)
                shutil.copyfile(scriptpath, copypath)
