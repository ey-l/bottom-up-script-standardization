from unicodedata import name
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, namedtuple
from itertools import combinations, permutations, product, chain, starmap
from scipy.stats import entropy
import copy
import csv
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Any, Tuple
import ast
import astpretty
from pathlib import Path

import sys
sys.path.append(sys.path[0] + '/..')
PROJ_ROOT = Path(sys.path[0] + '/..')
from csv_logger import CsvLogger
import logging
LOGFILE_DEFAULT_PATH = PROJ_ROOT / 'scratch' / 'logs' / 'log.csv'
delimiter = ','
LOG_FMT = f'%(asctime)s{delimiter}%(levelname)s{delimiter}%(message)s'
LOG_DATEFMT = '%Y/%m/%d %H:%M:%S'
LOG_MAX_SIZE = 100000000  # 100MB
LOG_HEADER = ['date', 'level']

def write_python_script(dirpath, filename, code, verbose=False):
    """
    Write out the transformed script.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filename = dirpath + filename + '.py'
    with open(filename, 'w') as f:
        f.write(code)
    if verbose:
        print('Writing completed.')
    return filename
    
def compare_AST(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).iteritems():
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_AST(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(starmap(compare_AST, zip(node1, node2)))
    else:
        return node1 == node2

class TooLongError(ValueError):
    pass

def pad_to_length(seq, target_length, padding=None):
    """Extend the sequence seq with padding (default: None) so as to make
    its length up to target_length. Return seq. If seq is already
    longer than target_length, raise TooLongError.

    >>> pad([], 5, 1)
    [1, 1, 1, 1, 1]
    >>> pad([1, 2, 3], 7)
    [1, 2, 3, None, None, None, None]
    >>> pad([1, 2, 3], 2)
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TooLongError: sequence too long (3) for target length 2

    """
    length = len(seq)
    if length > target_length:
        raise TooLongError("sequence too long ({}) for target length {}"
                           .format(length, target_length))
    seq.extend([padding] * (target_length - length))
    return seq

def rank_seq(df, cols):
    "Return ranked dataframe indexes."
    df.sort_index(inplace=True)
    # get the last none nan value for each row
    a = list(df[cols].to_numpy())
    a = [x[~np.isnan(x)] for x in a]
    a = [x[-1] for x in a]
    # make tuples (index, value)
    b = list(zip(list(df.index), a))
    b.sort(key=lambda x: x[1]) #b.sort(reverse=True)
    r = [x[0] for x in b]
    return r, a

def get_RE_result(df, n_top=3):
    #labels = list(df['sequence'].apply(lambda x: ", ".join([f"({y.type} {y.param})" for y in x])).values)
    cols = [x for x in list(df.columns) if 'score' in x]
    df.sort_index(inplace=True)
    seqs = [",".join(z) for z in list(df.filter(regex=("step.")).apply(lambda x: [f"({y.type} {y.param})" for y in x]).values)]
    data = copy.deepcopy(df)
    indexes, values = rank_seq(data, cols)
    labels = [f"{seqs[i]} {str(values[i])}" for i in range(len(seqs))]
    tops = indexes[:n_top] # get indexes
    return data[cols], tops, labels

def plot_sequence_score(data, tops, labels, title):
    colours = sns.color_palette('colorblind')
    markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
    
    fig, ax = plt.subplots()
    for i, row in data.iterrows():
        if i not in tops:
            plt.plot(row, color=colours[0], marker=markers[0], markersize=5, linestyle=':', linewidth=1)
    for i in tops:
        label = labels[i]
        j = tops.index(i)+1
        plt.plot(data.loc[i], color=colours[j], marker=markers[j], markersize=10, linewidth=2, label=label)
    legend = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, prop={'size': 10}, title="sequence")
    ax.grid(linestyle=':', linewidth='2')
    ax.set_title(title)
    
    return fig, ax, legend