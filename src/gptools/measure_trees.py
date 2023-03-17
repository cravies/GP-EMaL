from deap import gp
import numpy as np
import pygraphviz as pgv
import itertools
from deap.creator import _numpy_array
from typing import NoReturn
import os
from os import path
import re

FS = "vadd,vsub,vmul,vdiv,max,min,relu,sigmoid,np_if"
COSTS = "sum,sum,prod,prod,exp,exp,exp,exp,exp,exp"
COST_DICT = {k:v for k, v in zip(FS.split(','),COSTS.split(','))}

def tree_stats_iterative(tree):
    stats_dict = {'exp':0, 'prod':0, 'sum':0, 'const':0, 'nodes':0, 'unique_feats':0}
    tree=str(tree)
    for key in COST_DICT:
       stats_dict[COST_DICT[key]] += tree.count(key)
    # count number of constants
    stats_dict['const'] = str(tree).count("f") - str(tree).count("if")
    stats_dict['nodes'] = stats_dict['exp'] + stats_dict['prod'] + stats_dict['sum'] + stats_dict['const']
    stats_dict['unique_feats'] += count_unique_features(tree)
    return stats_dict

def load_trees(path: str, oldnew='new') -> NoReturn:
    """ Load a set of .tree files from a directory
    and measure all of them
    Arguments:
    path: relative path (from this file) to the folder to walk
    oldnew: are we reading from old files? (GPMaLMO) 
    or new files (GPEMaL) cause they have different formats.
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.tree'):
            if oldnew=='new':
                measure_tree_new(f"{path}/{file}")
            else:
                measure_tree_old(f"{path}/{file}")

def measure_tree_new(path: str) -> NoReturn:
    """
    Measures the complexity of a specific tree file
    Arguments:
    path: relative path 
    """
    f = open(path)
    print("~"*30 + path + "~"*30)
    lines = f.readlines()
    for line in lines:
        if ' | ' in line:
            line = line.split(' | ')
            tree = line[1]
            file_line, stats_dict = main(tree,FS)
            print(file_line)

def measure_tree_old(path: str) -> NoReturn:
    """
    Measures the complexity of a specific tree file
    in old GPMaLMO format.
    Arguments:
    path: relative path 
    """
    f = open(path)
    print("~"*30 + path + "~"*30)
    lines = f.readlines()
    for line in lines:
        tree = line.replace('\n','')
        file_line, stats_dict = main(tree,FS)
        print(file_line)

def count_unique_features(tree: str) -> int:
    """
    counts the number of unique features in a tree str
    """
    tree_arr = tree.replace(')',' ').replace('(',' ').replace(',',' ').split()
    tree_arr = [i for i in tree_arr if 'i' not in i and 'f' in i]
    uniq_len = len(list(set(tree_arr)))
    return uniq_len

def main(tree: str, fs: str) -> NoReturn:
    """
    Calculate the complexity of a given GP tree individual.
    Arguments:
    tree -- the tree (a str) read from the .tree output file
    fs - the functional set for the GP algorithm (a str)
    returns: total_complexity - copmlexity of tree (float) 
    """ 
    # convert string to gp PrimitiveTree using from_string @classmethod
    # need to do this so we can calculate complexity 
    stats_dict = tree_stats_iterative(tree)
    treestr = str(tree).replace("'","")
    mystr="~"*30 + "\n" + "tree: " + "\n" + "~"*30 + "\n" + treestr + "\n" 
    mystr+="~"*30 + "\n" + "stats: \n" + str(stats_dict) + "\n" + "~"*30 + "\n"
    return mystr, stats_dict

if __name__=="__main__":
    measure_tree_old('winegpmalmo/1/wine-0.02216947891790294-7.0.tree')
    load_trees('./COIL20_pt2/COIL20_run_1')
