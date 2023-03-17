from deap import gp
import numpy as np
import pygraphviz as pgv
import itertools
from deap.creator import _numpy_array
from typing import NoReturn
import os
from os import path

#------- Array type inheritance ----------------

class ProxyArray(_numpy_array):
    pass

class RealArray(ProxyArray):
    pass

#--------- PSET operator definitions -------------

def np_relu(x):
    return x * (x > 0)

def np_protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def np_sigmoid(gamma):
    return expit(gamma)

def np_many_add(a, b, c, d, e):
    return a + b + c + d + e

# https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp
def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))

def np_if(a, b, c):
    return np.where(a < 0, b, c)

#---- constants, i.e function set, cost set, pset object, operator cost dictionary ----
#---- all this explained in paper -----

NUM_FEATURES=4
FS = "vadd,vsub,vmul,vdiv,max,min,relu,sigmoid,np_if"
PSET = gp.PrimitiveSetTyped("MAIN", itertools.repeat(RealArray, NUM_FEATURES), ProxyArray, "f")

if 'vadd' in FS:
    PSET.addPrimitive(np.add,[ProxyArray,ProxyArray],RealArray,name="vadd")
if 'vsub' in FS:
    PSET.addPrimitive(np.subtract, [ProxyArray, ProxyArray], RealArray, name="vsub")
if 'vmul' in FS:
    PSET.addPrimitive(np.multiply, [RealArray, RealArray], RealArray, name="vmul")
if 'vdiv' in FS:
    PSET.addPrimitive(np_protectedDiv, [RealArray, RealArray], RealArray, name="vdiv")
if 'sigmoid' in FS:
    PSET.addPrimitive(np_sigmoid, [RealArray], RealArray, name="sigmoid")
if 'relu' in FS:
    PSET.addPrimitive(np_relu, [RealArray], RealArray, name="relu")
if 'abs' in FS:
    PSET.addPrimitive(np.abs,[np.ndarray],np.ndarray,name="abs")
if 'max' in FS:
    PSET.addPrimitive(np.maximum, [RealArray, RealArray], RealArray, name="max")
if 'min' in FS:
    PSET.addPrimitive(np.minimum, [RealArray, RealArray], RealArray, name="min")
if 'np_if' in FS:
    PSET.addPrimitive(np_if, [RealArray, RealArray, RealArray], RealArray, name="np_if")

#add possible features
for i in range(2000):
    PSET.addTerminal(f"f{i}", RealArray)

COSTS = "sum,sum,prod,prod,exp,exp,exp,exp,exp,exp"
COST_DICT = {k:v for k, v in zip(FS.split(','),COSTS.split(','))}

#---- Complexity Measure ----

def tree_stats_recursive(stats_dict, subtree_root, indent, tree, labels, size=None):
    """
    Traverses the tree and returns a dict which stores statistics 
    for various tree aspects like operator counts, 
    total asymmetry, total node count
    :param stats_dict: The returned dictionary
    :param subtree_root: The root node for the subtree. It is an index (int)
    :param indent: The indent string, Starts at ''
    :param tree: The overall DEAP tree object
    :param toolbox: The DEAP toolbox
    :param labels: graph labels 
    :returns: stats_dict, a dictionary that collects tree stats
    
    example stats_dict for tree 'f1': {'asymmetry':1, 'exp_count':0, 'prod_count':0, 'linear_count':0, const_count:1, node_count:1}
    """

    #Lists for different types of operator
    subtree = tree.searchSubtree(subtree_root)
    this_arity = tree[subtree_root].arity
    children = []
    i = 0
    idx = subtree_root + 1
    while i < this_arity:
        child_slice = tree.searchSubtree(idx)
        children.append([child_slice.start, child_slice.stop])
        i += 1
        idx = child_slice.stop
    node_op = tree[subtree_root].name
    if node_op[0]!='f':
        print(f"{indent}{node_op}(")
        print(f"{indent}children: ",children)
    else:
        print(f"{indent}{node_op}")
    asymmetry = 0
    exp_count = 0
    prod_count = 0
    linear_count = 0
    const_count = 0
    node_count = 1 
    
    # if we have a feature, just set constant complexity
    if node_op[0]=='f':
        const_count = 1
    else:
        # also calculate asymmetry by breaking up subtree size into left and right subtree size
        # recursively get counts from child subtree
        for i,child in enumerate(children):
            #grab child stats dict
            stats_dict_child = {'asymmetry':0, 'exp_count':0, 'prod_count':0, 'linear_count':0, 'node_count':0, 'const_count':0}
            child_index = child[0]
            stats_dict_child = tree_stats_recursive(stats_dict_child, child_index, indent + ' |', tree, labels, size)
            exp_count += stats_dict_child['exp_count']
            prod_count += stats_dict_child['prod_count']
            linear_count += stats_dict_child['linear_count']
            const_count += stats_dict_child['const_count']
            node_count += stats_dict_child['node_count']
        # now we get count from our own node
        if COST_DICT[node_op]=='exp':
            exp_count += 1
        elif COST_DICT[node_op]=='prod':
            prod_count += 1
        elif COST_DICT[node_op]=='sum':
            linear_count += 1
        elif COST_DICT[node_op]=='const':
            const_count += 1
    print(f"{indent}{stats_dict}")
    if node_op[0]!='f':
        print(f"{indent})")
    #store totals
    stats_dict['asymmetry'] += asymmetry
    stats_dict['exp_count'] += exp_count
    stats_dict['prod_count'] += prod_count
    stats_dict['linear_count'] += linear_count
    stats_dict['node_count'] += node_count
    stats_dict['const_count'] += const_count
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

def scaling_term(tree,mu=0.8,max_depth=8):
    """
    Calculate the scaling term for the tree
    Punishes larger trees in a stepwise manner
    Default mu=0.8
    alpha = tree_height/max_height
    S = 1 if alpha < mu
    S = 2*alpha if alpha > mu
    """
    alpha = tree.height / max_depth
    if alpha < mu:
        return 1
    else:
        return 2*alpha

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
    tree_orig = tree
    tree = gp.PrimitiveTree.from_string(tree, PSET)
    nodes, edges, labels = gp.graph(tree)
    stats_dict = {'asymmetry':0, 'exp_count':0, 'prod_count':0, 'linear_count':0, 'const_count':0, 'node_count':0}
    stats_dict = tree_stats_recursive(stats_dict, 0, '', tree, labels)
    treestr = str(tree_orig).replace("'","")
    mystr=f"tree: | {treestr} | stats: {stats_dict}"
    return mystr, stats_dict

if __name__=="__main__":
    #measure_tree_old('winegpmalmo/1/wine-0.02216947891790294-7.0.tree')
    load_trees('./COIL20_pt2/COIL20_run_1')
