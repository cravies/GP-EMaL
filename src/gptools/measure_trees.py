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

#---- constants, i.e function set, cost set, pset object, operator cost dictionary ----
#---- all this explained in paper -----

NUM_FEATURES=4
FS = "vadd,vsub,vmul,vdiv,max,min,relu,sigmoid"
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

#add possible features
for i in range(2000):
    PSET.addTerminal(f"f{i}", RealArray)

COSTS = "sum,sum,prod,prod,exp,exp,exp,exp,exp"
COST_DICT = {k:v for k, v in zip(FS.split(','),COSTS.split(','))}

#---- Complexity Measure ----

def explore_tree_recursive(node_dict, subtree_root, indent, tree, labels, size=None):
    """
    Traverses the tree and returns a dict which associates node indices
    with operator, subtree size, and subtree asymmetry.
    :param node_dict: The returned dictionary
    :param subtree_root: The root node for the subtree. It is an index (int)
    :param indent: The indent string, Starts at ''
    :param tree: The overall DEAP tree object
    :param toolbox: The DEAP toolbox
    :param labels: graph labels 
    :returns: node_dict, a dictionary that associates node indices with their subtree operator,
    size, asymmetry, and children
    
    example node_dict for tree 'f1': {0: ['f1',1,0]}
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
    """
    if node_op[0]!='f':
        print(f"{indent}{node_op}(")
    else:
        print(f"{indent}{node_op}")
    """

    # if we have a feature, just set constant complexity
    if node_op[0]=='f':
        complexity = 2
        size = 1
        asymmetry=0
    else:
        # Else we recursively calculate complexity
        complexity=0
        # also calculate asymmetry by breaking up subtree size into left and right subtree size
        left_comp = 0
        right_comp = 0
        left_size = 0
        right_size = 0
        for i,child in enumerate(children):
            child_index = child[0]
            node_dict = explore_tree_recursive(node_dict, child_index, indent + ' |', tree, labels, size)
            ##print(f"{indent}{node_dict}")
            ##print(f"{indent}{node_dict[child_index]}")
            child_complexity = node_dict[child_index][0]
            child_size = node_dict[child_index][1]
            if i==0:
                left_comp += child_complexity
                left_size += child_size
            else:
                right_comp += child_complexity
                right_size += child_size
        # add asymmetry penalty
        size = left_size + right_size
        asymmetry = abs(left_size - right_size)
        complexity += 2**(asymmetry)-1
        # calculate complexity based on node operation
        cost = COST_DICT[node_op]
        #print(f"op: {node_op}, cost: {cost}")
        if cost.isdigit():
            #we have a numerical cost
            #print(f"{indent}const, {cost}")
            complexity += int(cost) + left_comp + right_comp
        elif cost=='sum':
            #print(f"{indent}add, {left_comp}+{right_comp}")
            complexity += left_comp + right_comp
        elif cost=='prod':
            #print(f"{indent}mul, {left_comp}*{right_comp}")
            complexity += left_comp * right_comp
        elif cost=='exp':
            #print(f"{indent}exp, 2**({left_comp}+{right_comp})")
            complexity = 2**(left_comp + right_comp)
        else:
            raise ValueError("Node cost error.")

    #max out complexity at 1m 
    if complexity>1e6:
        complexity = float("inf")
    
    """
    if node_op[0]!='f':
        print(f"{indent})")
    print(f"{indent}size: {size} asymmetry:{asymmetry} complexity:{complexity}")
    """
    node_dict[subtree_root] = [complexity,size] 
    # sort size dict by node index (i.e key)
    node_dict = dict(sorted(node_dict.items()))
    ##print(f"{indent}{tree[subtree_root].name} subtree size: {size}")
    ##print("~"*30)
    return node_dict



def load_trees(path: str) -> NoReturn:
    """ Load a set of .tree files from a directory
    and measure all of them
    Arguments:
    path: relative path (from this file) to the folder to walk
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.tree'):
            measure_tree(f"{path}/{file}")

def measure_tree(path: str) -> NoReturn:
    """
    Measures the complexity of a specific tree file
    Arguments:
    path: relative path 
    """
    count=0
    complexity=0
    f = open(path)
    print("~"*30 + path + "~"*30)
    lines = f.readlines()
    for line in lines:
        if ' | ' in line:
            line = line.split(' | ')
            tree = line[1]
            complexity += main(f"{count}_out",tree,FS)
            count += 1
    print("~"*30)
    print("Total individual complexity: ",complexity)
    print("~"*30)

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

def main(name: str, tree: str, fs: str) -> NoReturn:
    """
    Calculate the complexity of a given GP tree individual.
    Arguments:
    name - name for the .png tree plot we are making
    tree -- the tree (a str) read from the .tree output file
    fs - the functional set for the GP algorithm (a str)
    returns: total_complexity - copmlexity of tree (float) 
    """ 

    # convert string to gp PrimitiveTree using from_string @classmethod
    # need to do this so we can calculate complexity 
    tree = gp.PrimitiveTree.from_string(tree, PSET)

    nodes, edges, labels = gp.graph(tree)
    node_dict = explore_tree_recursive({}, 0, '', tree, labels)
    total_complexity=node_dict[0][0]
    total_complexity*=scaling_term(tree)
    
    print(f"Tree: | {str(tree)} | complexity: {total_complexity}")
    return total_complexity

if __name__=="__main__":
    measure_tree('./COIL20_pt2/COIL20_run_20/COIL20-0.03500168837096679-70.0.tree')
    load_trees('./COIL20_pt2/COIL20_run_1')
