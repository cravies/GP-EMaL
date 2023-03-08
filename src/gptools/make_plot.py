from deap import gp
import numpy as np
import pygraphviz as pgv
import itertools
from deap.creator import _numpy_array
from typing import NoReturn
import os

#needs to be adjusted. For COIL20 we have nf=4
NUM_FEATURES=4

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

def load_trees(path: str) -> NoReturn:
    """ Load a set of tree files from a directory
    and plot all of them
    Arguments:
    path: relative path (from this file) to the folder to walk
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.tree'):
            f = open(f"{path}/{file}")
            print("~"*30 + file + "~"*30)
            lines = f.readlines()
            for line in lines:
                if ' | ' in line:
                    line = line.split(' | ')
                    tree = line[1]
                    print(tree)

def main(name: str, tree: str, fs: str) -> NoReturn:
    """Plot a given GP tree from an individual.
    Arguments:
    name - name for the .png tree plot we are making
    tree -- the tree (a str) read from the .tree output file
    fs - the functional set for the GP algorithm (a str)
    returns: None
    """ 
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(RealArray, NUM_FEATURES), ProxyArray, "f")

    print("Imported function set: ", fs)
    if 'vadd' in fs:
        pset.addPrimitive(np.add,[ProxyArray,ProxyArray],RealArray,name="vadd")
    if 'vsub' in fs:
        pset.addPrimitive(np.subtract, [ProxyArray, ProxyArray], RealArray, name="vsub")
    if 'vmul' in fs:
        pset.addPrimitive(np.multiply, [RealArray, RealArray], RealArray, name="vmul")
    if 'vdiv' in fs:
        pset.addPrimitive(np_protectedDiv, [RealArray, RealArray], RealArray, name="vdiv")
    if 'sigmoid' in fs:
        pset.addPrimitive(np_sigmoid, [RealArray], RealArray, name="sigmoid")
    if 'relu' in fs:
        pset.addPrimitive(np_relu, [RealArray], RealArray, name="relu")
    if 'abs' in fs:
        pset.addPrimitive(np.abs,[np.ndarray],np.ndarray,name="abs")
    if 'max' in fs:
        pset.addPrimitive(np.maximum, [RealArray, RealArray], RealArray, name="max")
    if 'min' in fs:
        pset.addPrimitive(np.minimum, [RealArray, RealArray], RealArray, name="min")

    #add possible features
    for i in range(1000):
        pset.addTerminal(f"f{i}", RealArray)

    # convert string to gp PrimitiveTree using from_string @classmethod
    # need to do this so we can plot
    tree = gp.PrimitiveTree.from_string(tree, pset)

    nodes, edges, labels = gp.graph(tree)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw(f"{name}.png")

if __name__=="__main__":
    load_trees('./COIL20_pt2/COIL20_run_1')
