import glob
import gzip as gz
import math
import os
import random
from pathlib import Path
from deap import gp
#import pygraphviz as pgv
from gpmalmo import rundata as rd
import time
import sympy
import numpy as np
from numpy import linalg as la
import pandas as pd
from scipy.special._ufuncs import expit
from sympy import sympify
from gptools.multitree import str_ind
from gpmalmo import rundata as rd

def protectedDiv(left, right):
    if right == 0:
        return 1
    return left / right


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


def np_relu(x):
    return x * (x > 0)


def relu(x):
    # fast? https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    return x * (x > 0)


def _if(a, b, c):
    return b if a < 0 else c


def np_if(a, b, c):
    return np.where(a < 0, b, c)


# np...??


def erc_array():
    return random.uniform(-1, 1)


dat_set = set()


def add_to_string_cache(ind):
    hash = str_ind(ind)
    dat_set.add(hash)
    ind.str = hash


def check_uniqueness(ind1, ind2, num_to_produce, offspring):
    # deals with the case where we needed to create two individuals, but had 4 from TS, and the first two were okay,
    if len(offspring) != num_to_produce:
        hash1 = str_ind(ind1)
        if hash1 not in dat_set:
            dat_set.add(hash1)
            ind1.str = hash1
            offspring.append(ind1)
    # the case where we only needed to create 1 more individual, not 2!
    if len(offspring) != num_to_produce:
        hash2 = str_ind(ind2)
        if hash2 not in dat_set:
            dat_set.add(hash2)
            ind2.str = hash2
            offspring.append(ind2)

"""
The GPMaLMO algorithm takes a set of n input features
and constructs a set of m GP trees, where m < n.
These GP trees take the input features and construct output features
in the (m dimensional) embedding. 
Because GPMaLMO optimises two objectives, the possible GP tree solutions
form a pareto front, which is a set of GP trees with varying scores on the two 
objectives. If there were only 3 in the pareto front, it would look like this.
[[good obj 1, bad obj 2], [ok obj 1, ok obj 2], [bad obj 1, good obj 2]]
You can see this gives us the range of possible outcomes. each entry in the pareto front is 
pareto optimal in the sense that you can't improve its overall score because to increase 
its score on one objective would be to decrease it on another.

INPUT:
Ind is an "individual": a point in the pareto front. (point given by vnum, i.e "version" number)
For example if we have an input dataset of dimensionality 3 (as in the iris dataset)
ind will be of length 2 because we want to make two constructed features of the 3 input features.
Each entry in the pareto front is a candidate for the GP tree that will make that constructed feature.
We save a picture of each of the constructed feature trees
"""
"""
def draw_trees(vnum, ind):
    obj = rd.objective
    print("Objective: ",obj)    
    for fnum,tree in enumerate(ind):
        print("vnum: ", vnum)
        nodes, edges, labels = gp.graph(tree)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        #feature fnum, version vnum, second objective obj
        #higher vnum will be better at obj2, worse at obj1 and vice versa
        g.draw(f"feat_{fnum}_ver_{vnum}_obj_{obj}.png")
"""

def output_ind(ind, toolbox, data, suffix="", compress=False, csv_file=None, tree_file=None, del_old=False):
    """ Does some stuff

    :param ind: the GP Individual. Assumed two-objective
    :param toolbox: To evaluate the tree
    :param data: dict-like object containing data_t (feature-major array), outdir (string-like),
    dataset (name, string-like), labels (1-n array of class labels)
    :param suffix: to go after the ".csv/tree"
    :param compress: boolean, compress outputs or not
    :param csv_file: optional path/buf to output csv to
    :param tree_file: optional path/buf to output tree to
    :param del_old: delete previous generations or not
    """
    old_files = glob.glob(data.outdir + "*.tree" + ('.gz' if compress else ''))
    old_files += glob.glob(data.outdir + "*.csv" + ('.gz' if compress else ''))
    time_val, out = evaluateTrees(data.data_t, toolbox, ind)
    columns = ['C' + str(i) for i in range(out.shape[1])]
    df = pd.DataFrame(out, columns=columns)
    df["class"] = data.labels

    compression = "gzip" if compress else None

    f_name = ('{}' + ('-{}' * len(ind.fitness.values)) + '{}').format(data.dataset, *ind.fitness.values, suffix)

    if csv_file:
        df.to_csv(csv_file, index=None)
    else:
        outfile = f_name + '.csv'
        if compress:
            outfile = outfile + '.gz'
        p = Path(data.outdir, outfile)
        df.to_csv(p, index=None, compression=compression)

    outfile = f_name + '-aug.csv'
    combined_array = np.concatenate((out, data.data), axis=1)
    aug_columns = columns + ['X' + str(i) for i in range(data.data.shape[1])]
    df_aug = pd.DataFrame(combined_array, columns=aug_columns)
    df_aug["class"] = data.labels
    if compress:
        outfile = outfile + '.gz'
    p = Path(data.outdir, outfile)
    df_aug.to_csv(p, index=None, compression=compression)

    if tree_file:
        tree_file.write(str(ind[0]))
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(str(ind[i]))
    else:
        outfile = f_name + '.tree'
        if compress:
            outfile = outfile + '.gz'

        p = Path(data.outdir, outfile)
        with gz.open(p, 'wt') if compress else open(p, 'wt') as file:
            file.write(str(ind[0]))
            for i in range(1, len(ind)):
                file.write('\n')
                file.write(str(ind[i]))

    if del_old:
        for f in old_files:
            try:
                os.remove(f)
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))

def human_readable(individual):
    """ convert a candidate function into a human readable sympy expression""" 
    #dictionary to translate deap output to human readable
    #using sympy
    locals = {
        'vsub': lambda x, y : x - y,
        'vdiv': lambda x, y : x/y if y!=0 else 1,
        'vmul': lambda x, y : x*y,
        'vadd': lambda x, y : x + y,
    }
    expr = sympy.sympify(str(individual), locals=locals)
    return expr

def grad_tree(expr):
    """
    calculate all the partial derivatives of the sympy 
    expression for a tree function
    w.r.t features,
    f0, f1,..., fn
    i.e calculate the gradient w.r.t the input features
    g='f0*f1 + f1*f2 + cos(f1)'
    -> 
    grad_tree(g) = [f1,f0+f2-sin(f1),f1]
    """
    #basic tests
    feats=["f{}".format(i) for i in range(0,rd.num_features)]
    grad_tree=[sympy.diff(expr,feat) for feat in feats]
    return grad_tree

def evaluateTrees(data_t, toolbox, individual):
    """
    Evaluate trees,
    also return time taken
    """
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    result = np.zeros(shape=(num_trees, num_instances))

    # time each tree eval times, take median eval time, 
    # sum together for total median eval time
    evals=30
    times = np.zeros(shape=(num_trees,evals))

    for tree_ind, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=f)
        #evaluate eval times, take median eval time
        for eval_iter in range(evals):
            #time tree evaluation
            time_st = time.perf_counter()
            comp = func(*data_t)
            times[tree_ind,eval_iter]=float(time.perf_counter() - time_st)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[tree_ind] = comp
    dat_array = result.T
    #take the median eval time for each tree
    times = np.median(times,axis=1)
    #sum median execution times for all trees
    time_val = np.sum(times)
    return time_val, dat_array

def evaluateTreesTR(data_t, toolbox, individual):
    """
    evaluate trees and perform Tikhonov Regularisation, 
    i.e calculate total norm of partial derivatives
    of GP trees w.r.t input features. 
    We will then minimise this term as a secondary objective
    """
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    result = np.zeros(shape=(num_trees, num_instances))

    #partial derivative L2 norms
    pd_norms = []

    for i, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=f)
        func_sympy = human_readable(f)
        print("sympy function: ",func_sympy)
        #partial derivatives as callable functions
        pds = grad_tree(func_sympy)
        print("partial derivatives: ",pds)
        #compile as executable functions
        pds = [toolbox.compile(expr=str(pd)) for pd in pds]
        #call normal tree
        comp = func(*data_t)
        #get total norm of all partial derivatives at point of data
        pd_norm = [la.norm(pd(*data_t)) for pd in pds]
        pd_norms.append(pd_norm)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[i] = comp
    dat_array = result.T
    #norm of pd norms
    TR_term = la.norm(pd_norms)

    return TR_term, dat_array