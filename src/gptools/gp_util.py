import glob
import gzip as gz
import math
import os
import random
from pathlib import Path
from deap import gp
import pygraphviz as pgv
from gpmalmo import rundata as rd
import time
import timeit
import sympy
import numpy as np
from numpy import linalg as la
import pandas as pd
from scipy.special._ufuncs import expit
from sympy import sympify
from gptools.multitree import str_ind
from gpmalmo import rundata as rd
from scipy import stats
from matplotlib import pyplot as plt

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
def draw_trees(vnum, ind):
    for fnum,tree in enumerate(ind):
        nodes, edges, labels = gp.graph(tree)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        #feature fnum, version vnum
        #higher vnum will be better at obj2, worse at obj1 and vice versa
        g.draw(f"vnum_{vnum}_feat_{fnum}.png")

def plot_log(logbook):
    """
    Takes the run logbook and plots the
    population's median neighbourhood structure loss
    and second objective loss over generations
    :param logbook: The GP run logbook
    """
    second_obj = rd.objective
    print("Chapters: ",logbook.chapters)
    cost_log = logbook.chapters['cost']
    cost_median = [row['median'] for row in cost_log]
    second_obj_log = logbook.chapters[second_obj]
    second_obj_median = [row['median'] for row in second_obj_log]
    plt.plot(cost_median)
    plt.title("median neighbourhood structure cost")
    plt.xlabel("generation")
    plt.ylabel("cost")
    plt.savefig(f"{rd.dataset}_{rd.gens}_{rd.objective}_cost.png")
    plt.close()
    plt.plot(second_obj_median)
    plt.title(second_obj)
    plt.xlabel("generation")
    plt.ylabel(second_obj)
    plt.savefig(f"{rd.dataset}_{rd.gens}_{rd.objective}_{second_obj}.png")
    plt.close()
    
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
    out = evaluateTrees(data.data_t, toolbox, ind)
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
        total_complexity=functional_complexity(ind[0])
        tree_file.write(f"tree: | {str(ind[0])} | ")
        tree_file.write(f"complexity: {total_complexity}")
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(f"tree: | {str(ind[i])} | ")
            comp = functional_complexity(ind[i])
            total_complexity += comp
            tree_file.write(f"complexity: {comp}")
        tree_file.write(f"\ntotal complexity: {total_complexity}")
        tree_file.write("\n"+"~"*45+"\n")
    else:
        outfile = f_name + '.tree'
        if compress:
            outfile = outfile + '.gz'

        p = Path(data.outdir, outfile)
        with gz.open(p, 'wt') if compress else open(p, 'wt') as file:
            total_complexity = 0
            total_complexity += functional_complexity(ind[0])
            file.write(f"tree: | {str(ind[0])} | ")
            file.write(f"complexity: {total_complexity}")
            for i in range(1, len(ind)):
                file.write('\n')
                file.write(f"tree: | {str(ind[i])} | ")
                comp = functional_complexity(ind[i])
                total_complexity += comp
                file.write(f"complexity: {comp}")
            file.write(f"\ntotal complexity: {total_complexity}")
            file.write("\n"+"~"*45+"\n")
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
        'vdiv': lambda x, y : np_protectedDiv(x,y),
        'vmul': lambda x, y : x * y,
        'vadd': lambda x, y : x + y,
        'max': lambda x,y: max(x,y),
        'min': lambda x,y: min(x,y),
        'abs': lambda x: abs(x),
        'sigmoid': lambda x: np_sigmoid(x),
        'relu': lambda x: np_relu(x)
    }
    expr = sympy.sympify(str(individual), locals=locals)
    return expr

def functional_complexity(tree):
    """
    Evaluate a tree expression's "functional complexity"
    by counting operations
    """
    ratings={
        'vadd':1,'vsub':1,'vmul':1,'vdiv':2, 'max':2,
        'min':2,'np_if':2,'abs':2,'sigmoid':3,'relu':3
    }
    # grab string representation of tree
    expr = str(tree)
    # count number of times each operator occurs
    # add its complexity to total
    total=0
    for key in ratings.keys():
        total += expr.count(key) * ratings[key]
    # punish deeper trees
    total += 1 + tree.height
    return total

def functional_complexity_nested(tree_dict):
    """
    Evaluate a tree expression's "functional complexity"
    by counting operations.
    Punish nested functions exponentially to push
    nonlinear operations towards leaf nodes
    which makes the tree easier to interpret
    """
    ratings={
        'feature':1,
        'vadd':1,'vsub':1,'vmul':1,'vdiv':1, 
        'np_if':2,
        'abs':2, 'max':2, 'min':2,
        'sigmoid':2,'relu':2
    }
    # punish nested functions
    functions=['np_if','abs','max','min','sigmoid', 'relu']
    # count number of times each operator occurs
    # add its complexity to total
    # for "function" operators punish exponentially with subtree size
    total=0
    for node in tree_dict:
        #now add additional penalty for "interpretation complexity"
        op, subtree_size = tree_dict[node]
        if op[0]=='f':
            # we have a feature leaf node, i.e f1, f3, f6
            op='feature'
        if op in functions:
            cost = ratings[op]**(subtree_size-1)
        else:
            cost = ratings[op]
        outstr=f"node: {node} op: {op} subtree size: {subtree_size} complexity: {cost}"
        print(outstr)
        total += cost
    return total

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
    Evaluate trees
    """
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    result = np.zeros(shape=(num_trees, num_instances))

    for tree_ind, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=f)
        #evaluate over data
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[tree_ind] = comp
    dat_array = result.T
    return dat_array

def evaluateTreesTime(data_t, toolbox, individual):
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
    evals=50
    times = np.zeros(shape=(num_trees,evals))

    for tree_ind, f in enumerate(individual.str):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=f)
        # use timeit timer to measure times
        times[tree_ind, :]=timeit.repeat(lambda: func(*data_t), repeat=evals, number=10)
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[tree_ind] = comp
    dat_array = result.T
    #take the trimmed mean eval time for each tree
    #times = stats.trim_mean(times, 0.1, axis=1)
    # UPDATE - take the min.
    times = np.min(times, axis=1)
    #sum minimum execution times for all trees
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
        #print("compiled function: ", str(f))
        func_sympy = human_readable(f)
        #print("sympy function: ",func_sympy)
        #partial derivatives as callable functions
        pds = grad_tree(func_sympy)
        #print("partial derivatives: ",pds)
        #compile as executable functions
        pds = [toolbox.compile(expr=str(pd)) for pd in pds]
        #call normal tree
        comp = func(*data_t)
        #get total norm of all partial derivatives at point of data
        pd_norm = [la.norm(pd(*data_t)) for pd in pds]
        #print("partial derivative norms: ",pd_norm)
        pd_norms.append(pd_norm)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[i] = comp
    dat_array = result.T
    #norm of pd norms
    #print("Overall pd norms: ",pd_norms)
    TR_term = la.norm(pd_norms)
    #print("TR term: ",TR_term)

    return TR_term, dat_array

def evaluateTreesFunctional(data_t, toolbox, individual):
    """
    evaluate trees and perform functional complexity 
    evaluation by our in built evaluation complexity metric
    """
    num_instances = data_t.shape[1]
    num_trees = len(individual)

    if not individual.str or len(individual.str) == 0:
        raise ValueError(individual)

    result = np.zeros(shape=(num_trees, num_instances))

    #functional complexity array for set of trees
    f_comp_arr=[]

    for tree_ind, tree in enumerate(individual):
        # Traverse the tree
        nodes, edges, labels = gp.graph(tree)
        print("~"*30)
        _, size_dict = explore_tree_recursive({}, 0, '', tree, toolbox, labels)
        print("size dict: ",size_dict)
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=str(tree))
        # calculate functional complexity
        print("func: ",str(tree))
        f_comp = functional_complexity_nested(size_dict)
        print("complexity: ",f_comp)
        f_comp_arr.append(f_comp)
        #evaluate over data
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[tree_ind] = comp
    dat_array = result.T
    f_comp_total = np.sum(f_comp_arr)
    #print("total f_comp: ",f_comp_total)
    return f_comp_total, dat_array

def explore_tree_recursive(size_dict, subtree_root, indent, tree, toolbox, labels, size=None):
    """
    Traverses the tree and returns a dict which associates node indices
    with operator and subtree size.

    :param size_dict: The returned dictionary
    :param subtree_root: The root node for the subtree. It is an index (int)
    :param indent: The indent string, Starts at ''
    :param tree: The overall DEAP tree object
    :param toolbox: The DEAP toolbox
    :param labels: graph labels 
    :returns: size_dict, a dictionary that associates node indices with their subtree size and operator
    """
    subtree = tree.searchSubtree(subtree_root)
    print(f"{indent}{tree[subtree_root].name}")
    print("~"*30)
    this_arity = tree[subtree_root].arity
    children = []
    i = 0
    idx = subtree_root + 1
    while i < this_arity:
        child_slice = tree.searchSubtree(idx)
        children.append([child_slice.start, child_slice.stop])
        i += 1
        idx = child_slice.stop

    # recursively get size of subtree
    # start with one to count root of subtree, then recursively apply to children
    size = 1
    for child in children:
        child_size, size_dict = explore_tree_recursive(size_dict, child[0], indent + ' '*2, tree, toolbox, labels, size)
        size += child_size

    size_dict[subtree_root] = [tree[subtree_root].name,size] 
    # sort size dict by node index (i.e key)
    size_dict = dict(sorted(size_dict.items()))

    print(f"{indent}{tree[subtree_root].name} subtree size: {size}")
    print("~"*30)

    return size, size_dict
