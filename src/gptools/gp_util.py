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

def scaling_term(tree,mu=0.8):
    """
    Calculate the scaling term for the tree
    Punishes larger trees in a stepwise manner
    Default mu=0.8
    alpha = tree_height/max_height
    S = 1 if alpha < mu
    S = 2*alpha if alpha > mu
    """
    alpha = tree.height / rd.max_depth
    if alpha < mu:
        return 1
    else:
        return 2*alpha

def plot_log(logbook):
    """
    Takes the run logbook and plots the
    population's median neighbourhood structure loss
    and second objective loss over generations
    :param logbook: The GP run logbook
    """
    try:
        with open(f'{rd.dataset}_run.txt') as f:
            lines = f.readlines()
            line = lines[-1]
            num = line.strip()
            num=int(num)
    except FileNotFoundError:
        # it is the initial run
        # i.e no textfile
        num=1
    #folder to output to
    fname=f"{rd.outdir}/{rd.dataset}_run_{num}/"
    second_obj = rd.objective
    ##print("Chapters: ",logbook.chapters)
    cost_log = logbook.chapters['cost']
    cost_median = [row['median'] for row in cost_log]
    second_obj_log = logbook.chapters[second_obj]
    second_obj_median = [row['median'] for row in second_obj_log]
    plt.plot(cost_median)
    plt.title("median neighbourhood structure cost")
    plt.xlabel("generation")
    plt.ylabel("cost")
    plt.savefig(f"{fname}{rd.dataset}_cost_{num}.png")
    plt.close()
    plt.plot(second_obj_median)
    plt.title(second_obj)
    plt.xlabel("generation")
    plt.ylabel(second_obj)
    plt.savefig(f"{fname}{rd.dataset}_{rd.objective}_{num}.png")
    plt.close()

def output_ind(ind, toolbox, data, suffix="", compress=False, out_dir=rd.outdir, del_old=False):
    """
    For a given individual, make the data embedding, store it in a csv file,
    and output the inidividual trees to a .tree file. 
    Stores it in a folder in output dir out_dir, default ./runs/
    :param ind: the GP Individual. Assumed two-objective
    :param toolbox: To evaluate the tree
    :param data: dict-like object containing data_t (feature-major array), outdir (string-like),
    dataset (name, string-like), labels (1-n array of class labels)
    :param suffix: to go after the ".csv/tree"
    :param compress: boolean, compress outputs or not
    :param out_dir: folder to output to, default "./runs/"
    :param del_old: delete previous generations or not
    """
    old_files = glob.glob(out_dir + "*.tree" + ('.gz' if compress else ''))
    old_files += glob.glob(out_dir + "*.csv" + ('.gz' if compress else ''))
    out = evaluateTrees(data.data_t, toolbox, ind)
    columns = ['C' + str(i) for i in range(out.shape[1])]
    df = pd.DataFrame(out, columns=columns)
    df["class"] = data.labels

    compression = "gzip" if compress else None

    f_name = ('{}' + ('-{}' * len(ind.fitness.values)) + '{}').format(data.dataset, *ind.fitness.values, suffix)

    outfile = f_name + '-aug.csv'
    combined_array = np.concatenate((out, data.data), axis=1)
    aug_columns = columns + ['X' + str(i) for i in range(data.data.shape[1])]
    df_aug = pd.DataFrame(combined_array, columns=aug_columns)
    df_aug["class"] = data.labels
    if compress:
        outfile = outfile + '.gz'
    df_aug.to_csv(out_dir + outfile, index=None, compression=compression)

    # write all trees in indiviudual to tree file 
    with open(out_dir+f_name+".tree", "w") as tree_file:
        # Traverse the tree
        nodes, edges, labels = gp.graph(ind[0])
        #print("~"*30)
        node_dict = explore_tree_recursive({}, 0, '', ind[0], toolbox, labels)
        total_complexity=node_dict[0][0]
        tree_file.write(f"tree: | {str(ind[0])} | ")
        tree_file.write(f"complexity: {total_complexity}")
        for i in range(1, len(ind)):
            tree_file.write('\n')
            tree_file.write(f"tree: | {str(ind[i])} | ")
            # Traverse the tree
            nodes, edges, labels = gp.graph(ind[i])
            #print("~"*30)
            node_dict = explore_tree_recursive({}, 0, '', ind[i], toolbox, labels)
            comp = node_dict[0][0]
            total_complexity += comp
            tree_file.write(f"complexity: {comp}")
        tree_file.write(f"\ntotal complexity: {total_complexity}")
        tree_file.write("\n"+"~"*45+"\n")

    if del_old:
        for f in old_files:
            try:
                os.remove(f)
            except OSError as e:  ## if failed, report it back to the user ##
                print("Error: %s - %s." % (e.filename, e.strerror))

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
        ###print("compiled function: ", str(f))
        func_sympy = human_readable(f)
        ###print("sympy function: ",func_sympy)
        #partial derivatives as callable functions
        pds = grad_tree(func_sympy)
        ###print("partial derivatives: ",pds)
        #compile as executable functions
        pds = [toolbox.compile(expr=str(pd)) for pd in pds]
        #call normal tree
        comp = func(*data_t)
        #get total norm of all partial derivatives at point of data
        pd_norm = [la.norm(pd(*data_t)) for pd in pds]
        ###print("partial derivative norms: ",pd_norm)
        pd_norms.append(pd_norm)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[i] = comp
    dat_array = result.T
    #norm of pd norms
    ###print("Overall pd norms: ",pd_norms)
    TR_term = la.norm(pd_norms)
    ###print("TR term: ",TR_term)

    return TR_term, dat_array

def evaluateTreesFunctional(data_t, toolbox, individual):
    """
    evaluate trees and perform functional complexity 
    evaluation by our in built evaluation complexity metric

    :param data_t: The dataset, transposed. Made by gptools.util.init_data.
    Shape is [num_features, num_instances]
    :param toolbox: The DEAP toolbox object which stores evolutionary operators
    :param individual: A individual that represent an output embedding 
    as an array of trees which each calculate a constructed feature
    from input features. Stored as a deap.creator.Individual    
    """
    ##print(f"individual type: {type(individual)}, shape: {len(individual)}")
    ##print(f"data_t type: {type(data_t)}, shape: {data_t.shape}")
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
        #print("~"*30)
        node_dict = explore_tree_recursive({}, 0, '', tree, toolbox, labels)
        #functional complexity is root node complexity
        f_comp = node_dict[0][0]
        #scale by scaling term to punish larger trees
        scale = scaling_term(tree)
        f_comp *= scale
        #print("~"*30)
        #print("size dict: ",node_dict)
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=str(tree))
        # calculate functional complexity
        #print("func: ",str(tree))
        #print("complexity: ",f_comp)
        f_comp_arr.append(f_comp)
        #evaluate over data
        comp = func(*data_t)
        if (not isinstance(comp, np.ndarray)) or comp.ndim == 0:
            # it decided to just give us a constant back...
            comp = np.repeat(comp, num_instances)
        result[tree_ind] = comp
    dat_array = result.T
    f_comp_total = np.sum(f_comp_arr)
    ###print("total f_comp: ",f_comp_total)
    return f_comp_total, dat_array

def explore_tree_recursive(node_dict, subtree_root, indent, tree, toolbox, labels, size=None):
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
            node_dict = explore_tree_recursive(node_dict, child_index, indent + ' |', tree, toolbox, labels, size)
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
        cost = rd.cost_dict[node_op]
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
