from deap import gp
import numpy as np
import pygraphviz as pgv
import itertools
from deap.creator import _numpy_array
from typing import NoReturn
import os
from os import path
import re
import csv

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
    uniq = count_unique_features(tree)
    uniq_len = len(list(set(uniq)))
    stats_dict['unique_feats']=uniq_len
    return uniq, stats_dict

def run_folder(path, oldnew='new'):
    """ Load a set of .tree files from a directory
    and measure all of them
    Arguments:
    path: relative path (from this file) to the folder to walk
    oldnew: are we reading from old files? (GPMaLMO) 
    or new files (GPEMaL) cause they have different formats.
    """
    files = os.listdir(path)
    for file in files:
        fstr=f"{path}/{file}"
        if file.endswith('.tree'):
            if oldnew=='new':
                measure_tree_new(fstr)
            else:
                measure_tree_old(fstr)
        # if a directory, recurse
        if os.path.isdir(fstr):
            print("recursing down: ",fstr)
            run_folder(fstr,oldnew)

def measure_tree_new(path):
    """
    Measures the complexity of a specific tree file
    Arguments:
    path: relative path 
    """
    f = open(path)
    print("~"*30 + path + "~"*30)
    lines = f.readlines()
    unique_feats=[]
    stats_dict = {'exp':0, 'prod':0, 'sum':0, 'const':0, 'nodes':0, 'unique_feats':0}
    for line in lines:
        if ' | ' in line:
            line = line.split(' | ')
            tree = line[1]
            uniq, file_line, stats_dict_tmp = main(tree,FS)
            unique_feats.append(uniq)
            # add to total
            for key in stats_dict.keys():
                stats_dict[key] += stats_dict_tmp[key]
            print(file_line)
    flat_list = [item for sublist in unique_feats for item in sublist]
    print("unique features: ", set(flat_list))
    stats_dict['unique_feats']=len(list(set(flat_list)))
    print("~"*30)
    print("Individual total stats: ",str(stats_dict))
    print("~"*30)
    # add path to stats dict
    stats_dict['filename']=path
    stats_dict['algo']="OLD"
    # write results to csv file
    json_write(stats_dict)

def measure_tree_old(path):
    """
    Measures the complexity of a specific tree file
    in old GPMaLMO format.
    Arguments:
    path: relative path 
    """
    f = open(path)
    print("~"*30 + path + "~"*30)
    lines = f.readlines()
    stats_dict = {'exp':0, 'prod':0, 'sum':0, 'const':0, 'nodes':0, 'unique_feats':0}
    unique_feats=[]
    for line in lines:
        tree = line.replace('\n','')
        uniq, file_line, stats_dict_tmp = main(tree,FS)
        unique_feats.append(uniq)
        for key in stats_dict.keys():
            stats_dict[key] += stats_dict_tmp[key]
        print(file_line)
    flat_list = [item for sublist in unique_feats for item in sublist]
    print("unique features: ", set(flat_list))
    stats_dict['unique_feats']=len(list(set(flat_list)))
    print("~"*30)
    print("Individual total stats: ",str(stats_dict))
    print("~"*30)
    # add path to stats dict
    stats_dict['filename']=path
    stats_dict['algo']="NEW"
    # write results to csv file
    json_write(stats_dict)

def count_unique_features(tree):
    """
    counts the number of unique features in a tree str
    """
    tree_arr = tree.replace(')',' ').replace('(',' ').replace(',',' ').split()
    tree_arr = [i for i in tree_arr if 'i' not in i and 'f' in i]
    return tree_arr

def main(tree, fs):
    """
    Calculate the complexity of a given GP tree individual.
    Arguments:
    tree -- the tree (a str) read from the .tree output file
    fs - the functional set for the GP algorithm (a str)
    returns: total_complexity - copmlexity of tree (float) 
    """ 
    # convert string to gp PrimitiveTree using from_string @classmethod
    # need to do this so we can calculate complexity 
    uniq, stats_dict = tree_stats_iterative(tree)
    treestr = str(tree).replace("'","")
    mystr="~"*30 + "\n" + "tree: " + "\n" + "~"*30 + "\n" + treestr + "\n" 
    mystr+="~"*30 + "\n" + "stats: \n" + str(stats_dict) + "\n" + "~"*30 + "\n"
    return uniq, mystr, stats_dict

def json_write(data_dict, filename='tree_stats.csv'):
    """
    Takes a output dict, and writes it to our 
    output .csv file
    Arguments:
    output_dict: the dictionary whose values we want to write to our CSV file
    output_filename: our output csv filename
    columns:
    exp,prod,sum,const,nodes,unique_feats,filename,algo
    """
    # check if file already exists. If it doesn't, we
    # need to write our columns first
    if os.path.exists(filename):
        # file exists, append data to it
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_dict.values())
    else:
        # file does not exist, write columns and then data
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_dict.keys())
            writer.writerow(data_dict.values())

if __name__=="__main__":
    # EXAMPLES:
    # run on a specific tree file
    #measure_tree_old('winegpmalmo/1/wine-0.02216947891790294-7.0.tree')
    # run on every tree file in a directory
    run_folder('COIL20_pt2/')
    #run_folder('winegpmalmo','old')
    # recurse on COIL20_pt2 folder
    #run_folder('./COIL20_pt2/')
