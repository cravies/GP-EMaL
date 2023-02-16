import cachetools
from defaultlist import defaultlist

num = None
dataset = None
data = None
data_t = None
labels = None
outdir = None
pairwise_distances = None
ordered_neighbours = None
neighbours = None
all_orderings = None
identity_ordering = None
nobj = 3 
fitnessCache = defaultlist(lambda: cachetools.LRUCache(maxsize=1e6))
accesses = 0
stores = 0

max_depth = 8
max_height = 14
pop_size = 100
cxpb = 0.7
mutpb = 0.15
mutarpb = 0.15
num_trees = 34
gens = 1000
objective = "size"

num_instances = 0
num_features = 0

function_set=['vadd','vsub','vmul','vdiv',
            'relu','sigmoid','max','min','abs']
"""
sum: complexity of a node is the sum of the complexity
of its left and right children subtrees
prod: complexity of a node is the product of the 
complexity of its left and right children subtrees
exp: complexity of a node is 2**(L+R) where L is the 
complexity of its left child subtree and R is the 
complexity of its right child subtree
"""
op_costs=['sum','sum','prod','prod',
        'exp','exp','exp','exp','exp']
