import os

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

def show_trees():
    """
    Show complexity of trees 
    at each point in the pareto front.
    """
    for file in os.walk('./'):
        if ".tree" in file:
            print(file)

if __name__=="__main__":
    show_trees()
