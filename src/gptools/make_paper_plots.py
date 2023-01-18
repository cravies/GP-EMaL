import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_mean_pareto(metric, gens, filename='results.csv'):
    """
    Given a secondary optimisation metric, 
    and the number of generations the algorithm 
    ran for (these are its identifying characteristics)
    
    """
    df = pd.read_csv(filename)
    #remove infinite complexity (invalid solutions)
    infs=['inf',float('inf')]
    df = df[(df.second_objective!=infs[0]) & (df.second_objective!=infs[1])]
    pareto_dict = df.groupby('second_objective')['loss'].apply(list).to_dict()
    #replace with means
    for key in pareto_dict.keys():
        pareto_dict[key] = np.mean(pareto_dict[key])
    loss = list(pareto_dict.keys())
    complexity = list(pareto_dict.values())
    print("loss: ",loss)
    print("complexity: ",complexity)

"""
plot all results with 
a certain metric and generation number
"""

if __name__=="__main__":
   metric="functional"
   gens="3"
   plot_mean_pareto(metric,gens)
