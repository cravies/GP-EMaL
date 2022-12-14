import argparse
import gzip as gz
from pathlib import Path
from deap import gp
from matplotlib import pyplot as plt
from gpmalmo import rundata as rd
from gptools.gp_util import output_ind#, draw_trees
from gptools.read_data import read_data

def update_experiment_data(data, ns):
    dict = vars(ns)
    for i in dict:
        setattr(data, i, dict[i])
        # data[i] = dict[i]

warnOnce = False

def try_cache(rundata, hashable, index=0):
    if index==-1:
        return
    rundata.accesses = rundata.accesses + 1
    res = rundata.fitnessCache[index].get(hashable)
    if rundata.accesses % 1000 == 0:
        """
        print("Caches size: " + str(rundata.stores) + ", Accesses: " + str(rundata.accesses) + " ({:.2f}% hit rate)".format(
            (rundata.accesses - rundata.stores) * 100 / rundata.accesses))
        """
    return res


def cachedError(hashable, errorFunc, rundata, args, kargs, index=0):
    # global accesses
    if (not hasattr(rundata,'fitnessCache')) or (rundata.fitnessCache is None):
        if not rundata.warnOnce:
            print("NO CACHE.")
            rundata.warnOnce = True
        return errorFunc(*args, **kargs)

    res = try_cache(rundata, hashable, index)
    if not res:
        res = errorFunc(*args, **kargs)
        rundata.fitnessCache[index][hashable] = res
        rundata.stores = rundata.stores + 1
    # else:
    return res


def init_data(rd):
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logfile", help="log file path", type=str, default="log.out")
    parser.add_argument("-d", "--dataset", help="Dataset file name", type=str, default="wine")
    parser.add_argument("--dir", help="Dataset directory", type=str,
                        default="/home/lensenandr/datasetsPy/")
    parser.add_argument("-g", "--gens", help="Number of generations", type=int,default=1000)
    parser.add_argument("-od", "--outdir", help="Output directory", type=str, default="./")
    parser.add_argument("--erc", dest='use_ercs', action='store_true')
    parser.add_argument("--zeros", dest='use_zeros', action='store_true')
    parser.add_argument("--neighbours", dest="use_neighbours", action="store_true")
    parser.add_argument("--neighbours-mean", dest="use_neighbours_mean", action="store_true")
    parser.add_argument("--trees", dest="max_trees", type=int)
    parser.add_argument("-ob", "--obj", help="objective (time or size)", type=str, default="size")

    parser.set_defaults(use_ercs=False)
    parser.set_defaults(use_zeros=False)
    parser.set_defaults(use_neighbours=False)
    parser.set_defaults(use_neighbours_mean=False)
    args = parser.parse_args()
    print("ARGS: ",args)
    update_experiment_data(rd, args)
    all_data = read_data("{}{}.data".format(args.dir, args.dataset))
    data = all_data["data"]
    rd.dataset = args.dataset
    rd.num_instances = data.shape[0]
    rd.num_features = data.shape[1]
    rd.labels = all_data["labels"]
    rd.data = data
    rd.data_t = data.T
    rd.objective = args.obj

def final_output(hof, toolbox, logbook, pop, rundata, pset):
    for i,res in enumerate(hof):
        print("feature num: ",i)
        output_ind(res, toolbox, rundata, compress=False)
        #draw_trees(i, res)
    p = Path(rundata.outdir, rundata.logfile + '.gz')
    with gz.open(p, 'wt') as file:
        file.write(str(logbook))
    pop_stats = [str(p.fitness) for p in pop]
    pop_stats.sort()
    hof_stats = [str(h.fitness) for h in hof]
    #accuracy proxy arr for pareto front
    accs = [h.fitness.values[0] for h in hof]
    #second objective arr for pareto front
    second_obj = [h.fitness.values[1] for h in hof]
    plot_pareto_front(accs, second_obj)
    # hof_stats.sort()
    print("POP:")
    print("\n".join(pop_stats))
    print("PF:")
    print("\n".join(hof_stats))

def plot_pareto_front(accs, second_obj):
    """
    Plot the pareto front tradeoff between 
    Neighbourhood structure (accuracy proxy)
    "accs"
    And second objective metric
    "second_obj"
    """
    plt.plot(accs, second_obj)
    plt.xlabel("Accuracy proxy loss")
    plt.ylabel(f"{rd.objective}")
    plt.title(f"Accuracy loss and {rd.objective} \n pareto front")
    plt.tight_layout()
    plt.savefig(f"pareto_front_{rd.objective}.png")
    #write pareto front
    f = open(f"pareto_{rd.objective}.txt", "a")
    f.write("accs: "+str(accs)+"\n")
    f.write(f"{rd.objective}: "+str(second_obj)+"\n")
    f.close()