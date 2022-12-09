import itertools
import math

import numpy as np
from deap import gp
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA

from gptools.gp_util import np_protectedDiv, np_sigmoid, np_relu, np_if, erc_array
from gptools.weighted_generators import ProxyArray, RealArray

def get_pset_weights(data, num_features, rundata):
    num_var_pca = round(math.sqrt(num_features))
    print("PCA vars: " + str(num_var_pca))
    dat_pca = PCA(copy=True, n_components=1)
    dat_pca.fit(data)
    print(dat_pca)
    pc = dat_pca.components_[0]
    # care about magnitude, not direction
    pc = np.abs(pc)
    ranked_pca_features = np.argsort(-pc)  # sorts highest to smallest magnitude
    print(ranked_pca_features)

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(RealArray, num_features), ProxyArray, "f")

    pset.addPrimitive(np.add,[ProxyArray,ProxyArray],RealArray,name="vadd")
    #pset.addPrimitive(np_many_add, [ProxyArray, ProxyArray, ProxyArray, ProxyArray, ProxyArray], RealArray,
    #                  name="vadd")
    pset.addPrimitive(np.subtract, [ProxyArray, ProxyArray], RealArray, name="vsub")
    pset.addPrimitive(np.multiply, [RealArray, RealArray], RealArray, name="vmul")
    """
    Just comment out non sympy differentiable functions because we want to
    take function smoothness measures
    """
    #pset.addPrimitive(np_protectedDiv, [RealArray, RealArray], RealArray, name="vdiv")
    #pset.addPrimitive(np_sigmoid, [RealArray], RealArray, name="sigmoid")
    #pset.addPrimitive(np_relu, [RealArray], RealArray, name="relu")
    #pset.addPrimitive(np.abs,[np.ndarray],np.ndarray,name="abs")
    #pset.addPrimitive(np.maximum, [RealArray, RealArray], RealArray, name="max")
    #pset.addPrimitive(np.minimum, [RealArray, RealArray], RealArray, name="min")
    #pset.addPrimitive(np_if, [RealArray, RealArray, RealArray], RealArray, name="np_if")
    # deap you muppet
    pset.context["array"] = np.array
    num_ercs = math.ceil(num_features / 10)
    # so we get as many as we do terms...
    if rundata.use_ercs:
        print("Using {:d} ERCS".format(num_ercs))
        for i in range(num_ercs):  # range(num_features):
            pset.addEphemeralConstant("rand", erc_array, RealArray)
    weights = {ProxyArray: [], RealArray: []}
    for t in pset.terminals[ProxyArray]:
        weights[ProxyArray].append(t)
        weights[RealArray].append(t)

    if rundata.use_neighbours:
        dm = distance_matrix(data, data)
        rundata.neighbours = dm.argsort()[:, 1:(1 + rundata.num_neighbours)]
        # print(tsnedata.neighbours)
        if rundata.use_neighbours_mean:
            for j in range(num_features):
                feat_vals = getNeighbourFeats(0, j, data, rundata.neighbours)
                for i in range(1, rundata.num_neighbours):
                    feat_vals = feat_vals + getNeighbourFeats(i, j, data, rundata.neighbours)
                    # print(feat_vals)
                feat_vals = np.true_divide(feat_vals, rundata.num_neighbours)
                print(feat_vals)
                name = 'nf{}'.format(j)
                print('Adding ' + name)
                pset.addTerminal(np.copy(feat_vals), RealArray, name=name)
                weights[ProxyArray].append(pset.mapping[name])
                weights[RealArray].append(pset.mapping[name])

        else:
            for i in range(rundata.num_neighbours):
                for j in range(num_features):
                    name = 'n{}f{}'.format(i, j)
                    print('Adding ' + name)
                    pset.addTerminal(np.copy(getNeighbourFeats(i, j, data, rundata.neigbhours)), RealArray, name=name)
                    weights[ProxyArray].append(pset.mapping[name])
                    weights[RealArray].append(pset.mapping[name])
    # don't forget weights
    return pset, weights


def getNeighbourFeats(n_index, f_index, data, neighbours):
    these_neighbours = neighbours[:, n_index]
    return data[these_neighbours, f_index]
