import numpy as np
from numba import jit
from Utils import dataloader as dl

# @jit(nopython=True)
def isClique(nodes: np.array, adjmat: np.array):
    """
    Verify the current selection of nodes constitutes a clique
    method for checking edges from:
    https://stackoverflow.com/questions/48323576/check-if-all-rows-in-one-array-are-present-in-another-bigger-array
    :param nodes:
    :param adjmat:
    :return:
    """
    edgelist = np.ascontiguousarray(dl.nodelist_to_edgelist(nodes,adjmat))
    if edgelist.shape == (0,): return False

    # Check that each row in cliqueList is also in the actual edgelist
    cliquelist = np.ascontiguousarray(dl.nodelist_to_edgelist(nodes, 1 - np.eye(len(adjmat))))
    void_dt = np.dtype((np.void, edgelist.dtype.itemsize * edgelist.shape[1]))
    edgelist, cliquelist =  edgelist.view(void_dt).ravel(), cliquelist.view(void_dt).ravel()
    return np.in1d(cliquelist, edgelist).sum() == len(cliquelist)
    # idxnodes = np.where(nodes == 1)[0]
    # iternodes = itertools.combinations(idxnodes, 2)
    # for iter in iternodes:
    #     if adjmat[iter[0],iter[1]] == 0: return False
    # return True

@jit(nopython=True)
def get_edgecount(adjmat: np.array):
    edgecount = np.zeros(len(adjmat))
    for i in range(len(edgecount)):
        edgecount[i] = adjmat[i].sum()
    return edgecount

@jit(nopython=True)
def BackKhuriFitness(ind: np.array, adjMat: np.array):
    """
    Proposed fitness function
    incorporates graded penalty to harshly reduce the presence of infeasible individuals
    :param pop: population, each individual is indicating the presence of each node
    :return: fitness score of the individual
    """
    fit = 0.
    for i in range(len(ind)):
        pen = 0
        for j in range(i,len(ind)):
            pen += ind[j]*adjMat[i,j]
        fit += (ind[i] - len(ind)*ind[i]*pen)
    return fit


def cliqueSize(nodelist: np.array, adjMat: np.array):
    if isClique(nodelist, adjMat): return nodelist.sum()
    else: return 0

@jit(nopython=True)
def BKFitPop(pop: np.array, adjmat: np.array):
    """
    wrapper for BackKhuriFitness across complete population
    :param pop: population matrix
    :param adjmat:adjacency matrix
    :return:
    """
    fit = np.zeros((len(pop)))
    for i in range(len(fit)):
        fit[i] = BackKhuriFitness(pop[i], adjmat)
    # total = fit.sum()
    # for i in range(len(fit)): fit[i] = fit[i] / total
    return fit