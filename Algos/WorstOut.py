import numpy as np
import Utils.dataloader as dl
import itertools
from numba import jit

def WorstOutHeuristic(nda_adjmat: np.array, params: dict):
    # initialize nodelist
    nodes = np.ones(len(nda_adjmat)).astype(np.int32)
    # generate edge count for heuristic removal
    edgeCount = get_edgecount(nda_adjmat)
    frames = []
    if params['animate']: frames.append(dl.nodelist_to_edgelist(nodes, nda_adjmat))

    while not (isClique(nodes, nda_adjmat) or nodes.sum() == 0):
        # select edges of remaining nodes
        idxEdges = np.where(nodes == 1)[0]
        edgesRemaining = edgeCount[idxEdges]

        # remove node with fewest edges
        nodes[idxEdges[np.where(edgesRemaining == edgesRemaining.min())[0][0]]] = 0
        if params['animate']: frames.append(dl.nodelist_to_edgelist(nodes, nda_adjmat))

    return {
    'soln_edgelist': dl.nodelist_to_edgelist(nodes, nda_adjmat),
    'soln_size': nodes.sum(), 'frames': frames
    }


@jit(nopython=True)
def get_edgecount(adjmat: np.array):
    edgecount = np.zeros(len(adjmat))
    for i in range(len(edgecount)):
        edgecount[i] = adjmat[i].sum()
    return edgecount


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