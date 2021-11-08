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

    while not isClique(nodes, nda_adjmat):
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
    idxnodes = np.where(nodes == 1)[0]
    iternodes = itertools.combinations(idxnodes, 2)
    for iter in iternodes:
        if adjmat[iter[0],iter[1]] == 0: return False
    return True