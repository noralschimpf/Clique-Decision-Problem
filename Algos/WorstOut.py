import numpy as np
import Utils.dataloader as dl
import itertools

def WorstOutHeuristic(nda_adjmat: np.array, params: dict):
    # initialize nodelist
    nodes = np.ones(len(nda_adjmat)).astype(np.int32)
    # generate edge count for heuristic removal
    edgeCount = np.zeros(len(nodes))
    for i in range(len(edgeCount)):
        edgeCount[i] = nda_adjmat[i].sum()
    frames = []

    while not isClique(nodes, nda_adjmat):
        # select edges of remaining nodes
        idxEdges = np.where(nodes == 1)
        edgesRemaining = edgeCount[idxEdges]

        # remove node with fewest edges
        nodes[np.where(edgesRemaining == edgesRemaining.min())] = 0
        if params['animate']: frames.append(dl.nodelist_to_edgelist(nodes))

    return {
    'soln_edgelist': dl.nodelist_to_edgelist(nodes, nda_adjmat),
    'soln_size': nodes.sum(), 'frames': frames
    }




def isClique(nodes: np.array, adjmat: np.array):
    idxnodes = np.where(nodes == 1)
    iternodes = itertools.combinations(idxnodes, 2)
    for iter in iternodes:
        if adjmat[iter[0],iter[1]] == 0: return False
    return True