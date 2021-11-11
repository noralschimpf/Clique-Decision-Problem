import numpy as np
import Utils.dataloader as dl
import itertools
from Utils.heuristics import get_edgecount, isClique

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