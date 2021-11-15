from numba import jit, prange, uint32, boolean
import numpy as np
from Utils import dataloader as dl
from sklearn.preprocessing import MinMaxScaler


@jit(nopython=True)
def agrMat(population: np.array, adjmat: np.array):
    """
    Generate agreement matrix from a population of routes
    :param population:
    :return:
    """

    agrMat = np.zeros((len(population[0]), len(population[0])))
    # Aggregate each edge into agreement matrix
    for individual in population:
        agrMat += dl.nodelist_to_adjmarks(individual, adjmat)

    # Combine symetric edges to one half of matrix ([i,j] == [j,i])
    for i in range(agrMat.shape[0]):
        for j in range(i):
            agrMat[j,i]=0
    return agrMat

# @jit(nopython=True, cache=True)
def greedy_cliquegen(population: np.array, data: np.array):
    """
    Greedy clique generation from agreement matrix
    Inspired from "A Hybrid Heuristic for the Traveling Salesman Problem" (Baraglia, Hidalgo, & Perego)
    :param agrScores:
    :return:
    """
    # ignore edges connecting nodes to self:
    for i in range(len(population)):
        population[i,i] = 0.

    nodelist = np.zeros(len(population))
    edges_remn = np.array([[x,y] for x in range(len(population)) for y in range(x)])
    edges_remn_prob = np.array([population[x,y] for x in range(len(population)) for y in range(x)])
    remn_mask = np.ones(len(edges_remn)).astype(bool)

    edges = []; edges_prob = []; probmatch = True # last_edge = None; probmatch = True

    while probmatch:
        cand_tries = 0; acc_tries = 0
        # filter possible candidate edges by last selected edge
        cand_edge = None
        if len(edges) == 0:
            cand_edges = edges_remn
            cand_probs = edges_remn_prob
        else:
            # cand_mask = mask_arr(edges_remn, last_edge, remn_mask)
            cand_mask = np.array([True if (edges_remn[i,1] in last_edge and not remn_mask[i] == 1) else False for i in range(len(edges_remn))])
            cand_edges = edges_remn[cand_mask]
            cand_probs = edges_remn_prob[cand_mask]
        # X tries to choose a candidate edge
        while cand_edge is None:
            if len(cand_edges) == 0:
                return nodelist
            if cand_tries == len(population):
                idx_edge = np.where(cand_probs == cand_probs.max())[0][0]
                cand_edge = cand_edges[idx_edge]
                # nodelist[cand_edge[0]] = 1; nodelist[cand_edge[1]] = 1
                edges.append(cand_edge)
                edges_prob.append(cand_probs[idx_edge])
                mask_idx = [i for i in range(len(edges_remn)) if (cand_edge == edges_remn[i]).all()][0]
                remn_mask[mask_idx] = 1
            idx_edge = np.random.randint(0,high=len(cand_probs))
            if np.random.rand() <= cand_probs[idx_edge]:
                cand_edge = cand_edges[idx_edge]
                edges.append(cand_edge)
                edges_prob.append(cand_probs[idx_edge])
                mask_idx = [i for i in range(len(edges_remn)) if (cand_edge == edges_remn[i]).all()][0]
                remn_mask[mask_idx] = 1
            else: cand_tries += 1

        # X tries to accept new edge by joint probability of random edges
        while acc_tries < len(population):
            if np.random.rand() <= np.prod(edges_prob):
                nodelist[cand_edge[0]] = 1; nodelist[cand_edge[1]] = 1
                last_edge = cand_edge
                break
            else: acc_tries += 1
        if acc_tries == len(population): probmatch = False

    return nodelist

@jit(nopython=True, cache=True, parallel=True)
def mask_arr(edges_remn: np.array, last_edge, remn_mask: np.array):
    arr = np.zeros(len(edges_remn), dtype=boolean)
    for i in prange(len(edges_remn)):
        if edges_remn[i,1] in last_edge and not remn_mask[i] == 1:
            arr[i] = True
        else: arr[i] = False
    return arr


@jit(nopython=True, parallel=True)
def acc_cGA_routegen(population, data):
    """
    numba-compiled list comprehension for compact Genetic Algorithm population sampling
    :param population:
    :param data:
    :return:
    """
    samples = np.empty((2,len(data)), dtype=uint32)
    for i in prange(2):
        samples[i] = greedy_cliquegen(population, data)[:,0]
    return samples

@jit(nopython=True, parallel=True)
def acc_popupdate(population, samples, BKScore, adjmat, k):
    """
    numba-compiled function for updating cGA path representation table
    :param population:
    :param winner_edges:
    :param loser_edges:
    :param k:
    :return:
    """
    # Aggregate BKPopulation efects on adjMat
    aggr = np.zeros_like(population)
    for s in range(len(samples)):
        adj = dl.nodelist_to_edgelist(samples[s], adjmat)
        aggr += adj*BKScore[s]

    aggr = MinMaxScaler(feature_range=(-1,1)).fit_transform(aggr) / k

    return population + aggr