from numba import jit, prange, uint32, boolean
import numpy as np
from Utils import dataloader as dl
from sklearn.preprocessing import MinMaxScaler


@jit(nopython=True)
def agrMat(population: np.array):
    """
    Generate agreement matrix from a population of routes
    :param population:
    :return:
    """

    agrMat = np.zeros((len(population[0])))
    # Aggregate each edge into agreement matrix
    for individual in population:
        agrMat += individual
    return agrMat

@jit(nopython=True, cache=True)
def greedy_cliquegen(population: np.array):
    """
    Greedy clique generation from agreement matrix
    Inspired from "A Hybrid Heuristic for the Traveling Salesman Problem" (Baraglia, Hidalgo, & Perego)
    :param agrScores:
    :return:
    """
    nodelist = np.zeros(len(population), dtype=uint32)
    jointprob = True; tries = 0
    while jointprob:
        cand_node = np.random.randint(0,len(nodelist))
        if np.random.random() <= population[cand_node]:
            tmp = nodelist.copy(); tmp[cand_node] = 1
            if np.random.rand() <= np.prod(tmp[tmp == 1]):
                nodelist = tmp
            else: tries += 1
        else: tries += 1
        if tries == len(population): jointprob = False
    return nodelist
    # for i in range(len(nodelist)):
    #     if np.random.random() <= population[i]: nodelist[i] = 1
    # return nodelist

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

# @jit(nopython=True)
def acc_popupdate(population, samples, Score, adjmat, k):
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
        if Score[s] > 0:
            aggr += samples[s]*Score[s]
        else:
            aggr -= samples[s]

    # mini, maxi = -len(samples)*len(population)*sum(range(len(population)+1)), len(samples)*len(population)
    # MinMaxScaler implementation: BK penalty/reward range -> -1 - 1/BK_max
    # aggr = ((aggr - mini) / (maxi - mini)) * (1/sum(range(len(population)+1)) - -1) - 1
    aggr = aggr / (np.abs(aggr).sum() * k)

    # choose best-half of samples as reward, increment probability by 1/k
    # reward, penalty = np.zeros_like(population), np.zeros_like(population)
    # splitpoint = int(len(samples)/2)
    # winners, losers = np.argsort(BKScore)[splitpoint:], np.argsort(BKScore)[:splitpoint]
    # for w in winners:
    #     reward += samples[w] / samples[w].sum()
    # for l in losers:
    #     penalty += samples[l] / samples[l].sum()
    # reward = reward / k; penalty = penalty/k

    return population + aggr