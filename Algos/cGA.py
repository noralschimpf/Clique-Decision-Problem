import numpy as np
from scipy.special import betaincinv
from sklearn.preprocessing import MinMaxScaler
from Utils import GeneticFns as GF, heuristics as H, dataloader as dl
from Algos import WisdomOfCrowds as WOC


def cGA_WoC(nda_adjMat: np.array, params: dict):
    '''
    Modular implementation of a genetic algorithm
    :param data: complete list of cities
    :param restrictions: path restrictions between cities / nodes
    :param status: tracks the status of the genetic algorithm. Includes DEBUG and ANIMATE flags
    :param params: dictionary of variable parameters including:
    {k: population size,g: generations, f_fit: fitness function, f_cross: crossover function, f_mut: mutation function,
    b_1: b1 parameter for inverse regularized beta function, b_2: b2 for inverse regularized beta function}
    :return:
    '''
    GA_frames, woc_frames = [], []

    # initialize the adjacency table
    adjTable = np.zeros_like(nda_adjMat)
    for i in range(adjTable.shape[0]):
        for j in range(i):
            if nda_adjMat[i,j] == 1: adjTable[i,j] = .5

    best_fits = np.zeros(params['g'])
    frames = []
    soln_nodelist, soln_fit = None, None
    scl = MinMaxScaler(feature_range=(0, 1))

    best_fits, woc_fits = np.zeros(params['g']), np.zeros(params['g'])

    for g in range(params['g']):
        # Generate two sample routes based on the path-representation model
        # Assign a probability of selection for each result using a fitness function
        #   Score each function with a heuristic (0 best, increasing for worse)
        samples = np.array([WOC.greedy_cliquegen(population, nda_adjMat) for x in range(2)])
        fit = H.BKFitPop(samples, nda_adjMat)
        best_fits[g] = min(fit)

        # WISDOM OF CROWDS: generate Agreement Matrix using current population
        # scoreMat = betaincinv(params['b_1'], params['b_2'], population)
        woc_nodelist = WOC.greedy_cliquegen(population, nda_adjMat)
        woc_fits[g] = params['f_fit'](woc_nodelist)

        # Append best route of generation to animation
        if params['animate'] and g % 10 == 0:
            idx_best = [i for i in range(len(fit)) if fit[i] == best_fits[g]][0]
            best_nodelist = population[idx_best]
            GA_frames.append(dl.nodelist_to_edgelist(best_nodelist, nda_adjMat))
            woc_frames.append(dl.nodelist_to_edgelist(woc_nodelist, nda_adjMat))

        # Update population based on fitness sample
        # aggregate BackKhuriFitness scores of samples, scale and apply to population table
        population = GF.acc_popupdate(population, samples, fit, nda_adjMat, params['k'])

    return {'soln_nodelist': best_nodelist, 'soln_edgelist': dl.nodelist_to_edgelist(best_nodelist, nda_adjMat),
            'soln_size': best_nodelist.sum(), 'training': best_fits, 'frames': frames, 'soln_fit': best_fits[-1]}