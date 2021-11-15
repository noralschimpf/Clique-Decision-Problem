import numpy as np
import tqdm
from numba import uint32, jit
from sklearn.preprocessing import MinMaxScaler
from scipy.special import betaincinv
import Utils.GeneticFns as GF
import Utils.dataloader as dl
from Algos import WisdomOfCrowds as WOC
from Utils.heuristics import BackKhuriFitness, BKFitPop

def WoC_BackKhuri(nda_adjMat: np.array, params: dict):
    """
    Genetic Algorithm implemented by Thomas Back and Sami Kuri in
    "An Evolutionary Heuristic for the Maximum Independent Set Problem"
    DIFFERENCE: no 60% crossover probability incorporated. crossovers are taken based on initial sampling ONLY
    first succesfull use of GA on Maximum Clique Problem
    :param nda_adjMat: adjacency matrix
    :param params: dictionary of variable parameters including:
    {k: population size,g: generations, f_fit: fitness function, f_cross: crossover function, f_mut: mutation function}
    :return:
    """

    # DEFINED IN PAPER
    params['k'] = 50
    params['g'] = 400
    params['f_fit'] = BackKhuriFitness
    params['f_cross'] = GF.cross_p_point
    params['f_mut'] = GF.mut_invert

    # Author algorithm solves for maximum clique in G' = (V,E')
    nda_adjCompl = 1 - nda_adjMat

    # initialize the population
    # inididuals: list of nodes:  0 or 1 if in clique
    population = GF.pop_init(params['k'], len(nda_adjCompl))
    best_fits = np.zeros(400)
    frames = []
    soln_nodelist, soln_fit = None, None
    scl = MinMaxScaler(feature_range=(0, 1))

    for g in range(400):

        # Assess fitness using graded penalty
        fitness = BKFitPop(population, nda_adjCompl)

        # WISDOM OF CROWDS: generate Agreement Matrix using current population
        agrMat = WOC.agrMat(population)
        # scoreMat = betaincinv(params['b_1'], params['b_2'], agrMat / agrMat.sum())
        scoreMat = agrMat / params['k']
        woc_nodelist = WOC.greedy_cliquegen(scoreMat)
        best_fits[g] = BackKhuriFitness(woc_nodelist, nda_adjCompl)

        if soln_fit is None or best_fits[g] > soln_fit:
            soln_fit = best_fits[g]
            soln_nodelist = woc_nodelist

        fitness = scl.fit_transform(fitness.reshape(-1,1))
        fitness = np.exp(2*(fitness-1))
        fitness = fitness / fitness.sum()
        bestfit_normed = max(fitness)

        # Append best route of generation to animation
        if params['animate'] and g % 10 == 0:
            frames.append(dl.nodelist_to_edgelist(soln_nodelist, nda_adjMat))

        # Select and "breed" pairs
        pairs = GF.pairgen(fitness, np.arange(len(population)).astype(np.uint32), 50)
        pair1, pair2 = population[pairs[:50]],population[pairs[50:]]
        #   Merge and mutate pairs
        population = GF.acc_cross(pair1, pair2, params['k'], GF.cross_p_point) # np.array([GF.cross_p_point(pair1[i], pair2[i], p=2) for i in range(50)])
        population = GF.acc_mut(population, GF.mut_invert) # np.array([GF.mut_invert(population[i], 1/len(population[0])) for i in range(50)])

    # soln_nodelist = 1 - soln_nodelist
    return {'soln_edgelist': dl.nodelist_to_edgelist(soln_nodelist, nda_adjMat), 'soln_size': soln_nodelist.sum(),
            'soln_nodelist': soln_nodelist, 'training': best_fits, 'frames': frames, 'soln_fit': soln_fit}

def WoC_BK_Simulate(nda_adjMat: np.array, params: dict):
    nda_adjCompl = 1 - nda_adjMat
    fit_curves = np.zeros((params['n'], params['g']))
    soln_fits = np.zeros(params['n'])
    best_fit_idx = -1;
    best_edgelist = None
    GA_Name = 'Genetic {}'.format(params['GA_algo'].__name__)
    frames = []
    for n in tqdm.trange(params['n']):
        if n == 1: params['animate'] = False
        soln_dict = params['GA_algo'](nda_adjMat, params)

        if n == 0: frames = soln_dict['frames']
        fit_curves[n] = soln_dict['training']
        soln_fits[n] = soln_dict['soln_fit']
        if soln_fits[n] > soln_fits[best_fit_idx] or best_fit_idx == -1:
            best_fit_idx = n
            best_edgelist = soln_dict['soln_edgelist']

    return {
        'soln_edgelist': best_edgelist, 'soln_size': len(dl.edgelist_to_nodelist(best_edgelist)),
        'fit_curves': fit_curves, 'fits': soln_fits, 'frames': frames
    }