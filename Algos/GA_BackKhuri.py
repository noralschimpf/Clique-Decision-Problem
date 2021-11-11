import numpy as np
import tqdm
import Utils.GeneticFns as GF
import Utils.dataloader as dl
from numba import uint32, jit
from sklearn.preprocessing import MinMaxScaler

def GA_BackKhuri(nda_adjMat: np.array, params: dict):
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
    scl = MinMaxScaler(feature_range=(0, 1))

    for g in range(400):

        # Assess fitness using graded penalty
        fitness = BKFitPop(population, nda_adjCompl)
        best_fits[g] = max(fitness)
        fitness = scl.fit_transform(fitness.reshape(-1,1))
        fitness = fitness / fitness.sum()
        bestfit_normed = max(fitness)

        # Append best route of generation to animation
        if params['animate'] and g % 10 == 0:
            idx_best = [i for i in range(len(fitness)) if fitness[i] == bestfit_normed][0]
            best_ind = population[idx_best]
            frames.append(dl.nodelist_to_edgelist(1-best_ind, nda_adjMat))

        # Select and "breed" pairs
        pairs = GF.pairgen(fitness, np.arange(len(population)).astype(np.uint32), 50)
        pair1, pair2 = population[pairs[:50]],population[pairs[50:]]
        #   Merge and mutate pairs
        population = np.array([GF.cross_p_point(pair1[i], pair2[i], p=2) for i in range(50)])
        population = np.array([GF.mut_invert(population[i], 1/len(population[0])) for i in range(50)])

    soln_nodelist = 1 - population[np.where(fitness == max(fitness))[0][0]]
    return {'soln_edgelist': dl.nodelist_to_edgelist(soln_nodelist, nda_adjMat), 'soln_size': soln_nodelist.sum(),
            'soln_nodelist': soln_nodelist, 'training': best_fits, 'frames': frames}

def GA_Simulate(nda_adjMat: np.array, params: dict):
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
        soln_fits[n] = params['f_fit'](soln_dict['soln_nodelist'], nda_adjMat)
        if soln_fits[n] > soln_fits[best_fit_idx] or best_fit_idx == -1:
            best_fit_idx = n
            best_edgelist = soln_dict['soln_edgelist']

    return {
        'soln_edgelist': best_edgelist, 'soln_size': len(dl.edgelist_to_nodelist(best_edgelist)),
        'fit_curves': fit_curves, 'fits': soln_fits, 'frames': frames
    }

@jit(nopython=True)
def BKFitPop(pop: np.array, adjmat: np.array):
    """
    wrapper for BackKhuriFitness
    :param pop:
    :param adjmat:
    :return:
    """
    fit = np.zeros((len(pop)))
    for i in range(len(fit)):
        fit[i] = BackKhuriFitness(pop[i], adjmat)
    # total = fit.sum()
    # for i in range(len(fit)): fit[i] = fit[i] / total
    return fit

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
