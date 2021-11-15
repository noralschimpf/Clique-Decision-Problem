import numpy as np, tqdm
from scipy.special import betaincinv
from sklearn.preprocessing import MinMaxScaler
from Utils import GeneticFns as GF, heuristics as H, dataloader as dl
from Algos import WisdomOfCrowds as WOC


def cGA_BackKhuri_Marchiori(nda_adjMat: np.array, params: dict):
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

    # BackKhuri Fitness metric relies on complement of adjacenency matrix
    nda_adjCompl = 1 - nda_adjMat

    # initialize the adjacency table
    nodeTable = np.zeros(len(nda_adjMat))
    for i in range(nodeTable.shape[0]):
        nodeTable[i] = .5

    best_fits = np.zeros(params['g'])
    frames = []
    soln_nodelist, soln_fit = None, None
    scl = MinMaxScaler(feature_range=(0, 1))

    best_fits, woc_fits = np.zeros(params['g']), np.zeros(params['g'])

    for g in range(params['g']):
        # Generate two sample routes based on the path-representation model
        # Assign a probability of selection for each result using a fitness function
        #   Score each function with a heuristic (0 best, increasing for worse)
        samples = np.array([WOC.greedy_cliquegen(nodeTable) for x in range(5)])

        # Apply Heuristic Algorithm to Generate Cliques from Samples
        samples = H.MarchRecursive(samples, nda_adjMat)

        fit = [H.cliqueSize(x, nda_adjMat) for x in samples]
        best_fits[g] = max(fit)
        if soln_nodelist is None or best_fits[g] > soln_fit:
            soln_fit = best_fits[g]
            soln_nodelist = samples[[i for i in range(len(fit)) if fit[i] == best_fits[g]][0]]

        # Append best route of generation to animation
        if params['animate'] and g % 10 == 0:
            idx_best = [i for i in range(len(fit)) if fit[i] == best_fits[g]][0]
            best_nodelist = samples[idx_best]
            GA_frames.append(dl.nodelist_to_edgelist(best_nodelist, nda_adjMat))

        # Update population based on fitness sample
        # aggregate BackKhuriFitness scores of samples, scale and apply to population table
        nodeTable = WOC.acc_popupdate(nodeTable, samples, fit, nda_adjMat, params['k'])

    return {'soln_nodelist': soln_nodelist, 'soln_edgelist': dl.nodelist_to_edgelist(soln_nodelist, nda_adjMat),
            'soln_size': soln_nodelist.sum(), 'training': best_fits, 'frames': frames, 'soln_fit': soln_fit}

def cGA_Simulate(nda_adjMat: np.array, params: dict):
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