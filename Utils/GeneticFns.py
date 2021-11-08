import numpy as np
from numba import jit, prange, uint32
from Algos import WisdomOfCrowds as woc

@jit(nopython=True)
def pop_init(k, nodes):
    population = np.zeros((k, nodes), dtype=uint32)
    for i in range(len(population)):
        for j in range(len(population[0])):
            population[i, j] = 0 if np.random.rand() < 0.5 else 1
    return population

@jit(nopython=True, cache=True)
def cross_p_point(pair1, pair2, p=1):
    """
    Select p indexes at random to join pair 1 and pair2
    To insure all cities remain in the route, crossover inserts remaining cities by the ordering of pair2, not the
    direct values in pair2
    :param pair1: city list order
    :param pair2: city list order
    :param p: number of cross points
    :return: single list
    """
    crosspoints = np.arange(0,len(pair1)+1,int(len(pair1)/(p+1)))
    crosspoints = crosspoints[1:]; crosspoints[-1] = len(pair1)
    combo = pair1
    for i in range(len(crosspoints) - 1):
        if i % 2 == 1: combo[crosspoints[i]:crosspoints[i+1]] = pair2[crosspoints[i]:crosspoints[i+1]]
    return combo

@jit(nopython=True)
def cross_prob(pair1, pair2, p1=.4):
    """
    For each city, choose by probability assigned to each pair
    Uses same ordering solution as cross_p_point to insure all cities remain in-use
    :param pair1: list of cities
    :param pair2: list of cities
    :param p1: probability for choosing pair1
    :return:
    """
    output = np.array([pair1[i] if np.random.uniform(0,1) <= p1 else -1 for i in range(len(pair1))])
    return output

@jit(nopython=True)
def setdif(a, b):
    result = []
    for i in range(len(a)):
        if not a[i] in b: result.append(a[i])
    return result


@jit(nopython=True, parallel=True)
def acc_fit(data, population, f_fit):
    """
    Numba-compiled list comprehension for fitness function assesment of population
    :param data:
    :param population:
    :param f_fit:
    :return:
    """
    fit = np.empty((len(population)))
    for i in prange(len(fit)):
        fit[i] = f_fit(data[population[i]-1])
    return fit
    # np.array([params['f_fit'](data[x - 1]) for x in population])

@jit(nopython=True, parallel=True)
def acc_cross(pair1, pair2, k, f_cross):
    """
    numba-compiled list comprehension for pair breeding
    :param pair1:
    :param pair2:
    :param k:
    :param f_cross:
    :return:
    """
    newpop = np.empty((k, len(pair1[0])), dtype=int32)
    for i in prange(k):
        newpop[i] = f_cross(pair1[i], pair2[i])
    return newpop
    # np.array([params['f_cross'](pair1[i], pair2[i]) for i in range(params['k'])])

@jit(nopython=True, parallel=True)
def pairgen(fit: np.array, idx: np.array, k: int):
    """
    numba-accelerated process of generating pairs
    :param fit:
    :param idx:
    :param k:
    :return:
    """
    pairs = np.empty((k*2), dtype=np.int32)
    for x in prange(k*2):
        pairs[x] = rand_choice_nb(idx, fit)
    return pairs

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Workaround for numba-supported np.random.choice
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def mut_invert(ind, p):
    """
    mutation function: each node is inverted with probability p
    :param ind: individual
    :param p: mutation probability
    :return: mutated individual
    """
    for i in range(len(ind)):
        mut = True if np.random.random() <= p else False
        if mut and ind[i] == 1: ind[i] = 0
        elif mut and ind[i] == 0: ind[i] = 1
    return ind