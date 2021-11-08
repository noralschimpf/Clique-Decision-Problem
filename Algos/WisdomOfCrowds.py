# from numba import jit, prange, int32
# import numpy as np
#
# @jit(nopython=True, parallel=True)
# def acc_cGA_routegen(population, data):
#     """
#     numba-compiled list comprehension for compact Genetic Algorithm population sampling
#     :param population:
#     :param data:
#     :return:
#     """
#     samples = np.empty((2,len(data)), dtype=int32)
#     for i in prange(2):
#         samples[i] = woc.greedy_routegen(population, data)[:,0]
#     return samples
#
# @jit(nopython=True, parallel=True)
# def acc_popupdate(population, winner_edges, loser_edges, k):
#     """
#     numba-compiled function for updating cGA path representation table
#     :param population:
#     :param winner_edges:
#     :param loser_edges:
#     :param k:
#     :return:
#     """
#     for arr in [winner_edges, loser_edges]:
#         for i in prange(len(arr) - 1):
#             x, y = arr[i], arr[i + 1]
#             if x > y:
#                 tmp = x
#                 x = y
#                 y = tmp
#             if (arr == winner_edges).all():
#                 population[x, y] += (1 / k)
#             else:
#                 population[x, y] -= (1 / k)
#     return population