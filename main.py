import Utils.dataloader as dl, Utils.viz as viz
import Algos.GA_BackKhuri as GA_BK, Algos.WorstOut as WorstOut
import os, tracemalloc, time


datadir = 'Data/KONNECT'

def main(algo):
    # load relevant TSP data
    files = os.listdir(datadir)
    files = files[1:2]
    for f in range(len(files)):
        # if DEBUG and 'General' in datadir: files[f] = 'Random4.tsp'
        dict_data = dl.load_dir(os.path.join(datadir,files[f])) if 'KONNECT' in datadir else dl.load_DIMACS(os.path.join(datadir,files[f]))
        print(files[f])

        traceflag = f <= 4
        if traceflag: tracemalloc.start()
        sttime = time.time()
        if traceflag: stcur, stpeak = tracemalloc.get_traced_memory()
        opt_soln = algo(dl.edgelist_to_adjMat(dict_data['nda']),
                        params={'animate': True, 'n': 100, 'g': 400, 'k': 50, 'GA_algo': GA_BK.GA_BackKhuri,
                                'f_fit': GA_BK.BackKhuriFitness})
        if traceflag:
            edcur, edpeak = tracemalloc.get_traced_memory()
        else:
            edpeak = -1000
        edtime = time.time()
        tracemalloc.stop()

        metrics = {'time': edtime - sttime, 'memory': edpeak / 1000}
        opt_soln['name'] = algo.__name__
        viz.report(dict_data, opt_soln, metrics)

if __name__ == '__main__':
    algos = [GA_BK.GA_Simulate, WorstOut.WorstOutHeuristic]
    for a in algos: main(a)