import Utils.dataloader as dl, Utils.viz as viz
import Algos.GA_BackKhuri as GA_BK, Algos.WorstOut as WorstOut
import os, tracemalloc, time




datadir = 'Data/KONNECT'
if 'KONNECT' in datadir: fileorder = ['brunson_southern-women', 'brunson_south-africa', 'ucidata-gama', 'moreno_taro',
                                      'ucidata-zachary', 'hiv', 'moreno_lesmis', 'arenas-jazz', 'moreno_innovation',
                                      'maayan-faa', 'opsahl-powergrid']
elif 'DIMACS' in datadir: fileorder = ['johnson8-2-4.mtx', 'MANN-a9.mtx',  'hamming6-2.mtx', 'hamming6-4.mtx',
                                       'johnson8-4-4.mtx', 'johnson16-2-4.mtx', 'C125-9.mtx', 'keller4.mtx',
                                       'brock200-1.mtx', 'brock200-2.mtx', 'brock200-3.mtx', 'brock200-4.mtx',
                                       'c-fat200-1.mtx', 'c-fat200-2.mtx', 'c-fat200-5.mtx']

def main(algo):
    # load relevant TSP data
    # fileorder = fileorder[1:2]
    for f in range(len(fileorder)):
        file =  os.path.join(datadir,fileorder[f])
        # if DEBUG and 'General' in datadir: files[f] = 'Random4.tsp'
        dict_data = dl.load_dir(file) if 'KONNECT' in datadir else dl.load_DIMACS(file)
        print(fileorder[f])

        traceflag = f <= 4
        if traceflag: tracemalloc.start()
        sttime = time.time()
        if traceflag: stcur, stpeak = tracemalloc.get_traced_memory()
        opt_soln = algo(dl.edgelist_to_adjMat(dict_data['nda']),
                        params={'animate': False, 'n': 100, 'g': 400, 'k': 50, 'GA_algo': GA_BK.GA_BackKhuri,
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
    # algos = [GA_BK.GA_Simulate, WorstOut.WorstOutHeuristic]
    algos = [GA_BK.GA_Simulate]
    for a in algos: main(a)