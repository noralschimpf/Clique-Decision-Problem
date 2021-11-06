import os, csv
import numpy as np

def load_dir(dir: os.path):
    output = {}
    files = os.listdir(dir)
    for file in files:
        joinfile = os.path.join(dir, file)
        if 'meta.' in file:
            with open(joinfile, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    output[row[0]] = row[1]
        elif 'out.' in file:
            head = 0
            with open(joinfile, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row[0][0] == '%': head+=1
                    else: break
            nda_edges = np.genfromtxt(joinfile, skip_header=head, dtype=int)
            output['nda'] = nda_edges
    return output