import os, csv
import numpy as np
import networkx as nx

def load_dir(dir: str):
    output = {}
    files = os.listdir(dir)
    for file in files:
        joinfile = os.path.join(dir, file)
        if 'meta.' in file:
            with open(joinfile, 'r') as f:
                reader = csv.reader(f, delimiter=':')
                for row in reader:
                    output[row[0]] = row[1]
        elif 'out.' in file:
            head = 0
            with open(joinfile, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if row[0][0] == '%': head+=1
                    else: break
            nda_edges = np.genfromtxt(joinfile, skip_header=head, dtype=int).astype(np.uint32)
            if nda_edges.shape[1] > 2: nda_edges = nda_edges[:,:2]
            output['nda'] = nda_edges
    output['name'] = os.path.split(dir)[1]
    return output


def load_DIMACS(file: str):
    output = {}
    skiphead = 0
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if '%' == row[0][0]:
                skiphead += 1
                output[row[1]] = row[2]
            else: break
    nda_edges = np.genfromtxt(file, skip_header=skiphead, dtype=int).astype(np.uint32)
    if nda_edges.shape[1] > 2: nda_edges = nda_edges[:, :2]
    output['nda'] = nda_edges
    output['name'] = os.path.split(file)[1]
    return output



def edgelist_to_adjMat(edgelist: np.array):
    G = nx.Graph()
    G.add_edges_from(edgelist)
    return nx.to_numpy_array(G).astype(np.uint32)
    # verts = edgelist.max().astype(np.uint32)
    # adjmat = np.zeros((verts, verts)).astype(np.uint32)
    # for row in edgelist:
    #     adjmat[row[0]-1, row[1]-1] = 1
    #
    # # insure symmetry of edges
    # for i in range(len(adjmat)):
    #     for j in range(i):
    #         if adjmat[i,j] == 1 or adjmat[j,i] == 1:
    #             adjmat[i,j] = adjmat[j,i] = 1
    # return adjmat

def adjMat_to_edgelist(adjmat: np.array):
    edgelist = np.zeros((adjmat.sum(), 2), dtype=np.uint32)
    edgeno = 0
    for i in range(len(adjmat)):
        for j in range(len(adjmat)):
            if adjmat[i,j] == 1:
                edgelist[edgeno,0], edgelist[edgeno,1] = i, j
                edgeno += 1
    return edgelist +1

def nodelist_to_edgelist(nodelist: np.array, adjmat: np.array):
    G = nx.from_numpy_array(adjmat)
    H = G.subgraph(nodes=np.where(nodelist==1)[0])
    return np.array(list(H.edges)) +1

def edgelist_to_nodelist(edgelist: np.array):
    nodes = []
    for edge in edgelist:
        if not edge[0] in nodes: nodes.append(edge[0])
        if not edge[1] in nodes: nodes.append(edge[1])
    return np.array(nodes)