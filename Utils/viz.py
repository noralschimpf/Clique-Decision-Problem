import networkx as nx
import numpy as np

def plotnet(nda: np.array):
    G = nx.Graph()
    G.add_nodes_from(nda)