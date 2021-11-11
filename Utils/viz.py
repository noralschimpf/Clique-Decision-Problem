import networkx as nx
import pylab as P
import numpy as np
from matplotlib.animation import FuncAnimation, writers
import os

def plotnet(nda: np.array):
    G = nx.Graph()
    G.add_nodes_from(nda)

def update_clique(frame: np.array, G: nx.Graph, pos):
    frame = frame.tolist()
    edgecolors = ['tab:red' if list(edge) in frame else 'tab:blue' for edge in nx.edges(G)]
    nodes = [x[0] for x in frame]
    nodecolors = ['tab:red' if node in nodes else 'tab:blue' for node in nx.nodes(G)]
    # G.clear()
    # G.add_edges_from(frame)
    nx.draw(G, pos=pos, node_color=nodecolors, edge_color=edgecolors)


def report(data: dict, soln: dict, metrics: dict):
    """
    Generates report figure of proposed route
    :param data: dictionary containing the cities, problem name, and any miscellaneous information
    :param soln: dictionary containing the route, algorithm used, and miscellaneous information
    :param metrics: dictionary containing the runtime and memory usage of the solution
    :return:
    """
    if not os.path.isdir('Figures/{}/{}'.format(soln['name'],data['name'])): os.makedirs('Figures/{}/{}'.format(soln['name'],data['name']))

    G = nx.Graph()
    G.add_edges_from(data['nda'])
    pos = nx.spring_layout(G)

    G_soln = nx.Graph()

    fig, ax = P.subplots(1,1)
    nx.draw(G, pos=pos, ax=ax, node_color='tab:blue', edge_color='tab:blue')
    fig.suptitle(
        'MCP {} Using {}\nsize: {:.3f}   time:{:.3f} s   mem: {:.3f} KB'.format(
            data['name'], soln['name'], soln['soln_size'], metrics['time'], metrics['memory']))

    # animate solution, if relevant
    if len(soln['frames']) > 0:
        animation = FuncAnimation(fig, func=update_clique, fargs=(G, pos), frames=soln['frames'])
        Writer = writers['ffmpeg']
        writer = Writer(fps=5, metadata={'artist': 'Me'}, bitrate=1800)
        animation.save('Figures/{}/{}/{}.mp4'.format(soln['name'], data['name'],data['name']), writer)


    # Plot solution overlaying total graph
    soln_edges = soln['soln_edgelist'].tolist()
    edgecolors = ['tab:red' if list(edge) in soln_edges else 'tab:blue' for edge in nx.edges(G)]
    nodes = set([x[0] for x in soln_edges])
    nodecolors = ['tab:red' if node in nodes else 'tab:blue' for node in nx.nodes(G)]
    nx.draw(G, pos=pos, node_color=nodecolors, edge_color=edgecolors)
    fig.savefig("Figures/{}/{}/{}.png".format(soln['name'],data['name'], data['name']), dpi=300)
    fig.clf(); ax.cla(); P.close()


    if 'GA' in soln['name']:
        # Plot training figure (for GA algos)
        fig, ax = P.subplots(1,1)
        for i in range(len(soln['fit_curves'])):
            ax.plot(soln['fit_curves'][i], color='b', alpha=0.3)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness Score")
        fig.suptitle(
            'MCP {} Using {}\nPopulation: {}'.format(
                data['name'], soln['name'], len(soln['fit_curves'])))
        # fig.suptitle("TSP {} (Dim: {})\nPopulation: {}".format(GA_Name, data['DIMENSION'], len(fit_curves)))
        fig.savefig("Figures/{}/{}/train.png".format(soln['name'],data['name']), dpi=300)
        fig.clf(); ax.cla(); P.close()

        # Plot size statistics (for GA algos)
        fig_stats, ax_stats = P.subplots(1, 1)
        ax_stats.hist(soln['fits'], 50)
        ax_stats.set_xlabel("Fitness Score")
        ax_stats.set_ylabel("Count")
        fig_stats.suptitle("MCP {} Using {}\nMean: {:.3f}    std: {:.3f}    time: {:.3f}s   mem: {:.3f}KB".format(
            soln['name'], data['name'], soln['fits'].mean(), soln['fits'].std(), metrics['time'], metrics['memory']))
        fig_stats.savefig('Figures/{}/{}/stats.png'.format(soln['name'], data['name']), dpi=300)
        fig_stats.clf(); ax_stats.cla(); P.close()