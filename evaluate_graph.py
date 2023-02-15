from bayesian_scoring import bayesian_score
from utils import load_gph, load_data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":

    # Check arguments
    if len(sys.argv) != 3:
        raise Exception("usage: python evaluate_graph.py <infile>.csv, <gphfile>.gph")
    inputfilename = sys.argv[1]
    gphfilename = sys.argv[2]

    # Load data
    df, vars = load_data(inputfilename)

    # Load graph
    G = load_gph(gphfilename, vars)

    # Score graph
    score = bayesian_score(vars, G, df)
    print("Score: {}".format(score))

    # Display the graph
    G = load_gph(gphfilename, vars, use_var_names=True)
    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes())), iterations=20)
    plt.figure(3, figsize=(8, 8))
    nx.draw_networkx(G, pos=pos)
    plt.show()