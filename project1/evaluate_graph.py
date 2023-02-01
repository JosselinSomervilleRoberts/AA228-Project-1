from bayesian_scoring import bayesian_score
from utils import load_gph, load_data
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