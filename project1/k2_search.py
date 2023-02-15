import networkx as nx
from tqdm import tqdm
import numpy as np
from bayesian_scoring import bayesian_score, bayesian_score_recompute_single_var
from utils import write_gph, inneighbors


def k2_iter(vars, df, num_iter, max_parents=2, data_name="small"):
    #past_orderings = set()
    best_score = -np.inf
    best_G = None

    # Compute empty score
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    empty_score, empty_score_comp = bayesian_score(vars, G, df)

    for idx in tqdm(range(num_iter)):
        # generate a random ordering
        ordering = np.random.permutation(len(vars))
        #while ordering in past_orderings:
        #    ordering = np.random.permutation(len(vars))
        #past_orderings.add(ordering)

        # run k2 on the ordering
        G, score = k2(ordering, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp.copy())
        if score > best_score:
            best_score = score
            best_G = G
            write_gph(best_G, vars, data_name=data_name, gph_name="k2_" + str(idx), score=best_score)
            print("New best score: {}".format(best_score))
    return best_G, best_score


# K2 algorithm
def k2(ordering, vars, df, max_parents=2, empty_score=None, empty_score_comp=None):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(ordering))))
    score, score_comp = empty_score, empty_score_comp
    if score is None or score_comp is None: score, score_comp = bayesian_score(vars, G, df)
    for (k, i) in enumerate(tqdm(ordering[1:])):
        if len(inneighbors(G, i)) >= max_parents:
            continue
        while True:
            score_best, j_best, score_comp_best = -np.inf, 0, None
            for j in ordering[:k]:
                if not G.has_edge(j, i):
                    G.add_edge(j, i)
                    new_score, new_score_comp = bayesian_score_recompute_single_var(score, score_comp, vars, G, df, i)
                    if new_score > score_best:
                        score_best, j_best, score_comp_best = new_score, j, new_score_comp
                    G.remove_edge(j, i)
            if score_best > score:
                score = score_best
                score_comp[i] = score_comp_best
                G.add_edge(j_best, i)
            else:
                break
    return G, score_best


if __name__ == "__main__":
    import sys
    from utils import load_data, write_gph

    # Check arguments
    if len(sys.argv) != 2:
        raise Exception("usage: python k2_search.py <infile>.csv")
    inputfilename = sys.argv[1]
    data_name = inputfilename.split("/")[-1].split(".")[0]

    # Load data
    df, vars = load_data(inputfilename)

    # Run k2
    G, score = k2_iter(vars, df, 1000, max_parents=4, data_name=data_name)
    print("Best score: {}".format(score))
    write_gph(G, vars, data_name=data_name, gph_name="k2_best", score=score)