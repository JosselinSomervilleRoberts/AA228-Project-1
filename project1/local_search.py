from utils import write_gph, is_cyclic
import networkx as nx
from tqdm import tqdm
import numpy as np
from bayesian_scoring import bayesian_score, bayesian_score_recompute_single_var


def rand_graph_neighbor_with_score(G, score, score_comp, df, vars):
    n = G.number_of_nodes()
    i = np.random.randint(1, n)
    j = i
    while j == i:
        j = np.random.randint(1, n)
    G_prime = G.copy()
    if G.has_edge(i, j):
        G_prime.remove_edge(i, j)
    else:
        G_prime.add_edge(i, j)
    if is_cyclic(G_prime):
        return G_prime, None, None, j
    score_prime, score_comp_prime = bayesian_score_recompute_single_var(score, score_comp, vars, G_prime, df, j)
    return G_prime, score_prime, score_comp_prime, j

# Local Search algorithm
def local_search(vars, df, k_max, data_name):
    # Generate initial graph
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    score, score_comp = bayesian_score(vars, G, df)

    for k in tqdm(range(k_max)):
        G_prime, score_prime, score_comp_prime, j = rand_graph_neighbor_with_score(G, score, score_comp, df, vars)
        if is_cyclic(G_prime):
            continue
        if score_prime > score:
            score = score_prime
            score_comp[j] = score_comp_prime    
            G = G_prime
            print("New best score: {}".format(score))
            if score > -425000: write_gph(G, vars, data_name=data_name, gph_name="best_" + str(k), score=score)
    return G, score


# Local Search algorithm with Simulated annealing, random restarts and random initializations
def random_graph_init(vars, df):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    score, score_comp = bayesian_score(vars, G, df)
    return G, score, score_comp

def local_search_with_optis(vars, df, k_max, data_name,
                            t_max=5,
                            k_max_without_improvements=2000,
                            score_improvement_to_save=5,
                            score_min_to_save=-4200,
                            log_score_every=1000):
    # Generate initial graph
    G, score, score_comp = random_graph_init(vars, df)

    # To keep track of the best graph
    last_saved_score = -np.inf
    k_last_improvement = -1
    k_last_restart = 0

    for k in tqdm(range(k_max)):
        temp = t_max * (1 - (k - k_last_restart) / (k_max - k_last_restart) )
        G_prime, score_prime, score_comp_prime, j = rand_graph_neighbor_with_score(G, score, score_comp, df, vars)
        if score_prime is None: continue # This means that the graph is cyclic

        # Simulated annealing
        diff = score_prime - score
        if diff > 0 or np.random.rand() < np.exp(diff/temp):
            score = score_prime
            score_comp[j] = score_comp_prime    
            G = G_prime
        
        # Random restarts
        if diff > 0:
            k_last_improvement = k
        if k - k_last_improvement > k_max_without_improvements:
            k_last_restart = k
            G, score, score_comp = random_graph_init(vars, df)

        # Saving best graph
        if diff > 0 and score >= score_min_to_save and score >= last_saved_score + score_improvement_to_save:
            last_saved_score = score
            print("New best score: {}".format(score))
            write_gph(G, vars, data_name=data_name, gph_name="score_" + str(int(round(-score))), score=score)

        # Logging
        if k % log_score_every == 0:
            print("Current score: {}".format(score))
    return G, score


if __name__ == "__main__":
    import sys
    from utils import load_data, write_gph

    # Check arguments
    if len(sys.argv) != 2:
        raise Exception("usage: python k2_search.py <infile>.csv")
    inputfilename = sys.argv[1]
    data_name = inputfilename.split("/")[-1].split(".")[0]

    # Optims
    use_optims = True if len(sys.argv) == 3 and sys.argv[2] == "optims" else False
    if use_optims:
        print("Using optims")
    else:
        print("Not using optims")

    # Load data
    df, vars = load_data(inputfilename)

    # Run k2
    G, score = None, None
    if use_optims:
        G, score = local_search_with_optis(vars, df, 1000, data_name=data_name)
    else:
        G, score = local_search(vars, df, 1000, data_name=data_name)
    print("Best score: {}".format(score))
    write_gph(G, vars, data_name=data_name, gph_name="best", score=score)