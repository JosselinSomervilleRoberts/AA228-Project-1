from utils import write_gph, is_cyclic
import networkx as nx
from tqdm import tqdm
import numpy as np
from bayesian_scoring import bayesian_score, bayesian_score_recompute_single_var


def rand_graph_neighbor_with_score(G, score, score_comp, df, vars, tabu=None):
    # There is a total of n(n-1) posssible actions
    n = G.number_of_nodes()

    # Tabu search
    first_iter = True
    while tabu is not None and (first_iter or (i, j, action) in tabu):
        first_iter = False
        i = np.random.randint(1, n)
        j = i
        while j == i:
            j = np.random.randint(1, n)
        if i > j: i, j = j, i
        actions = [0, 1, 2]
        if G.has_edge(i, j):
            actions.remove(1)
        elif G.has_edge(j, i):
            actions.remove(2)
        else:
            actions.remove(0)
        action = actions[0]
        possible_actions = [action for action in actions if (i, j, action) not in tabu]
        action = np.random.choice(possible_actions)
    tabu.add((i, j))

    # Generate neighbor
    G_prime = G.copy()
    recompute_i = False
    recompute_j = False
    if action == 0:
        if G.has_edge(i, j):
            G_prime.remove_edge(i, j)
            recompute_j = True
        if G.has_edge(j, i):
            G_prime.remove_edge(j, i)
            recompute_i = True
    elif action == 1:
        if G.has_edge(j, i):
            G_prime.remove_edge(j, i)
            recompute_i = True
        G_prime.add_edge(i, j)
        recompute_j = True
    elif action == 2:
        if G.has_edge(i, j):
            G_prime.remove_edge(i, j)
            recompute_j = True
        G_prime.add_edge(j, i)
        recompute_i = True

    # Check if neighbor is cyclic
    if is_cyclic(G_prime):
        return G_prime, None, (None, None), (i, j)

    # Compute new score
    score_prime, score_comp_prime_i, score_comp_prime_j = score, score_comp[i], score_comp[j]
    if recompute_i:
        score_prime, score_comp_prime_i = bayesian_score_recompute_single_var(score_prime, score_comp, vars, G_prime, df, i)
    if recompute_j:
        score_prime, score_comp_prime_j = bayesian_score_recompute_single_var(score_prime, score_comp, vars, G_prime, df, j)
    return G_prime, score_prime, (score_comp_prime_i, score_comp_prime_j), (i, j)

# Local Search algorithm
def local_search(vars, df, k_max, data_name, G=None):
    # Generate initial graph
    if G is None:
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
            if score > -425000: write_gph(G, vars, data_name=data_name, gph_name="local_best_" + str(k), score=score)
    return G, score


# Local Search algorithm with Simulated annealing, random restarts and random initializations
def random_graph_init(vars, df):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    score, score_comp = bayesian_score(vars, G, df)
    return G, score, score_comp

def local_search_with_optis(vars, df, k_max, data_name, G=None,
                            t_max=5,
                            k_max_without_improvements=2000,
                            score_improvement_to_save=5,
                            score_min_to_save=-4200,
                            log_score_every=1000,
                            return_on_restart=False):
    # Generate initial graph
    score, score_comp = None, None
    init_G = G.copy()
    if G is None:
        G, score, score_comp = random_graph_init(vars, df)
    else:
        score, score_comp = bayesian_score(vars, G, df)

    # To keep track of the best graph
    last_saved_score = -np.inf
    k_last_improvement = -1
    k_last_restart = 0
    score_of_last_improvement = - np.inf
    best_G = G.copy()
    
    # Tabu
    tabu = set()
    tabu_full_threshold = 0.9 * len(vars) * (len(vars) - 1)

    for k in tqdm(range(k_max)):
        temp = t_max * (1 - (k) / (k_max) )
        G_prime, score_prime, (score_comp_prime_i, score_comp_prime_j), (i, j) = rand_graph_neighbor_with_score(G, score, score_comp, df, vars, tabu=tabu)
        if score_prime is None: continue # This means that the graph is cyclic

        # Simulated annealing
        diff = score_prime - score
        if diff > 0 or np.random.rand() < np.exp(diff/temp):
            score = score_prime
            score_comp[i] = score_comp_prime_i  
            score_comp[j] = score_comp_prime_j   
            G = G_prime
            tabu.clear()
        
        # Random restarts
        if score > score_of_last_improvement:
            k_last_improvement = k
            score_of_last_improvement = score
            best_G = G.copy()
        if k - k_last_improvement > k_max_without_improvements or len(tabu) > tabu_full_threshold: # No improvement for k_max_without_improvements steps or Tabu is 90% full
            if return_on_restart:
                print("Stopping at step {} ({} steps without improvement)".format(k, k - k_last_improvement))
                return G, score
            print("Restarting at step {} ({} steps without improvement)".format(k, k - k_last_improvement))
            k_last_restart = k
            k_last_improvement = k
            G = best_G.copy()
            score, score_comp = bayesian_score(vars, G, df)
            tabu.clear()

        # Saving best graph
        if diff > 0 and score >= score_min_to_save and score >= last_saved_score + score_improvement_to_save:
            last_saved_score = score
            print("New best score: {}".format(score))
            write_gph(G, vars, data_name=data_name, gph_name="tabu_score_" + str(int(round(-score))), score=score)

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
    write_gph(G, vars, data_name=data_name, gph_name="local_best", score=score)