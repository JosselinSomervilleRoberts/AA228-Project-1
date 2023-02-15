import sys
import scipy.special
import numpy as np
import networkx as nx
import csv
from dash import Dash, html
import dash_cytoscape as cyto
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm








def compute(infile, outfile):
    df = pd.read_csv(infile, delimiter=',')
    df_max = df.max()
    var_names = list(df.columns)
    df = df.groupby(var_names).size().reset_index(name='count')
    vars = [Variable(var_names[i], df_max[i]) for i in range(len(var_names))]
    
    # JUST FOR TESTING
    # G = nx.DiGraph()
    # for i in range(len(vars)): G.add_node(i)
    # for i in range(len(vars)//2): G.add_edge(2*i, 2*i+1)

    # NUM_ITER_TIMEIT = 100
    # from time import time

    # initial_score, init_comp = bayesian_score(vars, G, df)
    # print("Initial Bayesian score: {}".format(initial_score))
    # print("Initial Bayesian score components: {}".format(init_comp))

    # G.add_edge(0, 2)
    # t_start = time()
    # for _ in tqdm(range(NUM_ITER_TIMEIT)):
    #     new_score, new_comp = bayesian_score(vars, G, df)
    # t_end = time()
    # print("\nNew Bayesian score: {}".format(new_score))
    # print("New Bayesian score components: {}".format(new_comp))
    # print("Time taken for {} iterations: {} s".format(NUM_ITER_TIMEIT, round(t_end - t_start, 2)))

    # t_start = time()
    # for _ in tqdm(range(NUM_ITER_TIMEIT)):
    #     clever_score, clever_comp = bayesian_score_recompute_single_var(initial_score, init_comp, vars, G, df, 2)
    # t_end = time()
    # clever_compoments = init_comp.copy()
    # clever_compoments[2] = clever_comp
    # print("\nClever Bayesian score: {}".format(clever_score))
    # print("Clever Bayesian score components: {}".format(clever_compoments))
    # print("Time taken for {} iterations: {} s".format(NUM_ITER_TIMEIT, round(t_end - t_start, 2)))

    # k2_iter(vars, df, 1000, max_parents=4, name="large")
    local_search(vars, df, k_max=100000, name="small_local")


def k2_iter(vars, df, num_iter, max_parents=2, name="small"):
    #past_orderings = set()
    best_score = -np.inf
    best_G = None

    # Compute empty score
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    empty_score, empty_score_comp = bayesian_score(vars, G, df)

    # idx2names
    idx2names = {i: vars[i].name for i in range(len(vars))}

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
            write_gph(best_G, idx2names, "results/best_" + name + "_" + str(idx) + ".gph")
            print("New best score: {}".format(best_score))
    return best_G, best_score


# K2 algorithm
def k2(ordering, vars, df, max_parents=2, empty_score=None, empty_score_comp=None):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(ordering))))
    score, score_comp = empty_score, empty_score_comp
    if score is None or score_comp is None: score, score_comp = bayesian_score(vars, G, df)
    for (k, i) in enumerate(ordering[1:]):
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



def is_cyclic(G):
    return nx.is_directed_acyclic_graph(G) == False

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
def local_search(vars, df, k_max, name):
    # Generate initial graph
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    score, score_comp = bayesian_score(vars, G, df)
    idx2names = {i: vars[i].name for i in range(len(vars))}

    for k in tqdm(range(k_max)):
        G_prime, score_prime, score_comp_prime, j = rand_graph_neighbor_with_score(G, score, score_comp, df, vars)
        if is_cyclic(G_prime):
            continue
        if score_prime > score:
            score = score_prime
            score_comp[j] = score_comp_prime    
            G = G_prime
            print("New best score: {}".format(score))
            if score > -425000: write_gph(G, idx2names, "results/best_" + name + "_" + str(k) + ".gph")


# Local Search algorithm with Simulated annealing, random restarts and random initializations
def random_graph_init(vars, df):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    score, score_comp = bayesian_score(vars, G, df)
    return G, score, score_comp

def local_search_with_optis(vars, df, k_max, name,
                            t_max=5,
                            k_max_without_improvements=2000,
                            score_improvement_to_save=5,
                            score_min_to_save=-4200,
                            log_score_every=1000):
    # Generate initial graph
    G, score, score_comp = random_graph_init(vars, df)
    idx2names = {i: vars[i].name for i in range(len(vars))}

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
            write_gph(G, idx2names, "results/best_" + name + "_" + str(int(round(-score))) + ".gph")

        # Logging
        if k % log_score_every == 0:
            print("Current score: {}".format(score))


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
