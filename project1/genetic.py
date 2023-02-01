from genetic_utils import genetic, create_graph_from_gene, plot_fitness
from bayesian_scoring import bayesian_score
import numpy as np

# This function uses the genetic algorithm to find the best bayesian network
# given an input pandas dataframe called df (provided as a parameter)
# and a list of column names called cols (provided as a parameter)
# The fitness function is the Bayesian Score of the network.
# Only part of the Bayesian score is recomputed to save time.
# The gene range is 0 to 2, which corresponds to:
# 0 = no edge
# 1 = directed edge from i to j
# 2 = directed edge from j to i
# The number of genes is the number of possible edges in the network, which is n(n-1)/2.
# The crossover rate is 0.8, the mutation rate is 0.01, the selection rate is 0.2, and the number of generations is 100.
# The population size is 1000.
# The best fitness is the Bayesian Score of the best network.
# The time is 0.5 seconds.

def fitness_function_bayesian(individual, df, vars):
    G = create_graph_from_gene(individual, vars)
    score, score_comp = bayesian_score(vars, G, df)
    if score < -10000000000: return 0
    return score + 4350


def run(df, vars, data_name):
    pop_size = 500
    num_genes = int(len(vars) * (len(vars) - 1) / 2)
    gene_range = [0, 2]
    cross_rate = 0.6
    mut_rate = 0.05
    select_rate = 0.3
    num_generations = 200
    fitness_function = lambda individual: fitness_function_bayesian(individual, df, vars)
    saving_function = lambda individual, score, iter: write_gph(create_graph_from_gene(individual, vars), vars, data_name=data_name, gph_name="gen_" + str(iter), score=score)
    best_individual, best_fitness, average_fitness, run_time = genetic(pop_size, num_genes, gene_range, cross_rate, mut_rate, select_rate, num_generations, fitness_function, saving_function=saving_function)
    print('Best Individual: ', best_individual)
    print('Best Fitness: ', best_fitness[-1])
    print('Time: ', run_time)
    plot_fitness(best_fitness, average_fitness, num_generations)
    return create_graph_from_gene(best_individual, vars), best_fitness[-1]


if __name__ == '__main__':
    import sys
    from utils import load_data, write_gph

    # Check arguments
    if len(sys.argv) != 2:
        raise Exception("usage: python genetic.py <infile>.csv")
    inputfilename = sys.argv[1]
    data_name = inputfilename.split("/")[-1].split(".")[0]

    # Load data
    df, vars = load_data(inputfilename)

    # run the genetic algorithm
    G, score = run(df, vars, data_name)
    print("Best score: {}".format(score))
    write_gph(G, vars, data_name=data_name, gph_name="best", score=score)
