# This file is mostly a duplicate of genetic_utils.py with some variations
# of the algorithm to handle permutations

import random
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import networkx as nx
from tqdm import tqdm
from utils import write_gph
import random
from k2_search import k2
from bayesian_scoring import bayesian_score
from heapq import heappush, heappushpop, heappop
from local_search import local_search_with_optis

# This function initializes the population
# Input: population size, number of genes, gene range
# Output: population
def init_population(pop_size, num_genes):
    population = np.array([np.random.permutation(num_genes) for i in range(pop_size)])
    return population



def fitness_fn_permutation(individual, vars, df, max_parents=2, empty_score=None, empty_score_comp=None):
    # This function computes the fitness of an individual by computing
    # the bayesian score of the best graph found by the K2 algorithm
    # with the individual as the ordering of the variables

    # Compute the best graph found by the K2 algorithm
    G, score = k2(individual, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp)
    return score, G


class Record:

    def __init__(self, fitness, individual, G=None):
        self.fitness = fitness
        self.individual = individual
        self.G = G

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


def genetic_algorithm(fitness_fn, size_population, nb_genes,
                     num_generations, fraction_elites, probability_crossover, 
                     probability_mutation,
                     data_name, vars,
                     keep_best_nb=10,
                     nb_gens_max_without_improvement=10):
    # Print parameters of the algorithm
    num_elites = int(size_population - 2 * np.ceil(((1 -fraction_elites) * size_population) / 2))
    print(f"Population size: {size_population} | Number of generations: {num_generations} | Number of elites: {num_elites} | Probability of crossover: {probability_crossover} | Probability of mutation: {probability_mutation}")
    print(f"Number of genes: {nb_genes} | Data name: {data_name} | Keep best nb: {keep_best_nb} | Max gens without improvement: {nb_gens_max_without_improvement}")

    # Initialize the population
    population = init_population(size_population, nb_genes)
    fitness_scores = np.zeros(size_population)
    new_fitness = [None] * size_population
    best_individuals = []
    G = [None] * size_population
    idx_best_saved = 0

    # best score (for ealy stopping)
    best_score = -np.inf
    last_gen_improvement = -1

    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population
        for i, individual in enumerate(tqdm(population)):
            if new_fitness[i] is None: # If the individual was modified, it needs to be recomputed
                fitness_scores[i], G[i] = fitness_fn(individual)
            else: # it is an individual from the previous generation, so we already know it's score
                fitness_scores[i] = new_fitness[i]
        new_fitness = [None] * size_population
        
        # Sort the population by fitness in descending order
        indices = np.argsort(-fitness_scores)
        for i in range(min(keep_best_nb, size_population)):
            rec = Record(fitness_scores[indices[i]], population[indices[i]], G[indices[i]])
            if len(best_individuals) < keep_best_nb:
                heappush(best_individuals, rec)
                idx_best_saved += 1
                write_gph(rec.G, vars, data_name=data_name, gph_name="genetic_" + str(idx_best_saved), score=fitness_scores[indices[i]])
            elif not rec in best_individuals:
                rec_popped = heappushpop(best_individuals, rec)
                if rec_popped.fitness == fitness_scores[indices[i]]: # if the individual was not added to the heap
                    # no individual will be added since they are sorted, so it will only get worse
                    break
                else: # new best individual, so we save its graph
                    idx_best_saved += 1
                    write_gph(rec.G, vars, data_name=data_name, gph_name="genetic_" + str(idx_best_saved), score=fitness_scores[indices[i]])

        # Displays useful information about the current generation
        print(f"Generation {generation + 1} | Best fitness: {fitness_scores[indices[0]]} | Average fitness: {np.mean(fitness_scores)}")

        # Early stoping if no improvement for X generations
        if fitness_scores[indices[0]] > best_score:
            best_score = fitness_scores[indices[0]]
            last_gen_improvement = generation
        elif generation - last_gen_improvement >= nb_gens_max_without_improvement:
            break
        
        # Select the elites individuals
        elites = [population[idx] for idx in indices[:num_elites]]
        new_fitness[:num_elites] = fitness_scores[indices[:num_elites]]
        
        # Generate the offspring for the next generation
        offspring = elites[:]
        while len(offspring) < size_population:
            # Select two parents using the tournament selection method
            parent1, fitness_parent_1, parent2, fitness_parent_2 = tournament_selection(population, fitness_scores, 2)
            
            # Crossover with a probability of `probability_crossover`
            if random.random() < probability_crossover:
                child1, child2 = crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                new_fitness[len(offspring)] = fitness_parent_1
                offspring.append(parent1)
                new_fitness[len(offspring)] = fitness_parent_2
                offspring.append(parent2)
        
        # Mutate the offspring with a probability of `probability_mutation`
        for i in range(len(offspring)):
            if random.random() < probability_mutation:
                mutate(offspring[i])
                new_fitness[i] = None
        
        # Replace the population with the offspring
        population = np.array(offspring)
    
    # Return the best individuals found
    return best_individuals

def tournament_selection(population, fitness_scores, size_tournament):
    indices = np.random.choice(len(population), size_tournament, replace=False)
    tournament = population[indices]
    tournament_fitness = fitness_scores[indices]
    indices = np.argsort(-tournament_fitness)
    return tournament[indices[0]], tournament_fitness[indices[0]], tournament[indices[1]], tournament_fitness[indices[1]]

def mix_genes(genes1, genes2, i, j):
    new_genes = np.zeros(len(genes1), dtype=int)
    new_genes[i:j] = genes1[i:j]
    remaining = [gene for gene in genes2 if gene not in genes1[i:j]]
    new_genes[:i] = remaining[:i]
    new_genes[j:] = remaining[i:]
    return new_genes

def crossover(parent1, parent2):
    # randomly keeps a section of parent1 and fills the rest by the order specified by the gene of parent2
    # It is not possible to simply fill the missing genes byt the genes of parent2 as we are modeling permutations
    # and therefore each gene can only appear once
    i = random.randint(0, len(parent1) - 1)
    j = random.randint(0, len(parent1) - 1)
    if i > j:
        i, j = j, i
    child1 = mix_genes(parent1, parent2, i, j)
    child2 = mix_genes(parent2, parent1, i, j)
    return child1, child2

def mutate(individual):
    # randomly swaps two genes of the individual
    i = random.randint(0, len(individual) - 1)
    j = i
    while j == i: j = random.randint(0, len(individual) - 1)
    individual[i], individual[j] = individual[j], individual[i]


if __name__ == "__main__":
    import sys
    from utils import load_data, write_gph, load_gph

    # Check arguments
    if len(sys.argv) < 2:
        raise Exception("usage: python k2_search.py <infile>.csv")
    inputfilename = sys.argv[1]
    data_name = inputfilename.split("/")[-1].split(".")[0]

    # Load data
    df, vars = load_data(inputfilename)

    # If it's only reoptimization
    best_G = None
    only_reoptimization = False
    if len(sys.argv) == 3:
        gph_name = sys.argv[2]
        best_G = load_gph(gph_name, vars)
        only_reoptimization = True
        print("Reoptimization of the graph " + gph_name)

    if not only_reoptimization:
        # Compute empty score
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(vars))))
        empty_score, empty_score_comp = bayesian_score(vars, G, df)

        # Constants
        max_parents = 4

        # Define the fitness function
        fitness_fn = lambda individual: fitness_fn_permutation(individual, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp.copy())

        # Run the genetic algorithm
        best_individuals = genetic_algorithm(fitness_fn,
                                            nb_genes=len(vars),
                                            size_population=30,
                                            num_generations=10,
                                            fraction_elites=0.18,
                                            probability_crossover=0.8,
                                            probability_mutation=0.1,
                                            data_name=data_name,
                                            vars=vars,
                                            nb_gens_max_without_improvement=3)
        best_ordering, best_score, best_G = None, None, None
        while len(best_individuals) > 0:
            rec = heappop(best_individuals)
            best_score = rec.fitness
            best_ordering = rec.individual
            best_G = rec.G

        if best_G is None: best_G, best_score = k2(best_ordering, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp)
        print("Best score: {}".format(best_score))
        write_gph(best_G, vars, data_name=data_name, gph_name="best", score=best_score)

    # Improve with local search
    best_G, best_score = local_search_with_optis(vars, df, k_max=100000, data_name=data_name, G=best_G, t_max=40.0,
                            k_max_without_improvements=1000,
                            score_improvement_to_save=1.0,
                            score_min_to_save=-np.inf,
                            log_score_every=100,
                            return_on_restart=False)
    print("Best score after local search: {}".format(best_score))
    write_gph(best_G, vars, data_name=data_name, gph_name="best_after_optims", score=best_score)