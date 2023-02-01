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
    return score


def genetic_algorithm(fitness_fn, size_population, nb_genes,
                     num_generations, num_elites, probability_crossover, 
                     probability_mutation):
    # Initialize the population
    population = init_population(size_population, nb_genes)
    #print(population[0])

    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population
        # population_fitness = [(individual, fitness_fn(individual)) for individual in population]
        population_fitness = []
        for individual in tqdm(population):
            #print("indiv:", individual)
            population_fitness.append((individual, fitness_fn(individual)))
        
        # Sort the population by fitness in descending order
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select the elites individuals
        elites = [individual for (individual, fitness) in population_fitness[:num_elites]]
        
        # Generate the offspring for the next generation
        offspring = elites[:]
        while len(offspring) < size_population:
            # Select two parents using the tournament selection method
            parent1, parent2 = tournament_selection(population_fitness, 2)
            
            # Crossover with a probability of `probability_crossover`
            if random.random() < probability_crossover:
                child1, child2 = crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1)
                offspring.append(parent2)
        
        # Mutate the offspring with a probability of `probability_mutation`
        for i in range(len(offspring)):
            if random.random() < probability_mutation:
                mutate(offspring[i])
        
        # Replace the population with the offspring
        population = offspring[:]

        # Displays useful information about the current generation
        print(f"Generation {generation + 1} | Best fitness: {population_fitness[0][1]}")
    
    # Return the best individual from the final population
    population_fitness = [(individual, fitness_fn(individual)) for individual in population]
    population_fitness.sort(key=lambda x: x[1], reverse=True)
    return population_fitness[0][0], population_fitness[0][1]

def tournament_selection(population_fitness, size_tournament):
    tournament = random.sample(population_fitness, size_tournament)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0], tournament[1][0]

def mix_genes(genes1, genes2, i, j):
    new_genes = np.zeros(len(genes1), dtype=int)
    new_genes[i:j] = genes1[i:j]
    remaining = [gene for gene in genes2 if gene not in genes1[i:j]]
    new_genes[:i] = remaining[:i]
    new_genes[j:] = remaining[i:]
    return new_genes

def crossover(parent1, parent2):
    # randomly keeps a section of parent1 and fills the rest by the order specified by the gene of parent2
    # It is not possible to simply fill the missing genes byt the genes of parent2 as we are modeling eprmutations
    # and therefore each gene can only appear once
    i = random.randint(0, len(parent1) - 1)
    j = random.randint(0, len(parent1) - 1)
    if i > j:
        i, j = j, i
    child1 = mix_genes(parent1, parent2, i, j)
    child2 = mix_genes(parent2, parent1, i, j)
    #print("child 1:", child1)
    #print("child 2:", child2)
    return child1, child2

def mutate(individual):
    # randomly swaps two genes of the individual
    i = random.randint(0, len(individual) - 1)
    j = random.randint(0, len(individual) - 1)
    individual[i], individual[j] = individual[j], individual[i]


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

    # Compute empty score
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(vars))))
    empty_score, empty_score_comp = bayesian_score(vars, G, df)

    # Constants
    max_parents = 4

    # Define the fitness function
    fitness_fn = lambda individual: fitness_fn_permutation(individual, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp.copy())

    # Run the genetic algorithm
    best_ordering, best_score = genetic_algorithm(fitness_fn, nb_genes=len(vars), size_population=100, num_generations=100, num_elites=10, probability_crossover=0.8, probability_mutation=0.1)
    best_individual, best_score = k2(best_ordering, vars, df, max_parents=max_parents, empty_score=empty_score, empty_score_comp=empty_score_comp)
    print("Best score: {}".format(best_score))
    write_gph(G, vars, data_name=data_name, gph_name="best", score=best_score)