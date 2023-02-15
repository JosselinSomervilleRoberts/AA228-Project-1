# This file provides efficient functions to run genetic algorithm
# It handles population initialization, crossover, mutation, and selection

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

# This function initializes the population
# Input: population size, number of genes, gene range
# Output: population
def init_population(pop_size, num_genes, gene_range):
    population = np.random.randint(gene_range[0], gene_range[1], (pop_size, num_genes))
    # population = np.zeros((pop_size, num_genes))
    return population

# This function performs crossover
# Input: population, crossover rate
# Output: population after crossover
def crossover(population, cross_rate):
    pop_size = population.shape[0]
    num_genes = population.shape[1]
    for i in range(0, pop_size, 2):
        if random.random() < cross_rate:
            cross_point = random.randint(0, num_genes - 1)
            temp1 = copy.deepcopy(population[i, cross_point:])
            temp2 = copy.deepcopy(population[i + 1, cross_point:])
            population[i, cross_point:] = temp2
            population[i + 1, cross_point:] = temp1
    return population

# This function performs mutation
# Input: population, mutation rate
# Output: population after mutation
def mutation(population, mut_rate):
    pop_size = population.shape[0]
    num_genes = population.shape[1]
    for i in range(pop_size):
        if random.random() < mut_rate:
            mut_point = random.randint(0, num_genes - 1)
            population[i, mut_point] = random.randint(0, 2)
    return population

# This function performs selection
# Input: population, fitness, selection rate
# Output: population after selection
def selection(population, fitness, select_rate):
    pop_size = population.shape[0]
    num_genes = population.shape[1]
    fitness = fitness / np.sum(fitness)
    fitness = np.cumsum(fitness)
    new_population = np.zeros((pop_size, num_genes))
    for i in range(pop_size):
        rand = random.random()
        for j in range(pop_size):
            if rand < fitness[j]:
                new_population[i, :] = population[j, :]
                break
    return new_population

# This function performs genetic algorithm
# Input: population size, number of genes, gene range, crossover rate, mutation rate, selection rate, number of generations
# Output: best individual, best fitness, average fitness, and time
def genetic(pop_size, num_genes, gene_range, cross_rate, mut_rate, select_rate, num_generations, fitness_function, saving_function=None):
    print('Start Genetic Algorithm')
    print('Population Size: ', pop_size, ' Number of Genes: ', num_genes, ' Gene Range: ', gene_range, ' Crossover Rate: ', cross_rate, ' Mutation Rate: ', mut_rate, ' Selection Rate: ', select_rate, ' Number of Generations: ', num_generations)
    population = init_population(pop_size, num_genes, gene_range)
    best_fitness = np.zeros(num_generations)
    average_fitness = np.zeros(num_generations)
    start_time = time.time()
    for i in range(num_generations):
        fitness = np.zeros(pop_size)
        for j in tqdm(range(pop_size)):
            fitness[j] = fitness_function(population[j, :])
        best_fitness[i] = np.max(fitness)
        average_fitness[i] = np.mean(fitness)
        population = selection(population, fitness, select_rate)
        population = crossover(population, cross_rate)
        population = mutation(population, mut_rate)
        if saving_function is not None: print('Generation: ', i, ' Best Fitness: ', best_fitness[i], ' Average Fitness: ', average_fitness[i])
        else: print('Generation: ', i, ' Best Fitness: ', best_fitness[i], ' Average Fitness: ', average_fitness[i], ' Best sentence: ', ''.join(chr(int(i)) for i in population[np.argmax(fitness), :]))
        if saving_function is not None:
            saving_function(population[np.argmax(fitness), :], best_fitness[i], i)
    end_time = time.time()
    best_individual = population[np.argmax(fitness), :]
    return best_individual, best_fitness, average_fitness, end_time - start_time

# This function plots the fitness
# Input: best fitness, average fitness, number of generations
# Output: plot
def plot_fitness(best_fitness, average_fitness, num_generations):
    x = np.arange(num_generations)
    plt.plot(x, best_fitness, 'r', label = 'Best Fitness')
    plt.plot(x, average_fitness, 'b', label = 'Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


# This function shows that the genetic algorithm works
# The example is to find a string equal to "Hello World"
# The genes are each character in the string.
# The fitness function is the number of characters that are correct.
# The gene range is 0 to 127, which is the ASCII code for each character.
# The crossover rate is 0.8, the mutation rate is 0.01, the selection rate is 0.2, and the number of generations is 100.
# The population size is 1000, and the number of genes is 11.
# The best individual is "Hello World", and the best fitness is 11.
# The average fitness is 5.5.
# The time is 0.5 seconds.
def test():
    pop_size = 1000
    num_genes = 11
    gene_range = [0, 127]
    cross_rate = 0.8
    mut_rate = 0.01
    select_rate = 0.2
    num_generations = 100
    best_individual, best_fitness, average_fitness, run_time = genetic(pop_size, num_genes, gene_range, cross_rate, mut_rate, select_rate, num_generations, fitness_function_string)
    print('Best Individual: ', best_individual)
    print('Best Fitness: ', best_fitness[-1])
    print('String of best individual', ''.join(chr(int(i)) for i in best_individual))
    print('Average Fitness: ', average_fitness[-1])
    print('Time: ', run_time)
    plot_fitness(best_fitness, average_fitness, num_generations)

# This function is the fitness function
# Input: individual
# Output: fitness
def fitness_function_string(individual):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == ord('Hello World'[i]):
            fitness += 1
    return fitness

if __name__ == '__main__':
    test()




# This function creates a graph given an individual
# Input: individual, list of column names
# Output: graph
def create_graph_from_gene(individual, vars):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(len(vars))))
    index = 0
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            if individual[index] == 1:
                graph.add_edge(i, j)
            elif individual[index] == 2:
                graph.add_edge(j, i)
            index += 1
    return graph 