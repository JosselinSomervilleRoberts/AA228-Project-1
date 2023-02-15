# Structure Learning
This project aims to **predict Bayesian networks** given instances of some variables. It was the first project of **AA228 - Decision Under Uncertainty at Stanford**.

## Overview of the algorithms tried

I have tried many algorithms:

- Iteration through random orderings to perform K2
- Local Search with random initialization
- Local Search with restarts, simulated annealing and tabu
- Local Search with restarts, simulated annealing and tabu (using K2 as initialization)
- Genetic algorithm on graphs using genes between 0 and 2 to represent:
    o 0: no edge between $i$ and $j$
    o 1: edge from $i$ to $j$
    o 2: edge from $j$ to $i$
- Genetic algorithm on orderings that are then used for K2
- Genetic algorithm on orderings that are then used for K2 followed by a local search with
    restarts, simulated annealing and tabu (this one gave the best results).

## Bayesian score Optimization

One key aspect is that I very much optimized the computation of the Bayesian score.

- I simplified the Bayesian_score_component function, using the fact that alpha is simply a
    matrix full of ones. So we can get rid of the construction of alpha and only computing its
    shape and then adapt the function using the fact that alpha is full of ones:
    ```python
    def bayesian_score_component(M, alpha_shape):
    """Algorithm 5.1 - Page 98 of the book - Helper function."""
    # I've optimized the next line by using the fact that alpha is a vector of 1s
    # p = np.sum(scipy.special.loggamma(alpha + M))
    p = np.sum(scipy.special.loggamma(1 + M))

    # I've removed the next line because with a prior of 1, the loggamma of alpha is 0
    # p -= np.sum(scipy.special.loggamma(alpha))

    # The next line has been removed to be optimized by what follows (using the fact that alpha is a vector of 1s)
    # p += np.sum(scipy.special.loggamma(np.sum(alpha, axis=1)))
    p += alpha_shape[0] * np.log(scipy.special.factorial(alpha_shape[1] - 1))

    # I've optimized the next line by using the fact that alpha is a vector of 1s
    # p -= np.sum(scipy.special.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    p -= np.sum(scipy.special.loggamma(alpha_shape[1] + np.sum(M, axis=1)))
    return p
    ```
- I only recomputed the component that changed when adding a parent rather than the entire
    score.
- I did two group by. First, when I loaded the data, I grouped it by identical realizations. Then,
    what really made a difference is when I compute $M$, I group by the data by the realization of the node and its parent, which makes it so that instead of looping through 10k lines, we only get a few hundreds at most. Also I only recomputed the $M[i]$ needed and all the Ms, just as explained in the previous point.
    ```python
    def statistics_for_single_var(vars, G, df, var_index):
    """Computes M for a single var_index.
    This version is optimized for speed.
    It groups the data by parents and data instantiation to reduce massively the number of iterations."""

    q_var = np.prod(np.array([vars[j].r for j in inneighbors(G,var_index)], dtype=int))
    M_var = np.zeros((q_var, vars[var_index].r))
    parents = inneighbors(G,var_index)
    r_parents = np.array([vars[j].r for j in parents])
    has_no_parent = len(parents) == 0
    df2 = df.groupby(by=[vars[i].name for i in [var_index] + (parents)])['count'].sum().reset_index()
    # A row in df2 is now: [var, parent1, parent2, ..., count]
    # Which also simplifies the slicing

    for index, row in df2.iterrows():
        k = row[0] - 1 # value of variable
        j = 0 if has_no_parent else sub2ind(r_parents, row[1:-1] - 1)
        M_var[j,k] += row[-1]
    return M_var
    ```

- I precomputed a lot of terms that were computed at each iteration in the algorithm such as
    parents, r_parents, ...

**In the end, K2 ran in about 1.2s for the medium dataset with all the optimization compared to the 2 minutes people were talking about on Ed.**

## Local Search

For local search, I added restarts if after no more than X iterations, no improvement was found. I also added Simulated annealing, which simply consists in:
```python
# Simulated annealing
diff = score_prime - score
if diff > 0 or np.random.rand() < np.exp(diff/temp):
```

Then I added a tabu. To understand the tabu, I will first explain how I generate the neighbors:

- First we chose randomly $i$, $j$ such that $i$ < $j$.
- Then, we chose an action 0, 1 or 2 corresponding to:
        o 0: Delete edge between $i$ and $j$ or $j$ and $i$
        o 1: edge from $i$ to $j$ and delete edge from $j$ to $i$ if there is one
        o 2: edge from $j$ to $i$ and delete edge from $i$ to $j$ if there is one
Of course for a given graph, only two actions are possible for each edge. So there are $n*(n- 1 ) / 2 * 2 = n(n-1)$ actions possible per state.

- We keep track of this tuple $(i,j,a)$ and if it is already in the tabu, we don’t do the action and repeat the previous steps until finding a new action.


## Genetic algorithm

The genetic algorithm supported all the vanilla concepts of genetic algorithms: selection, crossover, mutation and elitism. For the population initialization, I chose to either have empty graphs, random graphs or graphs generated by K2 from random orderings.

I implemented a genetic algorithm on the graphs. There were $n(n-1)/2$ genes corresponding for each to the state between $i$ and $j$:
- 0 : no edge between $i$ and $j$
- 1: edge from $i$ to $j$
- 2: edge from $j$ to $i$

The crossover was implemented as simply taking a subpart of the gene of parent 1 and filling the rest with parent 2. The mutation simply consisted in changing a gene.

## Genetic algorithm on orderings

I tried to do a genetic algorithm on orderings to produce the best K2 solution. The genes were therefore permutations of $[1, ..., n]$. The evaluation simply consisted on running K2 with the genes as the ordering and then computing the Bayesian score (this was expensive).

The mutation consisted of simply swapping two random nodes in the ordering.

The crossover was a bit more interesting as a node can only appear once. The idea was therefore to take a subpart of the ordering from parent 1. Then fill the rest with the ordering of parent 2 that had the subpart of parent 1 chosen removed. Here is the algorithm:

```python
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
```

## Final algorithm

I used the Genetic algorithm on orderings and the after a certain number of generations selected the best individual and ran a local search with tabu, simulated annealing (with a quite low temperature) and tabu to optimize it even more.


## Running time

One run of K2:

- Small: about 0.2 seconds
- Medium: about 1.2 seconds
- Large: about 120 seconds

Final algorithm full pipeline:

- Small (20 generations of 200 followed by 1000 iterations of local search): about 20 minutes
- Medium ( 2 0 generations of 100 followed by 5000 iterations of local search): about 45
    minutes
- Large (5 generations of 30 followed by 20000 iterations of local search): about 6 hours.