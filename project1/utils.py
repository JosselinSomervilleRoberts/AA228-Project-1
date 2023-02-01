# This file provides useful function to work with graphs.
import pandas as pd
from datetime import datetime
import networkx as nx
import os

def seconds_since_beginning_of_project():
    # Returns the amount of seconds since the 02/01/2023 00:00:00
    return int(round((datetime.now() - datetime(2023, 2, 1)).total_seconds()))

def seconds_since_beginning_of_project_at_first_execution():
    # Returns the amount of seconds since the 02/01/2023 00:00:00
    # when the function is first called and then always returns the same value.
    if not hasattr(seconds_since_beginning_of_project_at_first_execution, "first_execution"):
        seconds_since_beginning_of_project_at_first_execution.first_execution = seconds_since_beginning_of_project()
    return seconds_since_beginning_of_project_at_first_execution.first_execution

def load_data(infile):
    df = pd.read_csv(infile, delimiter=',')
    df_max = df.max()
    var_names = list(df.columns)
    df = df.groupby(var_names).size().reset_index(name='count')
    vars = [Variable(var_names[i], df_max[i]) for i in range(len(var_names))]
    return df, vars

def is_cyclic(G):
    return nx.is_directed_acyclic_graph(G) == False

def write_gph(dag, vars, data_name, gph_name, score=None):
    # create directory if not exists
    sec = seconds_since_beginning_of_project_at_first_execution()
    score_filename = "results/{}/{}/scores.txt".format(data_name, sec)
    if not os.path.exists('results/{}/{}'.format(data_name, sec)):
        os.makedirs('results/{}/{}'.format(data_name, sec))
        f = open(score_filename, "a") # Create a log score file
        f.close()

    idx2names = {i: vars[i].name for i in range(len(vars))}
    filename = "results/{}/{}/{}.gph".format(data_name, sec, gph_name)

    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

    with open(score_filename, 'a') as file:
        file.write('{} : {}\n'.format(gph_name, score))

def inneighbors(G, i):
    """Helper function for finding the parents of a variable."""
    return list(G.predecessors(i))

class Variable:
    def __init__(self, name, r):
        self.name = name
        self.r = r


if __name__ == "__main__":
    print("Seconds since beginning of project: {}".format(seconds_since_beginning_of_project()))