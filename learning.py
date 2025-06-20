# using reinforcement learning to learn the graph connectivity
# 
import numpy as np
from errorcalc import calculate_error_graph, calculate_graph_connectivity, er_sq
import gymnasium as gym
from gymnasium import spaces
import numpy as np


from scipy.optimize import minimize

def flatten_nodes(nodes):
    return nodes.flatten()

def unflatten_nodes(flat_nodes, n_nodes):
    return flat_nodes.reshape((n_nodes, 2))

def make_error_function(dist_threshold, error_type):
    a_lot = 100000000000
    def error_function(nodes, targets):
        return calculate_error_graph(nodes, targets, error_type) + (1-calculate_graph_connectivity(nodes, dist_threshold))*a_lot
    return error_function


def optimize_nodes(nodes, targets, error_function):
    n_nodes = nodes.shape[0]
    def objective(flat_nodes):
        nodes = unflatten_nodes(flat_nodes, len(targets))
        return error_function(nodes, targets)
    
    result = minimize(objective, flatten_nodes(nodes), method='L-BFGS-B')
    best_nodes = unflatten_nodes(result.x, n_nodes)
    return best_nodes


def mutate_nodes(nodes, targets, error_function, n_mutations=1000):
    # keep track of the history of the positions of the nodes
    history = []
    for i in range(n_mutations):
        error = error_function(nodes, targets)
        random = np.random.uniform(-10, 10, (nodes.shape[0], 2))
        nodes += random
        new_error = error_function(nodes, targets)
        if new_error < error:
            error = new_error
            history.append(nodes.copy())
        else:
            nodes = nodes - random
    
        # print(f"Error: {error}, New Error: {new_error}")
    nodes = np.array(nodes)
    return nodes, history



if __name__ == "__main__":
    DIST_THRESHOLD = 100
    a_lot = 100000000000
    p = 1  

    def error_function(nodes, targets):
        connectivity = calculate_graph_connectivity(nodes, DIST_THRESHOLD)
        penalty = (1 - connectivity) ** 2 * a_lot  # p=2 ici
        return calculate_error_graph(nodes, targets, er_sq) + penalty



    nodes = np.array([[0,0],[1,0],[0,1],[1,1]])
    targets = np.array([[0,50],[200,50],[0,120],[150,200]])


    best_nodes = optimize_nodes(nodes, targets, error_function)
    print(best_nodes)
    print(error_function(best_nodes, targets))