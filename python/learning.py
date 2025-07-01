# using reinforcement learning to learn the graph connectivity
# 
import numpy as np
from errorcalc import calculate_graph_connectivity, cout_snt, cout_total, cout_min
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from scipy.optimize import minimize

def flatten_nodes(nodes):
    return nodes.flatten()

def unflatten_nodes(flat_nodes, n_nodes):
    return flat_nodes.reshape((n_nodes, 2))

def make_error_function(dist_threshold, error_type, error_power):
    a_lot = 100000000000
    def error_function(nodes, targets):
        return error_type(nodes, targets, error_power) + (1-calculate_graph_connectivity(nodes, dist_threshold))*a_lot
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
    return nodes, history

import numpy as np

def random_point_weighted_union_circles_unbiased(centers, radius, attract_point, sigma, max_trials=10000):
    centers = np.asarray(centers)
    attract_point = np.asarray(attract_point)
    n = len(centers)
    # Rectangle englobant
    min_xy = np.min(centers, axis=0) - radius
    max_xy = np.max(centers, axis=0) + radius
    w_max = 1.0  # max de la gaussienne

    for _ in range(max_trials):
        # 1. Tirer un point uniformément dans le rectangle
        point = np.random.uniform(min_xy, max_xy)
        # 2. Vérifier s'il est dans au moins un cercle
        dists = np.linalg.norm(centers - point, axis=1)
        if np.any(dists <= radius):
            # 3. Pondération
            w = np.exp(-np.sum((point - attract_point)**2) / (2 * sigma**2))
            if np.random.uniform(0, w_max) < w:
                print(f"took {_} trials")
                return point
    raise RuntimeError("Échec du tirage après {} essais".format(max_trials))


def get_adjacency_matrix(nodes, dist_threshold):
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                adjacency_matrix[i][j] = 0
            else:
                adjacency_matrix[i][j] = 1 if np.linalg.norm(nodes[i] - nodes[j]) < dist_threshold else 0
    return adjacency_matrix

def adjacency_matrix_to_neighboring_nodes(nidx,adjacency_matrix):
    """
    retournes une liste d'index de noeuds qui sont à une distance croissante de nidx
    selon adjacency_matrix
    """
    neighboring_nodes = [nidx]
    visited_nodes = {nidx}
    while len(neighboring_nodes) < len(adjacency_matrix):
        next_nodes = []
        for node in neighboring_nodes:
            for i in range(len(adjacency_matrix)):
                if adjacency_matrix[node][i] == 1 and i not in visited_nodes:
                    next_nodes.append(i)
                    visited_nodes.add(i)
        neighboring_nodes.extend(next_nodes)
        
    return neighboring_nodes

def random_points_on_circle_with_attraction_point(circle_center, radius, attract_point, beta_1=2, beta_2=2):
    # Génère tous les dist et angles d'un coup
    dist = 1-np.random.beta(1, beta_1)
    dist = np.sqrt(dist) * radius

    angles = np.random.uniform(-np.pi, np.pi)
    angle_attract = np.arctan2(attract_point[1] - circle_center[1], attract_point[0] - circle_center[0])
    alpha = 1-np.random.beta(1, beta_2)
    angles = (1 - alpha) * angles + alpha * angle_attract
    angles = angles % (2*np.pi)

    x = circle_center[0] + dist * np.cos(angles)
    y = circle_center[1] + dist * np.sin(angles)
    return np.column_stack((x, y))


def chose_node_near_node_weighted(nodes, nidx, dist_threshold):
    """
    chose randomly a node from nodes, the farther the node, the more unlikely it is to be chosen
    """
    nodes_without_nidx = nodes.copy()
    nodes_without_nidx = np.delete(nodes_without_nidx, nidx, axis=0)
    dists = np.linalg.norm(nodes_without_nidx - nodes[nidx], axis=1)
    weights = np.exp(-dists / dist_threshold)
    weights = weights / np.sum(weights)
    idx = np.random.choice(len(nodes_without_nidx), p=weights)
    return nodes_without_nidx[idx]



def safe_mutate_nodes(nodes, dist_threshold:float=1.1, sigma:float=10):
    # Assure que les nœuds sont connectés
    adjacency_matrix = get_adjacency_matrix(nodes, dist_threshold)
    start_node = np.random.randint(len(nodes))
    order = adjacency_matrix_to_neighboring_nodes(start_node, adjacency_matrix)
    new_nodes = [nodes[start_node]]
    print(order)
    print(order[1:])
    for nidx in order[1:]:
        # On utilise les nœuds déjà placés pour générer le suivant
        chosen_node = chose_node_near_node_weighted(new_nodes, nidx, dist_threshold)
        new_node = random_points_on_circle_with_attraction_point(nodes[nidx], dist_threshold, chosen_node, 2, 2)
        new_nodes.append(new_node)
    return np.array(new_nodes)

def random_point_on_circle_with_attraction_point(circle_center, radius, attract_point, sigma):
    d = np.linalg.norm(circle_center - attract_point)
    fun = lambda x: np.exp(-x)
    max_angle = 2*np.arctan(radius/d)
    angle = np.random.uniform(0, max_angle)
    a = d*np.tan(angle)**2 + np.sqrt(radius**2)





if __name__ == "__main__":
    DIST_THRESHOLD = 100
    a_lot = 100000000000
    p = 1  

    def error_function(nodes, targets):
        connectivity = calculate_graph_connectivity(nodes, DIST_THRESHOLD)
        penalty = (1 - connectivity) ** 2 * a_lot  # p=2 ici
        return cout_total(nodes, targets) + penalty



    nodes = np.array([[0,0],[1,0],[0,1],[1,1]])
    targets = np.array([[0,50],[200,50],[0,120],[150,200]])


    # nodes = optimize_nodes(nodes, targets, error_function=make_error_function(DIST_THRESHOLD, cout_snt, 2))
    # best_nodes = optimize_nodes(nodes, targets, error_function)
    # print(best_nodes)
    # print(error_function(best_nodes, targets))

    new_nodes = safe_mutate_nodes(nodes, sigma=0.1)
    print(new_nodes)
    # print(error_function(new_nodes, targets))


    from matplotlib import pyplot as plt
    plt.scatter(nodes[:,0], nodes[:,1], color='blue')
    plt.scatter(new_nodes[:,0], new_nodes[:,1], color='green', marker='x')
    # plt.scatter(targets[:,0], targets[:,1], color='red')
    plt.show()

