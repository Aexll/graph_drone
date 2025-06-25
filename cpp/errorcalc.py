import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx


"""





Fonction necessaire pour le calcul de l'erreur entre un graph et un set de points que l'on note "cout" du graph 
associé à un set de points.

plusieurs fonctions de cout sont possibles, on peut choisir la fonction de cout en fonction de l'application.







"""

def distance(p1, p2):
    """
    euclidean distance between two points
    """
    return np.linalg.norm(p1 - p2)

def cout_snt(nodes, targets,  n=1):
    """
    > single node to target distance
    > nodes and targets should both have the same shape
    > n is the power of the distance
    > dist_threshold is the distance threshold for the graph
    > if n is 1, the function returns the sum of the distances
    > if n is 2, the function returns the sum of the squared distances (sqrted)
    > if n is inf, the function returns the max of the distances (n-rooted)
    """
    return np.sum(distance(nodes, targets)**n)**(1/n)

def cout_total(nodes, targets, n=1):
    """
    > total distance between all nodes and all targets
    > n is the power of the distance
    > dist_threshold is the distance threshold for the graph
    """
    dists = np.linalg.norm(nodes[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2)
    return np.sum(dists**n)**(1/n)

def cout_min(nodes, targets, n=1):
    """
    > sum for each target of the minimum distance between the target and a node
    > n is the power of the distance
    """
    dists = np.linalg.norm(nodes[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=2)
    min_dists = np.min(dists, axis=0)
    return np.sum(min_dists**n)**(1/n)

## Graphs matrix  

def nodes_to_matrix(nodes,dist_threshold):
    diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # we exclude self-connections
    np.fill_diagonal(dists, np.inf)
    # we set the value to 1 if the distance is less than CONEXION_RADIUS, 0 otherwise
    ret = (dists < dist_threshold).astype(float)
    return ret


def get_neighbors(i,graph):
    return np.where(graph[i] == 1)[0]

connected_memory = {}
def connected(i,n,graph):
    if n <= 0: 
        v = np.zeros(len(graph))
        v[i] = 1
        return v
    if (i, n) in connected_memory:
        return connected_memory[(i, n)]

    arrays = [connected(l, n-1, graph) for l in set(get_neighbors(i,graph)).union({i})]
    v = np.maximum.reduce(arrays)
    connected_memory[(i, n)] = v
    return v


def calculate_graph_connectivity(nodes,dist_threshold):
    connected_memory.clear()
    graph = nodes_to_matrix(nodes, dist_threshold)
    return not (0 in connected(0, len(nodes), graph))


# BFS mieux optimisé
def is_graph_connected(nodes, dist_threshold):
    graph = nodes_to_matrix(nodes, dist_threshold)
    n = len(nodes)
    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    while queue:
        current = queue.pop(0)
        neighbors = np.where(graph[current] == 1)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return visited.all()

def is_graph_connected_nx(nodes, dist_threshold):
    graph = nodes_to_matrix(nodes, dist_threshold)
    G = nx.from_numpy_array(graph)
    return nx.is_connected(G)


# print(is_graph_connected_nx(np.array([[0,0],[1,0],[0,1],[1,1]]), 1.1))








# Erreur


# def er_sq(nodes, targets):
#     """
#     sum of all squared distances between nodes and targets sqrted
#     """
#     return np.sum(np.linalg.norm(nodes - targets, axis=1)**2)**0.5

# def er_lin(nodes, targets):
#     """
#     return euclidean distance between nodes and targets
#     """
#     return np.average(np.linalg.norm(nodes - targets, axis=1))

# def er_max(nodes, targets):
#     """
#     max of all distances between nodes and targets
#     """
#     return np.max(np.linalg.norm(nodes - targets, axis=1))


# def calculate_error_graph(nodes, targets, error_function):
#     return error_function(nodes, targets)

def error_function_wrapper(nodes, targets, dist_threshold, cost_function, cost_power):
    alot = 1000000
    if is_graph_connected(nodes, dist_threshold):
        return cost_function(nodes, targets, cost_power)
    else:
        return cost_function(nodes, targets, cost_power) + alot

def mutate_nodes(nodes, targets, start_error, stepsize, error_function):
    # 1 - muter les noeuds
    new_nodes = nodes + np.random.uniform(-stepsize, stepsize, nodes.shape).astype(np.float32)
    # 2 - calculer l'erreur
    error = error_function(new_nodes, targets)
    # 3 - accepter ou refuser la mutation
    if error < start_error:
        return new_nodes, error
    else:
        return nodes, start_error

def mutate_nodes_genetic_sampling(nodes, targets, start_error, stepsize, error_function, sampling_size):
    candidates = [mutate_nodes(nodes, targets, start_error, stepsize, error_function) for _ in range(sampling_size)]
    # Each candidate is a tuple: (nodes, error)
    best_nodes, best_error = min(candidates, key=lambda x: x[1])
    return best_nodes, best_error




def calc_optimal_graph(targets, dist_threshold, cost_function, cost_power, steps=10000, mutation_stepsize=1, sampling_size=10, use_genetic_sampling=False):
    barycenter = np.mean(targets, axis=0)
    nodes = np.full((len(targets), 2), barycenter).astype(np.float32)
    error = error_function_wrapper(nodes, targets, dist_threshold, cost_function, cost_power)
    history = []
    for i in range(steps):
        if use_genetic_sampling:
            nodes, error = mutate_nodes_genetic_sampling(
                nodes, targets, error, stepsize=(1-i/steps)*mutation_stepsize,
                error_function=lambda n, t: error_function_wrapper(n, t, dist_threshold, cost_function, cost_power),
                sampling_size=sampling_size
            )
            history.append(nodes)
        else:
            nodes, error = mutate_nodes(
                nodes, targets, error, stepsize=(1-i/steps)*mutation_stepsize,
                error_function=lambda n, t: error_function_wrapper(n, t, dist_threshold, cost_function, cost_power)
            )
            history.append(nodes)
    return nodes, history



# At the top of your file
global_targets = None

def worker_init(targets):
    global global_targets
    global_targets = targets

def calc_optimal_graph_worker(dist_threshold, cost_function, cost_power, steps, mutation_stepsize, sampling_size, use_genetic_sampling):
    global global_targets
    # print(global_targets)
    return calc_optimal_graph(global_targets, dist_threshold, cost_function, cost_power, steps, mutation_stepsize, sampling_size, use_genetic_sampling)


# def multicalc_optimal_graph(targets, dist_threshold, cost_function, cost_power, ngraphs=10, steps=10000):

#     with mp.Pool(ngraphs) as pool:
#         args = [(targets, dist_threshold, cost_function, cost_power, steps) for _ in range(ngraphs)]
#         results = pool.starmap(calc_optimal_graph, args)
#     return results

def multicalc_optimal_graph(targets, dist_threshold, cost_function, cost_power, ngraphs=1000, steps=10000, mutation_stepsize=1.0, sampling_size=10, use_genetic_sampling=False):
    with mp.Pool(ngraphs, initializer=worker_init, initargs=(targets,)) as pool:
        args = [(dist_threshold, cost_function, cost_power, steps, mutation_stepsize, sampling_size, use_genetic_sampling) for _ in range(ngraphs)]
        results, histories = zip(*pool.starmap(calc_optimal_graph_worker, args))
    return results, histories



#                         _   _              _                  _ _   _                   
#                        | | (_)            | |                (_) | | |                  
#    __ _  ___ _ __   ___| |_ _  ___    __ _| | __ _  ___  _ __ _| |_| |__  _ __ ___  ___ 
#   / _` |/ _ \ '_ \ / _ \ __| |/ __|  / _` | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \/ __|
#  | (_| |  __/ | | |  __/ |_| | (__  | (_| | | (_| | (_) | |  | | |_| | | | | | | | \__ \
#   \__, |\___|_| |_|\___|\__|_|\___|  \__,_|_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|___/
#    __/ |                                      __/ |                                     
#   |___/                                      |___/                                      

                         

def mutate_nodes_geneticaly(nodes, stepsize, steps, dist_threshold):
    """
    retournes un nouveau set de noeuds ayant leurs positions modifiées aléatoirement.
    sur plusieurs pas de temps.
    """
    for i in range(steps):
        new_nodes = nodes + np.random.uniform(-stepsize, stepsize, nodes.shape).astype(np.float32)
        connected = is_graph_connected(new_nodes, dist_threshold)
        if connected:
            nodes = new_nodes
        else:
            continue
    return nodes

def sort_kill_reproduce(nodes, targets, dist_threshold, cost_function, cost_power, keep_best=10):
    """
    sort les noeuds par erreur, tués les plus mauvais et reproduit les meilleurs.
    """
    sorted_nodes = sorted(nodes, key=lambda x: cost_function(x, targets, cost_power))
    sorted_nodes = sorted_nodes[:keep_best]
    sorted_nodes = reproduce(sorted_nodes, len(nodes))
    return sorted_nodes


def reproduce(nodes, wanted_size):
    """
    Prend des éléments de nodes et les duplique pour obtenir wanted_size noeuds.
    """
    nodes = np.asarray(nodes)
    n_current = len(nodes)
    n_to_add = wanted_size - n_current
    if n_to_add <= 0:
        return nodes[:wanted_size]
    # Choisir aléatoirement des indices à dupliquer
    indices = np.random.choice(n_current, size=n_to_add, replace=True)
    new_nodes = nodes[indices]
    return np.concatenate([nodes, new_nodes], axis=0)


# print(reproduce(np.array([[0,0],[1,0],[0,1],[1,1]]), 12))

# quit()


MATPLOTLIB = True



if __name__ == "__main__" and MATPLOTLIB:
    # ne pas mettre de seed car le multiprocessing ne fonctionne pas avec les seeds

    NB_NODES = 4
    BOX_SIZE = 4
    NGRAPHS = 100
    STEPS = 1000
    MUTATION_STEPSIZE = 1
    SAMPLING_SIZE = 3
    USE_GENETIC_SAMPLING = True
    SCALE_NODES = 10
#
    # nodes = np.array([[0,0],[1,0],[0,1],[1,1]])
    targets = np.array([[0,0],[3,0],[0,3],[3,4]]).astype(np.float32)


    dist_threshold = 1.1
    # targets = np.random.uniform(0, BOX_SIZE, (NB_NODES, 2)).astype(np.float32)
    print("targets", targets)   
    # nodes = np.full((len(targets), 2), np.mean(targets, axis=0)).astype(np.float32)
    # print("nodes", nodes)
    # print("targets", targets)
    # print("distances", np.linalg.norm(nodes - targets, axis=1))
    # print("average distance", np.average(np.linalg.norm(nodes - targets, axis=1)))
    # print("cout_snt", cout_snt(nodes, targets))
    # print("cout_total", cout_total(nodes, targets))
    # print("cout_min", cout_min(nodes, targets))
    # print(nodes_to_matrix(nodes, dist_threshold))
    # print(get_neighbors(0, nodes_to_matrix(nodes, dist_threshold)))
    # print(connected(0, 4, nodes_to_matrix(nodes, dist_threshold)))

    # nodes = calc_optimal_graph(targets, dist_threshold, cout_snt, 2)
    # print("nodes", nodes)

    import time

    start_time = time.time()
    results, history = multicalc_optimal_graph(targets, dist_threshold, cout_snt, 2, 
    ngraphs=NGRAPHS, 
    steps=STEPS, 
    mutation_stepsize=MUTATION_STEPSIZE,
    sampling_size=SAMPLING_SIZE,
    use_genetic_sampling=USE_GENETIC_SAMPLING
    )
    end_time = time.time()
    print(f"Temps d'exécution: {end_time - start_time} secondes")
    # print("history", history)
    errors = [cout_snt(result, targets) for result in results]
    # print("errors", errors)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    # ax1.scatter(nodes[:, 0], nodes[:, 1], label='Initial nodes', color='black')
    ax1.scatter(targets[:, 0], targets[:, 1], label='Targets', color='red')

    errors = np.array(errors)
    norm = mcolors.Normalize(vmin=errors.min(), vmax=errors.max())
    cmap = plt.colormaps['viridis']
    colors = [cmap(norm(error)) for error in errors]

    for result, color in zip(results, colors):
        ax1.scatter(result[:, 0], result[:, 1], color=color, label=f'Error: {cout_snt(result, targets):.2f}', s=SCALE_NODES)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax1, label='Error')
    # ax1.legend()
    ax1.set_title('Positions des noeuds colorées par erreur')

    # Plot histogram of errors
    ax2.hist(errors, bins=100, color='gray', edgecolor='black')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogramme des erreurs')

    plt.tight_layout()
    plt.show()
        