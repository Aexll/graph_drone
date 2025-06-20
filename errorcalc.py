import numpy as np

def error_linear(node_pos, target_pos):
    """
    distance between node and target
    """
    return np.linalg.norm(node_pos - target_pos)

def error_square(node_pos, target_pos):
    """
    sum of all squared distances between nodes and targets
    """
    return np.sum((node_pos - target_pos)**2)

def er_sq(nodes, targets):
    """
    sum of all squared distances between nodes and targets sqrted
    """
    return np.sum((nodes - targets)**2)**0.5

def er_lin(nodes, targets):
    """
    sum of all distances between nodes and targets
    """
    return np.sum(np.linalg.norm(nodes - targets, 1))

def calculate_error_graph(nodes, targets, error_function):
    error = 0
    for i in range(len(nodes)):
        error += error_function(nodes[i], targets[i])
    return error

def calculate_graph_connectivity(nodes,dist_threshold):
    connected_memory.clear()
    graph = nodes_to_matrix(nodes, dist_threshold)
    return not (0 in connected(0, len(nodes), graph))



    
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


if __name__ == "__main__":
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]])
    dist_threshold = 1.1
    print(nodes_to_matrix(nodes, dist_threshold))
    print(get_neighbors(0, nodes_to_matrix(nodes, dist_threshold)))
    print(connected(0, 4, nodes_to_matrix(nodes, dist_threshold)))
    # print(calculate_graph_connectivity(nodes, dist_threshold))