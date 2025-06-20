import numpy as np

def er_sq(nodes, targets):
    """
    sum of all squared distances between nodes and targets sqrted
    """
    return np.sum(np.linalg.norm(nodes - targets, axis=1)**2)**0.5

def er_lin(nodes, targets):
    """
    return euclidean distance between nodes and targets
    """
    return np.average(np.linalg.norm(nodes - targets, axis=1))

def er_max(nodes, targets):
    """
    max of all distances between nodes and targets
    """
    return np.max(np.linalg.norm(nodes - targets, axis=1))


def calculate_error_graph(nodes, targets, error_function):
    return error_function(nodes, targets)

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
    targets = np.array([[0,0],[3,0],[0,3],[3,4]])
    dist_threshold = 1.1
    print("nodes", nodes)
    print("targets", targets)
    print("distances", np.linalg.norm(nodes - targets, axis=1))
    print("average distance", np.average(np.linalg.norm(nodes - targets, axis=1)))
    print("er_sq", er_sq(nodes, targets))
    print("er_lin", er_lin(nodes, targets))
    print("er_max", er_max(nodes, targets))
    print(nodes_to_matrix(nodes, dist_threshold))
    print(get_neighbors(0, nodes_to_matrix(nodes, dist_threshold)))
    print(connected(0, 4, nodes_to_matrix(nodes, dist_threshold)))
    # print(calculate_graph_connectivity(nodes, dist_threshold))