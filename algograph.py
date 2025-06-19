import numpy as np

INF = 100000000

GRAPH: np.ndarray = np.array([
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0]
])
GRAPH_COUNT: int = len(GRAPH)


def update_graph(g: np.ndarray):
    global GRAPH, GRAPH_COUNT, xi_memory, omega_memory
    GRAPH = g
    GRAPH_COUNT = len(g)
    xi_memory.clear()
    omega_memory.clear()

def N(i):
    """
    returns the neighbors of drone i as a set of ids
    """
    return set(np.where(GRAPH[i] == 1)[0])

xi_memory = {}

def ξ(i,j,n):
    """
    returns if a path exists between drone i and drone j, with n hops or less
    """
    return xi_array(i, n)[j]

def xi_array(i,n):
    """
    returns the xi array for drone i
    """
    if n <= 0: 
        v = np.zeros(GRAPH_COUNT)
        v[i] = 1
        return v
    if (i, n) in xi_memory:
        return xi_memory[(i, n)]

    arrays = [xi_array(l, n-1) for l in N(i).union({i})]
    v = np.maximum.reduce(arrays)
    xi_memory[(i, n)] = v
    return v

omega_memory = {}

def ω(i,j,n):
    """
    returns the distance between node i and j, with n hops or less (if there is no path, returns INF)
    """
    return ω_array(i, n)[j]

def ω_array(i,n):
    """
    returns the ω array for drone i
    """
    if (i, n) in omega_memory:
        return omega_memory[(i, n)]

    if n <= 0:
        v = np.ones(GRAPH_COUNT) * INF
        v[i] = 0
        omega_memory[(i, n)] = v
        return v
    # Optimized ω_array using matrices

    # Compute xi arrays for n and n-1 
    xi_n = xi_array(i, n)
    xi_n1 = xi_array(i, n-1)

    # Initialize result array
    v = np.ones(GRAPH_COUNT) * INF

    # For nodes where reachability didn't change, copy previous omega
    unchanged = (xi_n == xi_n1)
    if n > 0:
        v[unchanged] = ω_array(i, n-1)[unchanged]

    # For nodes where reachability changed, compute min over neighbors
    changed = ~unchanged
    if np.any(changed): # if there are any changed nodes
        neighbors = list(N(i))
        if neighbors: # if there are any neighbors
            # Stack omega arrays for all neighbors at n-1
            neighbor_omegas = np.stack([ω_array(l, n-1) for l in neighbors], axis=0)
            # Take min over neighbors and add 1
            v[changed] = np.min(neighbor_omegas[:, changed], axis=0) + 1

    omega_memory[(i, n)] = v
    return v


    

def Δ(i,j,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    return 0

def Δ_array(i,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    return 0

def is_critical_edge(i,l):
    """
    returns True if the edge i,l is critical, False otherwise
    """
    return False

print("xi_array")
for i in range(GRAPH_COUNT):
    print(xi_array(i, GRAPH_COUNT))

print("ω_array")
for i in range(GRAPH_COUNT):
    print(ω_array(i, GRAPH_COUNT))




## DRAWING ##


import matplotlib.pyplot as plt
import networkx as nx

def draw_graph():
    """
    Draws the current graph using matplotlib and networkx.
    Assumes GRAPH is a numpy adjacency matrix.
    """
    if 'GRAPH' not in globals() or GRAPH is None:
        print("GRAPH is not defined.")
        return

    G = nx.Graph()
    n = GRAPH.shape[0]
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            if GRAPH[i, j] == 1 or GRAPH[j, i] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
    plt.title("Graph Visualization")
    plt.show()

# Example usage:
draw_graph()
