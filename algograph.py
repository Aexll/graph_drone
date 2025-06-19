import numpy as np

GRAPH: np.ndarray = np.array([
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
])
GRAPH_COUNT: int = len(GRAPH)

def update_graph(g: np.ndarray):
    global GRAPH
    GRAPH = g
    GRAPH_COUNT = len(g)

def N(i):
    """
    returns the neighbors of drone i as a set of ids
    """
    return set(np.where(GRAPH[i] == 1)[0])

xi_memory = {}

def ξ(i,j,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    if n <= 0: 
        return 1 if i == j else 0
    
    if (i, j, n) in xi_memory:
        return xi_memory[(i, j, n)]
    v = max(ξ(l, j, n-1) for l in N(i).union({i}))
    xi_memory[(i, j, n)] = v
    return v

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

def ω(i,j,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    return 0

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

print(xi_array(1, GRAPH_COUNT))





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
