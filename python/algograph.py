from math import inf
import numpy as np



def update_graph(g: np.ndarray):
    global xi_memory, omega_memory
    xi_memory.clear()
    omega_memory.clear()


def N(g: np.ndarray, i:int) -> set:
    """
    returns the neighbors of drone i as a set of ids
    """
    return set(np.where(g[i] == 1)[0])

xi_memory = {}

def ξ(g: np.ndarray, i:int, j:int, n:int) -> bool:
    """
    returns if a path exists between drone i and drone j, with n hops or less
    """
    return xi_array(g, i, n)[j]

def xi_array(g: np.ndarray, i:int, n:int) -> np.ndarray:
    """
    returns the xi array for drone i
    """
    if n <= 0: 
        v = np.zeros(len(g))
        v[i] = 1
        return v
    if (i, n) in xi_memory:
        return xi_memory[(i, n)]

    arrays = [xi_array(g, l, n-1) for l in N(g, i).union({i})]
    v = np.maximum.reduce(arrays)
    xi_memory[(i, n)] = v
    return v

omega_memory = {}

def ω(g: np.ndarray, i:int, j:int, n:int) -> float:
    """
    returns the distance between node i and j, with n hops or less (if there is no path, returns INF)
    """
    return ω_array(g, i, n)[j]

def ω_array(g: np.ndarray, i:int, n:int) -> np.ndarray:
    """
    returns the ω array for drone i
    """
    if (i, n) in omega_memory:
        return omega_memory[(i, n)]

    if n <= 0:
        v = np.ones(len(g)) * inf
        v[i] = 0
        omega_memory[(i, n)] = v
        return v
    # Optimized ω_array using matrices

    # Compute xi arrays for n and n-1 
    xi_n = xi_array(g, i, n)
    xi_n1 = xi_array(g, i, n-1)

    # Initialize result array
    v = np.ones(len(g)) * inf

    # For nodes where reachability didn't change, copy previous omega
    unchanged = (xi_n == xi_n1)
    if n > 0:
        v[unchanged] = ω_array(g, i, n-1)[unchanged]

    # For nodes where reachability changed, compute min over neighbors
    changed = ~unchanged
    if np.any(changed): # if there are any changed nodes
        neighbors = list(N(g, i))
        if neighbors: # if there are any neighbors
            # Stack omega arrays for all neighbors at n-1
            neighbor_omegas = np.stack([ω_array(g, l, n-1) for l in neighbors], axis=0)
            # Take min over neighbors and add 1
            v[changed] = np.min(neighbor_omegas[:, changed], axis=0) + 1

    omega_memory[(i, n)] = v
    return v


    

def Δ(g: np.ndarray, i:int, j:int, l:int, n:int) -> float:
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    return Δ_array(g, i, l, n)[j]

def Δ_array(g: np.ndarray, i:int, l:int, n:int) -> np.ndarray:
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    return ω_array(g, i, n) - ω_array(g, l, n)

def is_critical_edge(g: np.ndarray, i:int, l:int) -> bool:
    """
    returns True if the edge i,l is critical, False otherwise
    """

    # we substract 1-xi_array(i,GRAPH_COUNT+1) to avoid problemes when no conexions are possible
    if 0 in Δ_array(g, i, l, len(g)+1) - (1-xi_array(g, i, len(g)+1)) :
        return False

    for ii in N(g, i):
        for ll in N(g, l):
            if ii == l or ll == i:
                continue
            if 2 in Δ_array(g, i, ii, len(g)+1) + Δ_array(g, l, ll, len(g)+1):
                return False
    return True






## DRAWING ##
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx

    def draw_graph(g: np.ndarray):
        """
        Draws the current graph using matplotlib and networkx.
        Assumes GRAPH is a numpy adjacency matrix.
        """
        G = nx.Graph()
        n = g.shape[0]
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i+1, n):
                if g[i, j] == 1 or g[j, i] == 1:
                    G.add_edge(i, j)
                    if is_critical_edge(g, i, j):
                        print(f"Edge {i} {j} is critical")
                    else:
                        print(f"Edge {i} {j} is not critical")

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
        plt.title("Graph Visualization")
        plt.show()

    draw_graph(np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]))
