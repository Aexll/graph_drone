import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt

class CriticalEdgeDetector:
    """
    Implémentation des algorithmes distribués pour la détection d'arêtes critiques
    selon le papier "A Distributed Method for Detecting Critical Edges and Increasing
    Edge Connectivity in Undirected Networks"
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialise le détecteur avec un graphe NetworkX
        
        Args:
            graph: Graphe non dirigé NetworkX
        """
        self.graph = graph
        self.n = len(graph.nodes())
        self.nodes = list(graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Initialisation des matrices d'état
        self.xi = np.zeros((self.n, self.n))  # Matrice de connectivité
        self.omega = np.full((self.n, self.n), np.inf)  # Matrice des distances
        
        # Initialisation des conditions initiales
        for i in range(self.n):
            self.xi[i, i] = 1.0
            self.omega[i, i] = 0.0
    
    def get_neighbors(self, node_idx: int) -> List[int]:
        """
        Retourne les indices des voisins d'un nœud
        
        Args:
            node_idx: Index du nœud
            
        Returns:
            Liste des indices des voisins
        """
        node = self.nodes[node_idx]
        neighbors = list(self.graph.neighbors(node))
        return [self.node_to_idx[neighbor] for neighbor in neighbors]
    
    def algorithm_1_xi_omega(self, max_iterations: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithme 1: Identification distribuée de la structure de voisinage
        
        Équations (1) et (2) du papier:
        - xi[i,j](k+1) = max(xi[l,j](k)) pour l dans N_i ∪ {i}
        - omega[i,j](k+1) dépend de xi[i,j](k+1)
        
        Args:
            max_iterations: Nombre maximum d'itérations (par défaut n)
            
        Returns:
            Tuple (xi_final, omega_final)
        """
        if max_iterations is None:
            max_iterations = self.n
        
        xi_history = [self.xi.copy()]
        omega_history = [self.omega.copy()]
        
        for k in range(max_iterations):
            xi_new = np.zeros((self.n, self.n))
            omega_new = self.omega.copy()
            
            for i in range(self.n):
                neighbors = self.get_neighbors(i)
                neighbors_with_self = neighbors + [i]
                
                for j in range(self.n):
                    # Mise à jour de xi selon l'équation (1)
                    xi_new[i, j] = max(self.xi[l, j] for l in neighbors_with_self)
                    
                    # Mise à jour de omega selon l'équation (2)
                    if xi_new[i, j] == self.xi[i, j]:
                        # Pas de changement dans xi
                        omega_new[i, j] = self.omega[i, j]
                    elif xi_new[i, j] > self.xi[i, j]:
                        # xi a augmenté, calculer la nouvelle distance
                        if neighbors:
                            min_dist = min(self.omega[l, j] + 1 for l in neighbors
                                         if self.omega[l, j] != np.inf)
                            omega_new[i, j] = min_dist if min_dist != np.inf else np.inf
                        else:
                            omega_new[i, j] = np.inf
            
            self.xi = xi_new
            self.omega = omega_new
            
            xi_history.append(self.xi.copy())
            omega_history.append(self.omega.copy())
        
        return self.xi, self.omega
    
    def is_connected(self) -> bool:
        """
        Vérifie si le réseau est connecté selon le théorème 1
        
        Returns:
            True si le réseau est connecté, False sinon
        """
        # Le réseau est connecté si xi[i,j] != 0 pour tous i,j
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.xi[i, j] == 0:
                    return False
        return True
    
    def compute_delta(self, i: int, l: int) -> np.ndarray:
        """
        Calcule la mesure Delta selon l'équation (6)
        
        Args:
            i: Index du premier nœud
            l: Index du second nœud (voisin de i)
            
        Returns:
            Vecteur delta de taille n
        """
        delta = np.zeros(self.n)
        
        for j in range(self.n):
            if self.omega[i, j] != np.inf and self.omega[l, j] != np.inf:
                delta[j] = self.omega[i, j] - self.omega[l, j]
            else:
                delta[j] = 0  # Cas où l'un des nœuds n'est pas atteignable
        
        return delta
    
    def is_critical_edge(self, i: int, l: int) -> bool:
        """
        Détermine si l'arête (i,l) est critique selon le théorème 2
        
        Args:
            i: Index du premier nœud
            l: Index du second nœud
            
        Returns:
            True si l'arête est critique, False sinon
        """
        # Vérifier si i et l sont voisins
        if l not in self.get_neighbors(i):
            return False
        
        delta_il = self.compute_delta(i, l)
        
        # Vérifier les conditions du théorème 2
        # Condition 1: Aucun nœud j n'est équidistant de i et l
        for j in range(self.n):
            if delta_il[j] == 0:
                return False  # Il existe un chemin alternatif
        
        # Condition 2: Vérifier l'existence de cycles
        neighbors_i = self.get_neighbors(i)
        neighbors_l = self.get_neighbors(l)
        
        for i_prime in neighbors_i:
            if i_prime == l:
                continue
            for l_prime in neighbors_l:
                if l_prime == i:
                    continue
                
                delta_ii_prime = self.compute_delta(i, i_prime)
                delta_ll_prime = self.compute_delta(l, l_prime)
                
                for j in range(self.n):
                    if (delta_il[j] != 0 and 
                        delta_ii_prime[j] == 1 and 
                        delta_ll_prime[j] == 1):
                        return False  # Il existe un cycle
        
        return True  # L'arête est critique
    
    def find_all_critical_edges(self) -> List[Tuple[int, int]]:
        """
        Trouve toutes les arêtes critiques du graphe
        
        Returns:
            Liste des arêtes critiques sous forme de tuples (i, j)
        """
        critical_edges = []
        
        # Parcourir toutes les arêtes du graphe
        for edge in self.graph.edges():
            i = self.node_to_idx[edge[0]]
            j = self.node_to_idx[edge[1]]
            
            if self.is_critical_edge(i, j):
                critical_edges.append((i, j))
        
        return critical_edges
    
    def get_neighbor_structure(self, node_idx: int) -> Dict[int, List[int]]:
        """
        Retourne la structure de voisinage d'un nœud
        
        Args:
            node_idx: Index du nœud
            
        Returns:
            Dictionnaire {distance: [liste_des_nœuds]}
        """
        structure = {}
        
        for j in range(self.n):
            if j != node_idx and self.omega[node_idx, j] != np.inf:
                dist = int(self.omega[node_idx, j])
                if dist not in structure:
                    structure[dist] = []
                structure[dist].append(j)
        
        return structure
    
    def print_results(self):
        """
        Affiche les résultats des algorithmes
        """
        print("=== RÉSULTATS DES ALGORITHMES ===")
        print(f"Nombre de nœuds: {self.n}")
        print(f"Réseau connecté: {self.is_connected()}")
        
        print("\nMatrice Xi (connectivité):")
        print(self.xi)
        
        print("\nMatrice Omega (distances):")
        omega_display = self.omega.copy()
        omega_display[omega_display == np.inf] = -1  # Remplacer inf par -1 pour l'affichage
        print(omega_display)
        
        print("\nArêtes critiques:")
        critical_edges = self.find_all_critical_edges()
        if critical_edges:
            for i, j in critical_edges:
                print(f"  ({self.nodes[i]}, {self.nodes[j]})")
        else:
            print("  Aucune arête critique trouvée")
        
        print("\nStructure de voisinage pour chaque nœud:")
        for i in range(self.n):
            structure = self.get_neighbor_structure(i)
            print(f"  Nœud {self.nodes[i]}: {structure}")


def create_example_graph_1() -> nx.Graph:
    """
    Crée le graphe G1 de l'exemple du papier (Figure 1a)
    """
    G = nx.Graph()
    
    # Ajouter les nœuds
    G.add_nodes_from(range(1, 9))
    
    # Ajouter les arêtes selon la Figure 1a
    edges = [
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
        (4, 5),  # Arête critique
        (5, 6), (5, 7),
        (6, 7),
        (7, 8)
    ]
    
    G.add_edges_from(edges)
    return G


def create_example_graph_2() -> nx.Graph:
    """
    Crée le graphe G2 de l'exemple du papier (Figure 1b)
    """
    G = create_example_graph_1()
    
    # Ajouter l'arête qui élimine l'arête critique
    G.add_edge(6, 8)
    
    return G


def visualize_graph(G: nx.Graph, critical_edges: List[Tuple] = None, title: str = "Graphe"):
    """
    Visualise le graphe avec les arêtes critiques en rouge
    """
    plt.figure(figsize=(10, 8))
    
    # Position des nœuds
    pos = nx.spring_layout(G, seed=42)
    
    # Dessiner les nœuds
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.7)
    
    # Dessiner les arêtes normales
    normal_edges = list(G.edges())
    if critical_edges:
        # Convertir les indices en noms de nœuds si nécessaire
        critical_edge_names = []
        for edge in critical_edges:
            if isinstance(edge[0], int) and edge[0] < len(list(G.nodes())):
                # Si ce sont des indices, les convertir
                nodes = list(G.nodes())
                critical_edge_names.append((nodes[edge[0]], nodes[edge[1]]))
            else:
                critical_edge_names.append(edge)
        
        normal_edges = [e for e in G.edges() if e not in critical_edge_names and (e[1], e[0]) not in critical_edge_names]
        
        # Dessiner les arêtes critiques en rouge
        nx.draw_networkx_edges(G, pos, edgelist=critical_edge_names, 
                              edge_color='red', width=3, alpha=0.8, style='dashed')
    
    # Dessiner les arêtes normales
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, 
                          edge_color='black', width=1, alpha=0.6)
    
    # Dessiner les labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale pour tester les algorithmes
    """
    print("=== TEST DES ALGORITHMES DE DÉTECTION D'ARÊTES CRITIQUES ===\n")
    
    # Test avec le graphe G1 (avec arêtes critiques)
    print("1. Test avec le graphe G1 (Figure 1a du papier)")
    G1 = create_example_graph_1()
    
    detector1 = CriticalEdgeDetector(G1)
    
    # Exécuter l'algorithme 1
    xi, omega = detector1.algorithm_1_xi_omega()
    
    # Afficher les résultats
    detector1.print_results()
    
    # Visualiser le graphe
    critical_edges_1 = detector1.find_all_critical_edges()
    visualize_graph(G1, critical_edges_1, "Graphe G1 avec arêtes critiques")
    
    print("\n" + "="*60 + "\n")
    
    # Test avec le graphe G2 (sans arêtes critiques)
    print("2. Test avec le graphe G2 (Figure 1b du papier)")
    G2 = create_example_graph_2()
    
    detector2 = CriticalEdgeDetector(G2)
    
    # Exécuter l'algorithme 1
    xi, omega = detector2.algorithm_1_xi_omega()
    
    # Afficher les résultats
    detector2.print_results()
    
    # Visualiser le graphe
    critical_edges_2 = detector2.find_all_critical_edges()
    visualize_graph(G2, critical_edges_2, "Graphe G2 sans arêtes critiques")
    
    print("\n" + "="*60 + "\n")
    
    # Test avec un graphe simple pour validation
    print("3. Test avec un graphe simple (chemin linéaire)")
    G3 = nx.path_graph(5)  # Graphe en chemin: 0-1-2-3-4
    
    detector3 = CriticalEdgeDetector(G3)
    xi, omega = detector3.algorithm_1_xi_omega()
    detector3.print_results()
    
    critical_edges_3 = detector3.find_all_critical_edges()
    visualize_graph(G3, critical_edges_3, "Graphe en chemin (toutes les arêtes sont critiques)")


if __name__ == "__main__":
    main()