import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import copy
import time
from collections import defaultdict

class DistributedNode:
    """
    Représente un nœud dans le réseau distribué.
    Chaque nœud maintient uniquement ses propres données locales.
    """
    
    def __init__(self, node_id: int, initial_neighbors: List[int] = None):
        """
        Initialise un nœud distribué
        
        Args:
            node_id: Identifiant unique du nœud
            initial_neighbors: Liste des voisins initiaux (optionnel)
        """
        self.node_id = node_id
        self.neighbors = set(initial_neighbors) if initial_neighbors else set()
        
        # Variables locales pour l'algorithme xi-omega
        self.xi = defaultdict(float)  # xi[j] = valeur de connectivité vers j
        self.omega = defaultdict(lambda: float('inf'))  # omega[j] = distance vers j
        
        # Initialisation : le nœud se connaît lui-même
        self.xi[node_id] = 1.0
        self.omega[node_id] = 0.0
        
        # Messages à envoyer et reçus
        self.outgoing_messages = {}
        self.incoming_messages = {}
        
        # Historique pour détecter la convergence
        self.previous_xi = {}
        self.previous_omega = {}
        
        # Métadonnées
        self.iteration = 0
        self.converged = False
        self.known_nodes = {node_id}  # Ensemble des nœuds découverts
    
    def add_neighbor(self, neighbor_id: int):
        """Ajoute un nouveau voisin (pour les réseaux dynamiques)"""
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: int):
        """Supprime un voisin (pour les réseaux dynamiques)"""
        self.neighbors.discard(neighbor_id)
    
    def prepare_messages(self) -> Dict[int, Dict]:
        """
        Prépare les messages à envoyer aux voisins
        
        Returns:
            Dictionnaire {neighbor_id: message}
        """
        messages = {}
        
        # Créer un message contenant l'état xi et omega du nœud
        message = {
            'sender': self.node_id,
            'iteration': self.iteration,
            'xi': dict(self.xi),
            'omega': dict(self.omega),
            'known_nodes': self.known_nodes.copy()
        }
        
        # Envoyer le même message à tous les voisins
        for neighbor in self.neighbors:
            messages[neighbor] = copy.deepcopy(message)
        
        return messages
    
    def receive_message(self, sender_id: int, message: Dict):
        """
        Reçoit un message d'un voisin
        
        Args:
            sender_id: ID du nœud expéditeur
            message: Contenu du message
        """
        if sender_id not in self.incoming_messages:
            self.incoming_messages[sender_id] = []
        
        self.incoming_messages[sender_id].append(message)
    
    def update_xi_omega(self):
        """
        Met à jour les valeurs xi et omega selon l'algorithme distribué
        Implémentation correcte des équations du papier
        """
        # Sauvegarder l'état précédent
        self.previous_xi = dict(self.xi)
        self.previous_omega = dict(self.omega)
        
        # Découvrir de nouveaux nœuds à partir des messages
        all_known_nodes = set(self.known_nodes)
        
        for sender_id, messages in self.incoming_messages.items():
            if messages:  # Prendre le message le plus récent
                latest_message = messages[-1]
                all_known_nodes.update(latest_message['known_nodes'])
        
        self.known_nodes = all_known_nodes
        
        # Étape 1: Calculer xi pour tous les nœuds connus
        new_xi = defaultdict(float)
        new_omega = defaultdict(lambda: float('inf'))
        
        # Initialiser avec les valeurs actuelles
        for node in self.known_nodes:
            new_xi[node] = self.xi[node]
            new_omega[node] = self.omega[node]
        
        # Assurer que self reste à 1.0 et 0.0
        new_xi[self.node_id] = 1.0
        new_omega[self.node_id] = 0.0
        
        # Pour chaque nœud target j
        for target_node in self.known_nodes:
            if target_node == self.node_id:
                continue
                
            # Équation (1): xi_i,j(k+1) = max{xi_l,j(k) : l ∈ N_i ∪ {i}}
            candidates = [self.xi[target_node]]  # xi_i,j(k)
            
            # Ajouter les valeurs des voisins
            for neighbor_id in self.neighbors:
                if neighbor_id in self.incoming_messages:
                    messages = self.incoming_messages[neighbor_id]
                    if messages:
                        latest_message = messages[-1]
                        neighbor_xi = latest_message['xi']
                        if target_node in neighbor_xi:
                            candidates.append(neighbor_xi[target_node])
            
            # Si le target est un voisin direct, ajouter sa connectivité
            if target_node in self.neighbors:
                candidates.append(1.0)
            
            new_xi[target_node] = max(candidates)
        
        # Étape 2: Calculer omega pour tous les nœuds connus
        for target_node in self.known_nodes:
            if target_node == self.node_id:
                continue
                
            # Si xi a changé, recalculer omega
            if new_xi[target_node] != self.xi[target_node]:
                # Équation (2): Si xi augmente, omega = min{omega_l,j(k) + 1 : l ∈ N_i, xi_l,j(k) = xi_i,j(k+1)}
                min_distances = []
                
                # Si le target est un voisin direct et xi=1, alors omega=1
                if target_node in self.neighbors and new_xi[target_node] == 1.0:
                    min_distances.append(1.0)
                
                # Chercher parmi les voisins
                for neighbor_id in self.neighbors:
                    if neighbor_id in self.incoming_messages:
                        messages = self.incoming_messages[neighbor_id]
                        if messages:
                            latest_message = messages[-1]
                            neighbor_xi = latest_message['xi']
                            neighbor_omega = latest_message['omega']
                            
                            # Si le voisin a la même valeur xi que celle qu'on vient de calculer
                            if (target_node in neighbor_xi and 
                                abs(neighbor_xi[target_node] - new_xi[target_node]) < 1e-10):
                                if (target_node in neighbor_omega and 
                                    neighbor_omega[target_node] != float('inf')):
                                    min_distances.append(neighbor_omega[target_node] + 1)
                
                if min_distances:
                    new_omega[target_node] = min(min_distances)
                else:
                    new_omega[target_node] = float('inf')
            else:
                # xi n'a pas changé, garder omega
                new_omega[target_node] = self.omega[target_node]
        
        # Mettre à jour les valeurs
        self.xi = new_xi
        self.omega = new_omega
        
        # Nettoyer les messages traités
        self.incoming_messages.clear()
        self.iteration += 1
    
    def has_converged(self, tolerance: float = 1e-10) -> bool:
        """
        Vérifie si le nœud a convergé (pas de changement dans xi et omega)
        
        Args:
            tolerance: Tolérance pour la convergence
            
        Returns:
            True si convergé, False sinon
        """
        if not self.previous_xi or not self.previous_omega:
            return False
        
        # Vérifier si tous les nœuds connus ont des valeurs stables
        for node in self.known_nodes:
            if node == self.node_id:
                continue
            
            # Vérifier xi
            prev_xi = self.previous_xi.get(node, 0)
            curr_xi = self.xi.get(node, 0)
            if abs(curr_xi - prev_xi) > tolerance:
                return False
            
            # Vérifier omega
            prev_omega = self.previous_omega.get(node, float('inf'))
            curr_omega = self.omega.get(node, float('inf'))
            if abs(curr_omega - prev_omega) > tolerance:
                return False
        
        return True
    
    def get_neighbor_structure(self) -> Dict[int, List[int]]:
        """
        Retourne la structure de voisinage locale
        
        Returns:
            Dictionnaire {distance: [liste_des_nœuds]}
        """
        structure = defaultdict(list)
        
        for node in self.known_nodes:
            if node != self.node_id:
                distance = self.omega.get(node, float('inf'))
                if distance != float('inf'):
                    structure[int(distance)].append(node)
        
        return dict(structure)
    
    def is_critical_edge_local(self, neighbor_id: int) -> bool:
        """
        Détermine si l'arête vers un voisin est critique (version locale)
        
        Args:
            neighbor_id: ID du voisin
            
        Returns:
            True si l'arête est critique, False sinon
        """
        if neighbor_id not in self.neighbors:
            return False
        
        # Version simplifiée : une arête est critique si sa suppression
        # augmenterait la distance vers certains nœuds
        
        # Cette implémentation nécessiterait des informations sur l'état
        # après suppression de l'arête, ce qui n'est pas trivial en distribué
        # Pour l'instant, retourner False
        return False
    
    def print_local_state(self):
        """Affiche l'état local du nœud"""
        print(f"\n--- État du nœud {self.node_id} ---")
        print(f"Voisins: {sorted(self.neighbors)}")
        print(f"Nœuds connus: {sorted(self.known_nodes)}")
        print(f"Itération: {self.iteration}")
        print(f"Xi: {dict(self.xi)}")
        
        omega_display = {}
        for node, dist in self.omega.items():
            omega_display[node] = dist if dist != float('inf') else 'inf'
        print(f"Omega: {omega_display}")
        
        print(f"Structure de voisinage: {self.get_neighbor_structure()}")
        print(f"Convergé: {self.has_converged()}")


class DistributedNetwork:
    """
    Simulateur de réseau distribué pour tester les algorithmes
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialise le réseau distribué
        
        Args:
            graph: Graphe NetworkX à simuler
        """
        self.graph = graph
        self.nodes = {}
        
        # Créer les nœuds distribués
        for node_id in graph.nodes():
            neighbors = list(graph.neighbors(node_id))
            self.nodes[node_id] = DistributedNode(node_id, neighbors)
    
    def simulate_step(self):
        """
        Simule une étape de l'algorithme distribué
        """
        # Phase 1: Tous les nœuds préparent leurs messages
        all_messages = {}
        for node_id, node in self.nodes.items():
            all_messages[node_id] = node.prepare_messages()
        
        # Phase 2: Distribuer les messages
        for sender_id, messages in all_messages.items():
            for receiver_id, message in messages.items():
                if receiver_id in self.nodes:
                    self.nodes[receiver_id].receive_message(sender_id, message)
        
        # Phase 3: Tous les nœuds mettent à jour leur état
        for node in self.nodes.values():
            node.update_xi_omega()
    
    def simulate_until_convergence(self, max_iterations: int = 100) -> int:
        """
        Simule jusqu'à convergence
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Nombre d'itérations effectuées
        """
        for iteration in range(max_iterations):
            self.simulate_step()
            
            # Vérifier la convergence globale
            if all(node.has_converged() for node in self.nodes.values()):
                print(f"Convergence atteinte après {iteration + 1} itérations")
                return iteration + 1
        
        print(f"Convergence non atteinte après {max_iterations} itérations")
        return max_iterations
    
    def print_network_state(self):
        """Affiche l'état de tout le réseau"""
        print("=== ÉTAT DU RÉSEAU DISTRIBUÉ ===")
        for node_id in sorted(self.nodes.keys()):
            self.nodes[node_id].print_local_state()
    
    def add_edge(self, u: int, v: int):
        """Ajoute une arête au réseau (pour les réseaux dynamiques)"""
        if u in self.nodes and v in self.nodes:
            self.graph.add_edge(u, v)
            self.nodes[u].add_neighbor(v)
            self.nodes[v].add_neighbor(u)
    
    def remove_edge(self, u: int, v: int):
        """Supprime une arête du réseau (pour les réseaux dynamiques)"""
        if u in self.nodes and v in self.nodes:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
            self.nodes[u].remove_neighbor(v)
            self.nodes[v].remove_neighbor(u)


def test_distributed_algorithm():
    """
    Teste l'algorithme distribué xi-omega avec un debug détaillé
    """
    print("=== TEST DE L'ALGORITHME DISTRIBUÉ XI-OMEGA ===\n")
    
    # Créer un graphe simple pour tester
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    print("Graphe de test: chemin 1-2-3-4-5")
    print(f"Nœuds: {sorted(G.nodes())}")
    print(f"Arêtes: {sorted(G.edges())}")
    
    # Créer le réseau distribué
    network = DistributedNetwork(G)
    
    # État initial
    print("\n--- ÉTAT INITIAL ---")
    network.print_network_state()
    
    # Simuler l'algorithme étape par étape avec debug
    print("\n--- SIMULATION DÉTAILLÉE ---")
    for i in range(8):  # Assez d'itérations pour la convergence
        print(f"\n>>> ITÉRATION {i+1} <<<")
        
        # Phase 1: Afficher les messages envoyés
        print("Messages envoyés:")
        all_messages = {}
        for node_id, node in network.nodes.items():
            messages = node.prepare_messages()
            all_messages[node_id] = messages
            if messages:
                print(f"  Nœud {node_id} -> {list(messages.keys())}")
        
        # Phase 2: Distribuer les messages
        for sender_id, messages in all_messages.items():
            for receiver_id, message in messages.items():
                if receiver_id in network.nodes:
                    network.nodes[receiver_id].receive_message(sender_id, message)
        
        # Phase 3: Mettre à jour
        for node in network.nodes.values():
            node.update_xi_omega()
        
        # Afficher l'état après cette itération
        print(f"État après itération {i+1}:")
        for node_id in sorted(network.nodes.keys()):
            node = network.nodes[node_id]
            print(f"  Nœud {node_id}: connaît {sorted(node.known_nodes)}, xi={dict(node.xi)}")
        
        # Vérifier convergence
        converged = all(node.has_converged() for node in network.nodes.values())
        if converged:
            print(f"\n*** CONVERGENCE ATTEINTE après {i+1} itérations ***")
            break
    
    # État final
    print(f"\n--- ÉTAT FINAL ---")
    network.print_network_state()
    
    # Test avec un graphe en étoile
    print("\n" + "="*60)
    print("Test avec un graphe en étoile (nœud central: 0)")
    
    G2 = nx.star_graph(4)  # Étoile avec nœud central 0 et 4 branches
    network2 = DistributedNetwork(G2)
    
    print(f"\nGraphe en étoile: {sorted(G2.edges())}")
    print("\nSimulation rapide:")
    iterations2 = network2.simulate_until_convergence(max_iterations=10)
    
    print(f"\n--- ÉTAT FINAL DU GRAPHE EN ÉTOILE ---")
    network2.print_network_state()


if __name__ == "__main__":
    test_distributed_algorithm()