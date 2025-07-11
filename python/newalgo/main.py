import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


np.random.seed(21) 

class MessageType(Enum):
    LOCK = "lock"
    UNLOCK = "unlock"

class Message:
    def __init__(self, sender, destination, message_type, edge=None):
        self.sender = sender
        self.destination = destination
        self.message_type = message_type
        self.edge = edge  # L'arête associée (tuple de deux nœuds)

    def __str__(self):
        return f"Message(sender={self.sender}, destination={self.destination}, type={self.message_type.value}, edge={self.edge})"

class Node:
    
    pending_messages = []

    def __init__(self, id, neighbors):
        self.id = id
        self.neighbors = neighbors  # Liste des IDs des voisins
        self.is_locked = False
        self.locked_edge = None  # L'arête qui sera supprimée
        self.unlock_status = {}  # Dictionnaire: {neighbor_id: bool}
        self.outbox = []  # Messages à envoyer au prochain pas de temps
        
        # Initialiser le dictionnaire de débloquage
        for neighbor in neighbors:
            self.unlock_status[neighbor] = False
    
    def add_to_outbox(self, message):
        """Ajoute un message à la boîte d'envoi"""
        self.outbox.append(message)
    
    def send_messages(self):
        """Envoie tous les messages de la boîte d'envoi"""
        for message in self.outbox:
            Node.pending_messages.append(message)
            print(f"Node {self.id} sent message: {message}")
        self.outbox.clear()
    
    def process_message(self, message):
        """Traite un message reçu selon l'algorithme de verrouillage"""
        print(f"Node {self.id} received: {message}")
        
        if message.message_type == MessageType.LOCK:
            if not self.is_locked:
                # Cas 1: Drone non verrouillé reçoit un message de verrouillage
                self.is_locked = True
                self.locked_edge = message.edge
                # Mettre toutes les valeurs de débloquage à False
                for neighbor in self.unlock_status:
                    self.unlock_status[neighbor] = False
                
                # Propager le message de verrouillage à tous les voisins
                for neighbor in self.neighbors:
                    lock_message = Message(self.id, neighbor, MessageType.LOCK, message.edge)
                    self.add_to_outbox(lock_message)
                
                print(f"Node {self.id} is now locked on edge {self.locked_edge}")
            
            else:
                # Cas 2: Drone verrouillé reçoit un message de verrouillage
                # Renvoyer un message de débloquage
                unlock_message = Message(self.id, message.sender, MessageType.UNLOCK, self.locked_edge)
                self.add_to_outbox(unlock_message)
                print(f"Node {self.id} sends unlock message to {message.sender}")
        
        elif message.message_type == MessageType.UNLOCK:
            if self.is_locked and message.edge == self.locked_edge:
                # Cas 3: Drone verrouillé reçoit un message de débloquage avec la même arête
                self.unlock_status[message.sender] = True
                print(f"Node {self.id} marked neighbor {message.sender} as unlocked")
                
                # Vérifier si tous les voisins ont débloqué
                if all(self.unlock_status.values()):
                    print(f"Node {self.id} can now safely remove edge {self.locked_edge}")
                    self.unlock()
    
    def unlock(self):
        """Déverrouille le nœud"""
        self.is_locked = False
        self.locked_edge = None
        for neighbor in self.unlock_status:
            self.unlock_status[neighbor] = False
        print(f"Node {self.id} is now unlocked")
    
    def initiate_lock(self, edge):
        """Initie un verrouillage sur une arête"""
        if not self.is_locked:
            self.is_locked = True
            self.locked_edge = edge
            for neighbor in self.unlock_status:
                self.unlock_status[neighbor] = False
            
            # Envoyer message de verrouillage à tous les voisins
            for neighbor in self.neighbors:
                lock_message = Message(self.id, neighbor, MessageType.LOCK, edge)
                self.add_to_outbox(lock_message)
            
            print(f"Node {self.id} initiated lock on edge {edge}")
    
    def __str__(self):
        return f"Node {self.id}: locked={self.is_locked}, edge={self.locked_edge}, unlock_status={self.unlock_status}"


def create_graph():
    G = nx.Graph()

    # Add nodes with attributes
    G.add_node(0, label='0', color='red')
    G.add_node(1, label='1', color='blue')
    G.add_node(2, label='2', color='green')
    G.add_node(3, label='3', color='yellow')
    G.add_node(4, label='4', color='purple')
    G.add_node(5, label='5', color='orange')
    G.add_node(6, label='6', color='pink')

    # Add random edges
    edges = [
        tuple(np.random.choice(G.nodes(), size=2, replace=False))
        for _ in range(10)
    ]

    G.add_edges_from(edges)
    
    return G

def create_nodes_from_graph(G):
    """Crée les objets Node à partir du graphe NetworkX"""
    nodes = {}
    
    for node_id in G.nodes():
        neighbors = list(G.neighbors(node_id))
        nodes[node_id] = Node(node_id, neighbors)
    
    return nodes

def simulate_timestep(nodes):
    """Simule un pas de temps de l'algorithme"""
    print("\n=== DÉBUT DU PAS DE TEMPS ===")
    
    # Phase 1: Traiter les messages en attente
    messages_to_process = Node.pending_messages.copy()
    Node.pending_messages.clear()
    
    for message in messages_to_process:
        if message.destination in nodes:
            nodes[message.destination].process_message(message)
    
    # Phase 2: Envoyer les nouveaux messages
    for node in nodes.values():
        node.send_messages()
    
    print("=== FIN DU PAS DE TEMPS ===\n")

def print_nodes_status(nodes):
    """Affiche l'état de tous les nœuds"""
    print("\n--- État des nœuds ---")
    for node in nodes.values():
        print(node)
    print("--- Fin état ---\n")

def draw_graph(G):
    pos = nx.spring_layout(G)
    colors = [G.nodes[n]['color'] for n in G.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700, font_size=16)
    plt.title("Graph Visualization")
    plt.show()

def test_multiple_scenarios():
    """Teste l'algorithme avec différents scénarios"""
    print("\n" + "="*60)
    print("TEST DE SCÉNARIOS MULTIPLES")
    print("="*60)
    
    # Scénario 1: Tentative de verrouillage simultané
    print("\n--- SCÉNARIO 1: Tentative de verrouillage simultané ---")
    G = create_graph()
    nodes = create_nodes_from_graph(G)
    
    if len(G.edges()) >= 2:
        edges = list(G.edges())
        edge1, edge2 = edges[0], edges[1]
        
        print(f"Node {edge1[0]} tente de verrouiller l'arête {edge1}")
        print(f"Node {edge2[0]} tente de verrouiller l'arête {edge2}")
        
        # Les deux nœuds initient un verrouillage simultanément
        nodes[edge1[0]].initiate_lock(edge1)
        nodes[edge2[0]].initiate_lock(edge2)
        
        # Simulation
        timestep = 0
        while timestep < 15:
            timestep += 1
            print(f"\n--- Pas de temps {timestep} ---")
            simulate_timestep(nodes)
            
            if not Node.pending_messages and all(not node.outbox for node in nodes.values()):
                break
        
        print("\n--- Résultat du scénario 1 ---")
        for node in nodes.values():
            if node.is_locked:
                if all(node.unlock_status.values()):
                    print(f"✅ Node {node.id} peut supprimer l'arête {node.locked_edge}")
                else:
                    print(f"❌ Node {node.id} attend encore des débloquages pour {node.locked_edge}")

def visualize_lock_propagation(G, nodes, locked_edge):
    """Visualise la propagation du verrouillage dans le graphe"""
    pos = nx.spring_layout(G, seed=42)  # Position fixe pour cohérence
    
    # Couleurs des nœuds selon leur état
    node_colors = []
    for node_id in G.nodes():
        if node_id in nodes:
            node = nodes[node_id]
            if node.is_locked:
                if node.locked_edge == locked_edge:
                    node_colors.append('red')  # Verrouillé sur la bonne arête
                else:
                    node_colors.append('orange')  # Verrouillé sur une autre arête
            else:
                node_colors.append('lightblue')  # Non verrouillé
        else:
            node_colors.append('gray')
    
    # Couleurs des arêtes
    edge_colors = []
    for edge in G.edges():
        if edge == locked_edge or (edge[1], edge[0]) == locked_edge:
            edge_colors.append('red')  # Arête cible
        else:
            edge_colors.append('black')  # Autres arêtes
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            edge_color=edge_colors, node_size=800, font_size=14, font_weight='bold')
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Nœud verrouillé (arête cible)'),
        Patch(facecolor='orange', label='Nœud verrouillé (autre arête)'),
        Patch(facecolor='lightblue', label='Nœud libre'),
        Patch(facecolor='white', edgecolor='red', label='Arête cible')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(f"Propagation du verrouillage pour l'arête {locked_edge}")
    plt.show()

def main():
    # Créer le graphe et les nœuds
    G = create_graph()
    nodes = create_nodes_from_graph(G)
    
    print("Graphe créé avec les arêtes suivantes:")
    for edge in G.edges():
        print(f"  {edge}")
    
    print_nodes_status(nodes)
    
    # Visualiser le graphe initial
    draw_graph(G)
    
    # Test de l'algorithme: faire qu'un nœud initie un verrouillage
    if len(G.edges()) > 0:
        test_edge = list(G.edges())[0]  # Prendre la première arête
        initiator_node = test_edge[0]
        
        print(f"\nNode {initiator_node} initie un verrouillage sur l'arête {test_edge}")
        nodes[initiator_node].initiate_lock(test_edge)
        
        # Simuler plusieurs pas de temps
        timestep = 0
        max_timesteps = 20  # Protection contre les boucles infinies
        
        while timestep < max_timesteps:
            timestep += 1
            print(f"\n{'='*20} PAS DE TEMPS {timestep} {'='*20}")
            simulate_timestep(nodes)
            
            # Visualiser l'état du verrouillage après certains pas
            if timestep in [2, 4, 6]:
                visualize_lock_propagation(G, nodes, test_edge)
            
            print_nodes_status(nodes)
            
            # Si plus de messages en attente, arrêter
            if not Node.pending_messages and all(not node.outbox for node in nodes.values()):
                print("Plus de messages à traiter. Simulation terminée.")
                break
    
    print("\n=== ÉTAT FINAL ===")
    print_nodes_status(nodes)
    
    # Vérifier quel nœud peut supprimer l'arête en toute sécurité
    print("\n=== ANALYSE DE SÉCURITÉ ===")
    for node in nodes.values():
        if node.is_locked and all(node.unlock_status.values()):
            print(f"✅ Node {node.id} peut supprimer l'arête {node.locked_edge} en toute sécurité")
        elif node.is_locked:
            missing_unlocks = [neighbor for neighbor, unlocked in node.unlock_status.items() if not unlocked]
            print(f"❌ Node {node.id} ne peut pas encore supprimer l'arête {node.locked_edge}")
            print(f"   En attente de débloquage de: {missing_unlocks}")
    
    # Tester des scénarios additionnels
    test_multiple_scenarios()
    
    # Tester différents scénarios
    test_multiple_scenarios()

if __name__ == "__main__":
    main()

