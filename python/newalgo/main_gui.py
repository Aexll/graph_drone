import dearpygui.dearpygui as dpg
import networkx as nx
import numpy as np
import math
import threading
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional


# Seed pour la reproducibilité
np.random.seed(21)

class MessageType(Enum):
    LOCK = "lock"
    UNLOCK = "unlock"

class Message:
    def __init__(self, sender, destination, message_type, edge=None):
        self.sender = sender
        self.destination = destination
        self.message_type = message_type
        self.edge = edge

    def __str__(self):
        return f"Message(sender={self.sender}, destination={self.destination}, type={self.message_type.value}, edge={self.edge})"

class Node:
    pending_messages = []

    def __init__(self, id, neighbors, position=(0, 0)):
        self.id = id
        self.neighbors = neighbors
        self.position = position  # Position pour l'affichage
        self.is_locked = False
        self.locked_edge = None
        self.unlock_status = {}
        self.outbox = []
        
        for neighbor in neighbors:
            self.unlock_status[neighbor] = False
    
    def add_to_outbox(self, message):
        self.outbox.append(message)
    
    def send_messages(self):
        for message in self.outbox:
            Node.pending_messages.append(message)
        self.outbox.clear()
    
    def process_message(self, message):
        if message.message_type == MessageType.LOCK:
            if not self.is_locked:
                self.is_locked = True
                self.locked_edge = message.edge
                for neighbor in self.unlock_status:
                    self.unlock_status[neighbor] = False
                
                for neighbor in self.neighbors:
                    lock_message = Message(self.id, neighbor, MessageType.LOCK, message.edge)
                    self.add_to_outbox(lock_message)
            else:
                unlock_message = Message(self.id, message.sender, MessageType.UNLOCK, self.locked_edge)
                self.add_to_outbox(unlock_message)
        
        elif message.message_type == MessageType.UNLOCK:
            if self.is_locked and message.edge == self.locked_edge:
                self.unlock_status[message.sender] = True
                
                if all(self.unlock_status.values()):
                    self.unlock()
    
    def unlock(self):
        self.is_locked = False
        self.locked_edge = None
        for neighbor in self.unlock_status:
            self.unlock_status[neighbor] = False
    
    def initiate_lock(self, edge):
        if not self.is_locked:
            self.is_locked = True
            self.locked_edge = edge
            for neighbor in self.unlock_status:
                self.unlock_status[neighbor] = False
            
            for neighbor in self.neighbors:
                lock_message = Message(self.id, neighbor, MessageType.LOCK, edge)
                self.add_to_outbox(lock_message)
    
    def get_color(self, target_edge=None):
        """Retourne la couleur du nœud selon son état"""
        if not self.is_locked:
            return [100, 150, 255, 255]  # Bleu clair (libre)
        elif target_edge and self.locked_edge == target_edge:
            return [255, 100, 100, 255]  # Rouge (verrouillé sur l'arête cible)
        else:
            return [255, 165, 0, 255]    # Orange (verrouillé sur une autre arête)

class GraphSimulation:
    def __init__(self):
        self.G = None
        self.nodes = {}
        self.positions = {}
        self.edges_list = []
        self.is_running = False
        self.timestep = 0
        self.target_edge = None
        self.simulation_speed = 1.0  # Secondes entre les pas de temps
        self.log_messages = []
        
        # Paramètres d'affichage
        self.canvas_width = 600
        self.canvas_height = 400
        self.node_radius = 20
        self.scale = 200
        
    def create_graph(self):
        """Crée un graphe aléatoire"""
        self.G = nx.Graph()
        
        # Ajouter des nœuds
        for i in range(7):
            self.G.add_node(i)
        
        # Ajouter des arêtes aléatoires
        edges = [
            tuple(np.random.choice(list(self.G.nodes()), size=2, replace=False))
            for _ in range(10)
        ]
        self.G.add_edges_from(edges)
        
        # Calculer les positions avec NetworkX
        pos = nx.spring_layout(self.G, seed=42)
        
        # Convertir les positions pour l'affichage
        for node_id, (x, y) in pos.items():
            # Normaliser et centrer les positions
            screen_x = (x + 1) * self.canvas_width / 2
            screen_y = (y + 1) * self.canvas_height / 2
            self.positions[node_id] = (screen_x, screen_y)
        
        # Créer les objets Node
        self.nodes = {}
        for node_id in self.G.nodes():
            neighbors = list(self.G.neighbors(node_id))
            position = self.positions[node_id]
            self.nodes[node_id] = Node(node_id, neighbors, position)
        
        # Stocker la liste des arêtes
        self.edges_list = list(self.G.edges())
        
        self.log_message("Graphe créé avec succès")
        return True
    
    def simulate_timestep(self):
        """Simule un pas de temps"""
        if not self.is_running:
            return
            
        self.timestep += 1
        
        # Traiter les messages
        messages_to_process = Node.pending_messages.copy()
        Node.pending_messages.clear()
        
        for message in messages_to_process:
            if message.destination in self.nodes:
                self.nodes[message.destination].process_message(message)
                self.log_message(f"T{self.timestep}: Node {message.destination} traite {message.message_type.value} de {message.sender}")
        
        # Envoyer les nouveaux messages
        for node in self.nodes.values():
            node.send_messages()
        
        # Vérifier si la simulation doit s'arrêter
        if not Node.pending_messages and all(not node.outbox for node in self.nodes.values()):
            self.is_running = False
            self.log_message("Simulation terminée - Plus de messages à traiter")
        
        return len(messages_to_process) > 0
    
    def start_lock_simulation(self, edge_index):
        """Démarre une simulation de verrouillage"""
        if not self.edges_list or edge_index >= len(self.edges_list):
            return False
            
        self.target_edge = self.edges_list[edge_index]
        initiator_node = self.target_edge[0]
        
        # Réinitialiser l'état
        Node.pending_messages.clear()
        for node in self.nodes.values():
            node.is_locked = False
            node.locked_edge = None
            node.outbox.clear()
            for neighbor in node.unlock_status:
                node.unlock_status[neighbor] = False
        
        self.timestep = 0
        self.is_running = True
        
        # Initier le verrouillage
        self.nodes[initiator_node].initiate_lock(self.target_edge)
        self.log_message(f"Début du verrouillage sur l'arête {self.target_edge} par le nœud {initiator_node}")
        
        return True
    
    def stop_simulation(self):
        """Arrête la simulation"""
        self.is_running = False
        self.log_message("Simulation arrêtée par l'utilisateur")
    
    def log_message(self, message):
        """Ajoute un message au log"""
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(self.log_messages) > 50:  # Limiter le nombre de messages
            self.log_messages.pop(0)

# Instance globale de la simulation
sim = GraphSimulation()

def update_canvas():
    """Met à jour l'affichage du canvas"""
    dpg.delete_item("canvas", children_only=True)
    
    if not sim.G:
        return
    
    # Dessiner les arêtes
    for edge in sim.edges_list:
        node1, node2 = edge
        pos1 = sim.positions[node1]
        pos2 = sim.positions[node2]
        
        # Couleur de l'arête
        if sim.target_edge and (edge == sim.target_edge or (edge[1], edge[0]) == sim.target_edge):
            color = [255, 0, 0, 255]  # Rouge pour l'arête cible
        else:
            color = [100, 100, 100, 255]  # Gris pour les autres
        
        dpg.draw_line(pos1, pos2, color=color, thickness=2, parent="canvas")
    
    # Dessiner les nœuds
    for node_id, node in sim.nodes.items():
        pos = sim.positions[node_id]
        color = node.get_color(sim.target_edge)
        
        # Cercle du nœud
        dpg.draw_circle(pos, sim.node_radius, color=color, fill=color, parent="canvas")
        
        # Contour noir
        dpg.draw_circle(pos, sim.node_radius, color=[0, 0, 0, 255], thickness=2, parent="canvas")
        
        # Texte du nœud
        text_pos = (pos[0] - 5, pos[1] - 5)
        dpg.draw_text(text_pos, str(node_id), color=[0, 0, 0, 255], size=16, parent="canvas")

def create_graph_callback():
    """Callback pour créer un nouveau graphe"""
    sim.create_graph()
    update_canvas()
    update_edge_combo()
    update_log()

def start_simulation_callback():
    """Callback pour démarrer la simulation"""
    edge_index = dpg.get_value("edge_combo")
    if sim.start_lock_simulation(edge_index):
        update_canvas()
        update_log()

def stop_simulation_callback():
    """Callback pour arrêter la simulation"""
    sim.stop_simulation()
    update_log()

def step_simulation_callback():
    """Callback pour un pas de simulation manuel"""
    if sim.simulate_timestep():
        update_canvas()
        update_log()
        update_status()

def update_edge_combo():
    """Met à jour la liste des arêtes dans le combo"""
    if sim.edges_list:
        edge_labels = [f"{edge[0]} - {edge[1]}" for edge in sim.edges_list]
        dpg.configure_item("edge_combo", items=edge_labels)
        if edge_labels:
            dpg.set_value("edge_combo", 0)

def update_log():
    """Met à jour l'affichage du log"""
    log_text = "\n".join(sim.log_messages[-20:])  # Afficher les 20 derniers messages
    dpg.set_value("log_text", log_text)

def update_status():
    """Met à jour les informations de statut"""
    status_info = []
    status_info.append(f"Pas de temps: {sim.timestep}")
    status_info.append(f"Simulation: {'En cours' if sim.is_running else 'Arrêtée'}")
    status_info.append(f"Messages en attente: {len(Node.pending_messages)}")
    
    if sim.target_edge:
        status_info.append(f"Arête cible: {sim.target_edge}")
    
    # Compter les nœuds par état
    locked_count = sum(1 for node in sim.nodes.values() if node.is_locked)
    unlocked_count = len(sim.nodes) - locked_count
    status_info.append(f"Nœuds verrouillés: {locked_count}/{len(sim.nodes)}")
    
    # Nœuds pouvant supprimer l'arête
    safe_nodes = []
    for node in sim.nodes.values():
        if node.is_locked and all(node.unlock_status.values()):
            safe_nodes.append(str(node.id))
    
    if safe_nodes:
        status_info.append(f"Nœuds sûrs: {', '.join(safe_nodes)}")
    
    dpg.set_value("status_text", "\n".join(status_info))

def auto_simulation_thread():
    """Thread pour la simulation automatique"""
    while True:
        if sim.is_running:
            if sim.simulate_timestep():
                # Mettre à jour l'interface dans le thread principal
                dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: [update_canvas(), update_log(), update_status()])
            time.sleep(sim.simulation_speed)
        else:
            time.sleep(0.1)

def speed_callback(sender, app_data):
    """Callback pour changer la vitesse de simulation"""
    sim.simulation_speed = app_data

def main():
    """Fonction principale pour l'interface DearPyGUI"""
    dpg.create_context()
    
    # Fenêtre principale
    with dpg.window(label="Algorithme de Verrouillage Décentralisé", width=1200, height=800, tag="main_window"):
        
        # Section des contrôles
        with dpg.group(horizontal=True):
            with dpg.child_window(width=300, height=780):
                dpg.add_text("Contrôles", color=[255, 255, 0])
                dpg.add_separator()
                
                dpg.add_button(label="Créer nouveau graphe", callback=create_graph_callback, width=-1)
                dpg.add_separator()
                
                dpg.add_text("Arête à verrouiller:")
                dpg.add_combo([], tag="edge_combo", width=-1)
                
                dpg.add_separator()
                dpg.add_button(label="Démarrer simulation", callback=start_simulation_callback, width=-1)
                dpg.add_button(label="Arrêter simulation", callback=stop_simulation_callback, width=-1)
                dpg.add_button(label="Pas manuel", callback=step_simulation_callback, width=-1)
                
                dpg.add_separator()
                dpg.add_text("Vitesse (secondes):")
                dpg.add_slider_float(label="", min_value=0.1, max_value=3.0, default_value=1.0, 
                                   callback=speed_callback, width=-1)
                
                dpg.add_separator()
                dpg.add_text("Statut:", color=[255, 255, 0])
                dpg.add_text("", tag="status_text", wrap=280)
                
                dpg.add_separator()
                dpg.add_text("Légende:", color=[255, 255, 0])
                dpg.add_text("🔵 Nœud libre", color=[100, 150, 255])
                dpg.add_text("🔴 Nœud verrouillé (arête cible)", color=[255, 100, 100])
                dpg.add_text("🟠 Nœud verrouillé (autre arête)", color=[255, 165, 0])
                dpg.add_text("— Arête normale", color=[100, 100, 100])
                dpg.add_text("— Arête cible", color=[255, 0, 0])
            
            # Section graphique et log
            with dpg.child_window(width=880, height=780):
                # Canvas pour le graphe
                with dpg.child_window(width=870, height=420, label="Graphe"):
                    with dpg.drawlist(width=sim.canvas_width, height=sim.canvas_height, tag="canvas"):
                        pass
                
                dpg.add_separator()
                
                # Zone de log
                with dpg.child_window(width=870, height=320, label="Journal des événements"):
                    dpg.add_input_text(multiline=True, readonly=True, tag="log_text", 
                                     width=-1, height=-1, default_value="Cliquez sur 'Créer nouveau graphe' pour commencer")
    
    # Configuration de DearPyGUI
    dpg.create_viewport(title="Simulation d'Algorithme de Verrouillage", width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Démarrer le thread de simulation automatique
    sim_thread = threading.Thread(target=auto_simulation_thread, daemon=True)
    sim_thread.start()
    
    # Créer un graphe initial
    sim.create_graph()
    update_canvas()
    update_edge_combo()
    update_status()
    
    # Boucle principale
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
