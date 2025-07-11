#!/usr/bin/env python3
"""
Interface interactive simplifiée pour l'algorithme de verrouillage décentralisé
Version qui fonctionne bien en mode interactif avec DearPyGUI
"""

import dearpygui.dearpygui as dpg
import networkx as nx
import numpy as np
import threading
import time
import math
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
        self.edge = edge

    def __str__(self):
        return f"Msg({self.sender}→{self.destination}: {self.message_type.value})"

class Node:
    pending_messages = []

    def __init__(self, id, neighbors):
        self.id = id
        self.neighbors = neighbors
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

class GraphSimulator:
    def __init__(self):
        self.G = None
        self.nodes = {}
        self.positions = {}
        self.edges_list = []
        self.is_running = False
        self.timestep = 0
        self.target_edge = None
        self.log_messages = []
        self.auto_mode = False
        
        # Paramètres d'affichage
        self.canvas_width = 500
        self.canvas_height = 400
        self.node_radius = 25
        
    def create_graph(self):
        """Crée un nouveau graphe"""
        self.G = nx.Graph()
        
        # Ajouter des nœuds
        for i in range(7):
            self.G.add_node(i)
        
        # Ajouter des arêtes aléatoires
        edges = []
        for _ in range(8):
            while True:
                edge = tuple(sorted(np.random.choice(list(self.G.nodes()), size=2, replace=False)))
                if edge not in edges and edge[0] != edge[1]:
                    edges.append(edge)
                    break
        
        self.G.add_edges_from(edges)
        
        # Calculer les positions
        pos = nx.spring_layout(self.G, seed=42, iterations=50)
        
        for node_id, (x, y) in pos.items():
            screen_x = (x + 1) * self.canvas_width / 2.2 + 50
            screen_y = (y + 1) * self.canvas_height / 2.2 + 50
            self.positions[node_id] = (screen_x, screen_y)
        
        # Créer les objets Node
        self.nodes = {}
        for node_id in self.G.nodes():
            neighbors = list(self.G.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)
        
        self.edges_list = list(self.G.edges())
        self.reset_simulation()
        self.log_message("Nouveau graphe créé")
        
    def reset_simulation(self):
        """Remet à zéro la simulation"""
        Node.pending_messages.clear()
        self.is_running = False
        self.auto_mode = False
        self.timestep = 0
        self.target_edge = None
        
        for node in self.nodes.values():
            node.is_locked = False
            node.locked_edge = None
            node.outbox.clear()
            for neighbor in node.unlock_status:
                node.unlock_status[neighbor] = False
    
    def start_simulation(self, edge_index):
        """Démarre une simulation"""
        if not self.edges_list or edge_index >= len(self.edges_list):
            return False
            
        self.reset_simulation()
        self.target_edge = self.edges_list[edge_index]
        initiator_node = self.target_edge[0]
        
        self.nodes[initiator_node].initiate_lock(self.target_edge)
        self.is_running = True
        self.log_message(f"Simulation démarrée: nœud {initiator_node} verrouille {self.target_edge}")
        return True
    
    def step_simulation(self):
        """Exécute un pas de simulation"""
        if not self.is_running:
            return False
            
        self.timestep += 1
        
        # Traiter les messages
        messages_to_process = Node.pending_messages.copy()
        Node.pending_messages.clear()
        
        for message in messages_to_process:
            if message.destination in self.nodes:
                self.nodes[message.destination].process_message(message)
        
        # Envoyer les nouveaux messages
        for node in self.nodes.values():
            node.send_messages()
        
        # Log des événements
        if messages_to_process:
            self.log_message(f"Pas {self.timestep}: {len(messages_to_process)} messages traités")
        
        # Vérifier fin de simulation
        if not Node.pending_messages and all(not node.outbox for node in self.nodes.values()):
            self.is_running = False
            safe_nodes = [str(n.id) for n in self.nodes.values() 
                         if n.is_locked and all(n.unlock_status.values())]
            if safe_nodes:
                self.log_message(f"Simulation terminée. Nœuds sûrs: {', '.join(safe_nodes)}")
            else:
                self.log_message("Simulation terminée.")
            return False
        
        return True
    
    def log_message(self, message):
        """Ajoute un message au log"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_messages.append(f"[{timestamp}] {message}")
        if len(self.log_messages) > 15:
            self.log_messages.pop(0)

# Instance globale
sim = GraphSimulator()

def draw_graph():
    """Dessine le graphe sur le canvas"""
    dpg.delete_item("canvas", children_only=True)
    
    if not sim.G:
        dpg.draw_text((150, 200), "Cliquez sur 'Nouveau graphe'", parent="canvas", size=16)
        return
    
    # Dessiner les arêtes
    for edge in sim.edges_list:
        node1, node2 = edge
        pos1 = sim.positions[node1]
        pos2 = sim.positions[node2]
        
        if sim.target_edge and (edge == sim.target_edge or (edge[1], edge[0]) == sim.target_edge):
            color = [255, 50, 50, 255]  # Rouge pour l'arête cible
            thickness = 4
        else:
            color = [120, 120, 120, 255]  # Gris pour les autres
            thickness = 2
        
        dpg.draw_line(pos1, pos2, color=color, thickness=thickness, parent="canvas")
    
    # Dessiner les nœuds
    for node_id, node in sim.nodes.items():
        pos = sim.positions[node_id]
        
        # Couleur selon l'état
        if not node.is_locked:
            color = [100, 150, 255, 255]  # Bleu (libre)
        elif sim.target_edge and node.locked_edge == sim.target_edge:
            if all(node.unlock_status.values()):
                color = [50, 255, 50, 255]  # Vert (peut supprimer)
            else:
                color = [255, 100, 100, 255]  # Rouge (verrouillé cible)
        else:
            color = [255, 165, 0, 255]  # Orange (verrouillé autre)
        
        # Cercle du nœud
        dpg.draw_circle(pos, sim.node_radius, color=color, fill=color, parent="canvas")
        
        # Contour
        dpg.draw_circle(pos, sim.node_radius, color=[0, 0, 0, 255], thickness=2, parent="canvas")
        
        # Texte du nœud
        text_pos = (pos[0] - 8, pos[1] - 8)
        dpg.draw_text(text_pos, str(node_id), color=[0, 0, 0, 255], size=16, parent="canvas")

def update_status():
    """Met à jour l'affichage du statut"""
    if not sim.G:
        dpg.set_value("status_text", "Aucun graphe chargé")
        return
    
    status = []
    status.append(f"Pas de temps: {sim.timestep}")
    status.append(f"État: {'En cours' if sim.is_running else 'Arrêté'}")
    status.append(f"Messages: {len(Node.pending_messages)}")
    
    if sim.target_edge:
        status.append(f"Arête cible: {sim.target_edge}")
    
    locked_count = sum(1 for n in sim.nodes.values() if n.is_locked)
    status.append(f"Nœuds verrouillés: {locked_count}/{len(sim.nodes)}")
    
    dpg.set_value("status_text", "\\n".join(status))

def update_log():
    """Met à jour l'affichage du log"""
    log_text = "\\n".join(sim.log_messages[-10:])
    dpg.set_value("log_text", log_text)

def update_combo():
    """Met à jour la liste des arêtes"""
    if sim.edges_list:
        edge_labels = [f"{edge[0]} - {edge[1]}" for edge in sim.edges_list]
        dpg.configure_item("edge_combo", items=edge_labels)
        if edge_labels:
            dpg.set_value("edge_combo", 0)

# Callbacks
def new_graph_callback():
    sim.create_graph()
    draw_graph()
    update_status()
    update_combo()
    update_log()

def start_callback():
    edge_index = dpg.get_value("edge_combo")
    if sim.start_simulation(edge_index):
        draw_graph()
        update_status()
        update_log()

def step_callback():
    if sim.step_simulation():
        draw_graph()
        update_status()
        update_log()

def reset_callback():
    sim.reset_simulation()
    draw_graph()
    update_status()
    update_log()

def auto_callback():
    sim.auto_mode = not sim.auto_mode
    label = "Auto ON" if sim.auto_mode else "Auto OFF"
    dpg.set_item_label("auto_button", label)

def auto_simulation_worker():
    """Thread pour la simulation automatique"""
    while True:
        if sim.auto_mode and sim.is_running:
            if sim.step_simulation():
                # Programmer la mise à jour dans le thread principal
                dpg.set_frame_callback(dpg.get_frame_count() + 1, 
                                     lambda: [draw_graph(), update_status(), update_log()])
            time.sleep(1.0)  # 1 seconde entre les pas
        else:
            time.sleep(0.1)

def main():
    """Interface principale DearPyGUI"""
    dpg.create_context()
    
    with dpg.window(label="Algorithme de Verrouillage Décentralisé - Version Interactive", 
                    width=800, height=600, tag="main_window"):
        
        with dpg.group(horizontal=True):
            # Panneau de contrôle
            with dpg.child_window(width=250, height=580):
                dpg.add_text("🎮 CONTRÔLES", color=[255, 255, 100])
                dpg.add_separator()
                
                dpg.add_button(label="🔄 Nouveau graphe", callback=new_graph_callback, width=-1)
                dpg.add_separator()
                
                dpg.add_text("Arête à verrouiller:")
                dpg.add_combo([], tag="edge_combo", width=-1)
                
                dpg.add_button(label="▶️ Démarrer", callback=start_callback, width=-1)
                dpg.add_button(label="⏭️ Pas manuel", callback=step_callback, width=-1)
                dpg.add_button(label="🔄 Réinitialiser", callback=reset_callback, width=-1)
                dpg.add_button(label="Auto OFF", callback=auto_callback, width=-1, tag="auto_button")
                
                dpg.add_separator()
                dpg.add_text("📊 STATUT:", color=[255, 255, 100])
                dpg.add_text("", tag="status_text", wrap=230)
                
                dpg.add_separator()
                dpg.add_text("🎨 LÉGENDE:", color=[255, 255, 100])
                dpg.add_text("🔵 Libre", color=[100, 150, 255])
                dpg.add_text("🔴 Verrouillé (cible)", color=[255, 100, 100])
                dpg.add_text("🟢 Peut supprimer", color=[50, 255, 50])
                dpg.add_text("🟠 Verrouillé (autre)", color=[255, 165, 0])
                
                dpg.add_separator()
                dpg.add_text("📝 JOURNAL:", color=[255, 255, 100])
                dpg.add_text("", tag="log_text", wrap=230)
            
            # Zone graphique
            with dpg.child_window(width=530, height=580):
                dpg.add_text("🌐 GRAPHE", color=[255, 255, 100])
                dpg.add_separator()
                
                with dpg.drawlist(width=sim.canvas_width, height=sim.canvas_height, tag="canvas"):
                    pass
    
    # Configuration DearPyGUI
    dpg.create_viewport(title="Algorithme de Verrouillage Décentralisé", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Démarrer le thread de simulation automatique
    auto_thread = threading.Thread(target=auto_simulation_worker, daemon=True)
    auto_thread.start()
    
    # Initialisation
    sim.create_graph()
    draw_graph()
    update_status()
    update_combo()
    update_log()
    
    # Boucle principale
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
