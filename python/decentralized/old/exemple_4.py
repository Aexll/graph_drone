import dearpygui.dearpygui as dpg
import networkx as nx
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import copy
import threading
import json

class DistributedNode:
    """
    Représente un nœud dans le réseau distribué avec support GUI
    """
    
    def __init__(self, node_id: int, position: Tuple[float, float], initial_neighbors: List[int] = None):
        """
        Initialise un nœud distribué
        
        Args:
            node_id: Identifiant unique du nœud
            position: Position (x, y) pour l'affichage
            initial_neighbors: Liste des voisins initiaux
        """
        self.node_id = node_id
        self.position = position
        self.neighbors = set(initial_neighbors) if initial_neighbors else set()
        
        # Variables locales pour l'algorithme xi-omega
        self.xi = defaultdict(float)
        self.omega = defaultdict(lambda: float('inf'))
        
        # Initialisation
        self.xi[node_id] = 1.0
        self.omega[node_id] = 0.0
        
        # Messages
        self.outgoing_messages = {}
        self.incoming_messages = {}
        
        # Historique
        self.previous_xi = {}
        self.previous_omega = {}
        
        # Métadonnées
        self.iteration = 0
        self.converged = False
        self.known_nodes = {node_id}
        
        # Variables pour l'affichage
        self.color = [100, 150, 255]  # Couleur par défaut
        self.is_active = False
        self.message_count = 0
        self.last_update_time = time.time()
    
    def add_neighbor(self, neighbor_id: int):
        """Ajoute un nouveau voisin"""
        self.neighbors.add(neighbor_id)
    
    def remove_neighbor(self, neighbor_id: int):
        """Supprime un voisin"""
        self.neighbors.discard(neighbor_id)
    
    def prepare_messages(self) -> Dict[int, Dict]:
        """Prépare les messages à envoyer aux voisins"""
        messages = {}
        
        message = {
            'sender': self.node_id,
            'iteration': self.iteration,
            'xi': dict(self.xi),
            'omega': dict(self.omega),
            'known_nodes': self.known_nodes.copy(),
            'timestamp': time.time()
        }
        
        for neighbor in self.neighbors:
            messages[neighbor] = copy.deepcopy(message)
        
        self.message_count += len(messages)
        return messages
    
    def receive_message(self, sender_id: int, message: Dict):
        """Reçoit un message d'un voisin"""
        if sender_id not in self.incoming_messages:
            self.incoming_messages[sender_id] = []
        
        self.incoming_messages[sender_id].append(message)
        self.is_active = True
    
    def update_xi_omega(self):
        """Met à jour les valeurs xi et omega"""
        self.previous_xi = dict(self.xi)
        self.previous_omega = dict(self.omega)
        
        # Découvrir nouveaux nœuds
        all_known_nodes = set(self.known_nodes)
        
        for sender_id, messages in self.incoming_messages.items():
            if messages:
                latest_message = messages[-1]
                all_known_nodes.update(latest_message['known_nodes'])
        
        self.known_nodes = all_known_nodes
        
        # Mettre à jour xi et omega
        for target_node in self.known_nodes:
            if target_node == self.node_id:
                continue
            
            # Calculer nouvelle valeur xi
            candidates = [self.xi[target_node]]
            
            for neighbor_id in self.neighbors:
                if neighbor_id in self.incoming_messages:
                    messages = self.incoming_messages[neighbor_id]
                    if messages:
                        latest_message = messages[-1]
                        neighbor_xi = latest_message['xi']
                        if target_node in neighbor_xi:
                            candidates.append(neighbor_xi[target_node])
            
            new_xi = max(candidates)
            
            # Mettre à jour omega
            if new_xi == self.xi[target_node]:
                new_omega = self.omega[target_node]
            elif new_xi > self.xi[target_node]:
                min_distances = []
                
                for neighbor_id in self.neighbors:
                    if neighbor_id in self.incoming_messages:
                        messages = self.incoming_messages[neighbor_id]
                        if messages:
                            latest_message = messages[-1]
                            neighbor_omega = latest_message['omega']
                            if (target_node in neighbor_omega and 
                                neighbor_omega[target_node] != float('inf')):
                                min_distances.append(neighbor_omega[target_node] + 1)
                
                if min_distances:
                    new_omega = min(min_distances)
                else:
                    new_omega = float('inf')
            else:
                new_omega = self.omega[target_node]
            
            self.xi[target_node] = new_xi
            self.omega[target_node] = new_omega
        
        self.incoming_messages.clear()
        self.iteration += 1
        self.last_update_time = time.time()
    
    def has_converged(self, tolerance: float = 1e-10) -> bool:
        """Vérifie si le nœud a convergé"""
        if not self.previous_xi or not self.previous_omega:
            return False
        
        for node in self.known_nodes:
            if node == self.node_id:
                continue
            
            prev_xi = self.previous_xi.get(node, 0)
            curr_xi = self.xi.get(node, 0)
            if abs(curr_xi - prev_xi) > tolerance:
                return False
            
            prev_omega = self.previous_omega.get(node, float('inf'))
            curr_omega = self.omega.get(node, float('inf'))
            if abs(curr_omega - prev_omega) > tolerance:
                return False
        
        return True
    
    def get_display_color(self) -> List[int]:
        """Retourne la couleur d'affichage du nœud"""
        if self.has_converged():
            return [100, 255, 100]  # Vert si convergé
        elif self.is_active:
            return [255, 100, 100]  # Rouge si actif
        else:
            return [100, 150, 255]  # Bleu par défaut


class DistributedNetworkGUI:
    """
    Interface graphique pour l'algorithme distribué xi-omega
    """
    
    def __init__(self, graph: nx.Graph = None):
        """
        Initialise l'interface graphique
        
        Args:
            graph: Graphe NetworkX initial (optionnel)
        """
        self.graph = graph or nx.Graph()
        self.nodes = {}
        self.canvas_width = 800
        self.canvas_height = 600
        self.node_radius = 15
        self.running = False
        self.auto_step = False
        self.step_delay = 1.0
        self.iteration_count = 0
        
        # Statistiques
        self.total_messages = 0
        self.convergence_times = {}
        
        # Thread pour l'animation
        self.animation_thread = None
        self.stop_animation = False
        
        self._setup_gui()
        if self.graph.nodes():
            self._create_distributed_nodes()
    
    def _setup_gui(self):
        """Configure l'interface graphique"""
        dpg.create_context()
        
        # Configuration de la fenêtre principale
        with dpg.window(label="Algorithme Xi-Omega Distribué", 
                       width=1200, height=800, tag="main_window"):
            
            # Barre de contrôle
            with dpg.group(horizontal=True):
                dpg.add_button(label="Créer Graphe Exemple", callback=self._create_example_graph)
                dpg.add_button(label="Graphe Aléatoire", callback=self._create_random_graph)
                dpg.add_button(label="Réinitialiser", callback=self._reset_algorithm)
                dpg.add_separator()
                dpg.add_button(label="Étape", callback=self._single_step)
                dpg.add_button(label="Démarrer/Arrêter", callback=self._toggle_auto_step)
                dpg.add_slider_float(label="Délai (s)", default_value=1.0, min_value=0.1, 
                                   max_value=5.0, tag="delay_slider", callback=self._update_delay)
            
            # Statistiques
            with dpg.group(horizontal=True):
                dpg.add_text("Itération: 0", tag="iteration_text")
                dpg.add_text("Messages totaux: 0", tag="messages_text")
                dpg.add_text("Nœuds convergés: 0/0", tag="convergence_text")
            
            # Canvas pour dessiner le graphe
            with dpg.drawlist(width=self.canvas_width, height=self.canvas_height, tag="canvas"):
                pass
            
            # Panneau d'information
            with dpg.child_window(width=380, height=400, tag="info_panel"):
                dpg.add_text("Informations des nœuds", tag="info_title")
                dpg.add_separator()
                with dpg.group(tag="node_info_group"):
                    pass
        
        # Configuration des thèmes
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70))
        
        dpg.bind_theme(global_theme)
        
        # Configuration de la fenêtre
        dpg.create_viewport(title="Algorithme Xi-Omega Distribué", width=1200, height=800)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
    
    def _create_example_graph(self):
        """Crée un graphe d'exemple"""
        # Graphe en chemin avec une branche
        self.graph = nx.Graph()
        self.graph.add_edges_from([
            (1, 2), (2, 3), (3, 4), (4, 5),  # Chemin principal
            (3, 6), (6, 7)  # Branche
        ])
        self._create_distributed_nodes()
        self._reset_algorithm()
    
    def _create_random_graph(self):
        """Crée un graphe aléatoire"""
        n = 8
        p = 0.3
        self.graph = nx.erdos_renyi_graph(n, p)
        
        # Renommer les nœuds pour commencer à 1
        mapping = {i: i+1 for i in range(n)}
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
        # S'assurer que le graphe est connecté
        if not nx.is_connected(self.graph):
            self.graph = nx.connected_component_subgraphs(self.graph)[0]
        
        self._create_distributed_nodes()
        self._reset_algorithm()
    
    def _create_distributed_nodes(self):
        """Crée les nœuds distribués avec positions"""
        self.nodes = {}
        
        # Calculer positions avec spring layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Normaliser les positions pour le canvas
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            margin = 50
            for node_id, (x, y) in pos.items():
                # Normaliser entre margin et canvas_size - margin
                norm_x = margin + (x - x_min) / (x_max - x_min) * (self.canvas_width - 2 * margin)
                norm_y = margin + (y - y_min) / (y_max - y_min) * (self.canvas_height - 2 * margin)
                
                neighbors = list(self.graph.neighbors(node_id))
                self.nodes[node_id] = DistributedNode(node_id, (norm_x, norm_y), neighbors)
    
    def _reset_algorithm(self):
        """Réinitialise l'algorithme"""
        self.iteration_count = 0
        self.total_messages = 0
        self.convergence_times = {}
        
        for node in self.nodes.values():
            node.xi = defaultdict(float)
            node.omega = defaultdict(lambda: float('inf'))
            node.xi[node.node_id] = 1.0
            node.omega[node.node_id] = 0.0
            node.known_nodes = {node.node_id}
            node.iteration = 0
            node.converged = False
            node.is_active = False
            node.message_count = 0
        
        self._update_display()
    
    def _single_step(self):
        """Effectue une étape de l'algorithme"""
        if not self.nodes:
            return
        
        # Phase 1: Préparer les messages
        all_messages = {}
        for node_id, node in self.nodes.items():
            all_messages[node_id] = node.prepare_messages()
        
        # Phase 2: Distribuer les messages
        for sender_id, messages in all_messages.items():
            for receiver_id, message in messages.items():
                if receiver_id in self.nodes:
                    self.nodes[receiver_id].receive_message(sender_id, message)
        
        # Phase 3: Mettre à jour les nœuds
        for node in self.nodes.values():
            node.update_xi_omega()
            node.is_active = False
        
        # Compter les messages
        self.total_messages += sum(len(messages) for messages in all_messages.values())
        self.iteration_count += 1
        
        # Vérifier convergence
        for node in self.nodes.values():
            if node.has_converged() and node.node_id not in self.convergence_times:
                self.convergence_times[node.node_id] = self.iteration_count
        
        self._update_display()
    
    def _toggle_auto_step(self):
        """Démarre/arrête l'exécution automatique"""
        self.auto_step = not self.auto_step
        
        if self.auto_step and not self.animation_thread:
            self.stop_animation = False
            self.animation_thread = threading.Thread(target=self._animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()
        elif not self.auto_step:
            self.stop_animation = True
            if self.animation_thread:
                self.animation_thread.join(timeout=1.0)
                self.animation_thread = None
    
    def _animation_loop(self):
        """Boucle d'animation pour l'exécution automatique"""
        while self.auto_step and not self.stop_animation:
            self._single_step()
            time.sleep(self.step_delay)
            
            # Arrêter si convergence globale
            if all(node.has_converged() for node in self.nodes.values()):
                self.auto_step = False
                break
    
    def _update_delay(self, sender, value):
        """Met à jour le délai entre les étapes"""
        self.step_delay = value
    
    def _update_display(self):
        """Met à jour l'affichage"""
        self._draw_graph()
        self._update_info_panel()
        self._update_statistics()
    
    def _draw_graph(self):
        """Dessine le graphe sur le canvas"""
        dpg.delete_item("canvas", children_only=True)
        
        if not self.nodes:
            return
        
        # Dessiner les arêtes
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in self.nodes and node2 in self.nodes:
                pos1 = self.nodes[node1].position
                pos2 = self.nodes[node2].position
                
                dpg.draw_line(parent="canvas", p1=pos1, p2=pos2, 
                             color=[200, 200, 200], thickness=2)
        
        # Dessiner les nœuds
        for node in self.nodes.values():
            pos = node.position
            color = node.get_display_color()
            
            # Cercle du nœud
            dpg.draw_circle(parent="canvas", center=pos, radius=self.node_radius,
                           color=color, fill=color, thickness=2)
            
            # Bordure noire
            dpg.draw_circle(parent="canvas", center=pos, radius=self.node_radius,
                           color=[0, 0, 0], thickness=2)
            
            # Label du nœud
            dpg.draw_text(parent="canvas", pos=(pos[0]-5, pos[1]-5), 
                         text=str(node.node_id), color=[255, 255, 255], size=12)
            
            # Indicateur d'activité
            if node.is_active:
                dpg.draw_circle(parent="canvas", center=pos, radius=self.node_radius+5,
                               color=[255, 255, 0], thickness=3)
    
    def _update_info_panel(self):
        """Met à jour le panneau d'informations"""
        dpg.delete_item("node_info_group", children_only=True)
        
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            
            with dpg.group(parent="node_info_group"):
                status = "Convergé" if node.has_converged() else "En cours"
                color = [0, 255, 0] if node.has_converged() else [255, 255, 0]
                
                dpg.add_text(f"Nœud {node_id} - {status}", color=color)
                dpg.add_text(f"  Itération: {node.iteration}")
                dpg.add_text(f"  Nœuds connus: {len(node.known_nodes)}")
                dpg.add_text(f"  Messages: {node.message_count}")
                
                # Afficher xi et omega (limité pour l'espace)
                xi_display = {k: f"{v:.2f}" for k, v in list(node.xi.items())[:3]}
                omega_display = {k: f"{v:.1f}" if v != float('inf') else "∞" 
                               for k, v in list(node.omega.items())[:3]}
                
                dpg.add_text(f"  Xi: {xi_display}")
                dpg.add_text(f"  Omega: {omega_display}")
                dpg.add_separator()
    
    def _update_statistics(self):
        """Met à jour les statistiques"""
        converged_count = sum(1 for node in self.nodes.values() if node.has_converged())
        total_nodes = len(self.nodes)
        
        dpg.set_value("iteration_text", f"Itération: {self.iteration_count}")
        dpg.set_value("messages_text", f"Messages totaux: {self.total_messages}")
        dpg.set_value("convergence_text", f"Nœuds convergés: {converged_count}/{total_nodes}")
    
    def run(self):
        """Lance l'interface graphique"""
        try:
            while dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
        finally:
            self.stop_animation = True
            if self.animation_thread:
                self.animation_thread.join(timeout=1.0)
            dpg.destroy_context()


def main():
    """
    Fonction principale pour lancer l'interface graphique
    """
    print("=== Interface Graphique - Algorithme Xi-Omega Distribué ===")
    print("Contrôles:")
    print("- Créer Graphe Exemple: Graphe prédéfini")
    print("- Graphe Aléatoire: Génère un graphe aléatoire")
    print("- Étape: Exécute une itération")
    print("- Démarrer/Arrêter: Animation automatique")
    print("- Délai: Contrôle la vitesse d'animation")
    print("\nCodes couleurs:")
    print("- Bleu: Nœud normal")
    print("- Rouge: Nœud actif (traite des messages)")
    print("- Vert: Nœud convergé")
    print("- Bordure jaune: Nœud en cours de traitement")
    
    # Créer et lancer l'interface
    app = DistributedNetworkGUI()
    app.run()


if __name__ == "__main__":
    main()