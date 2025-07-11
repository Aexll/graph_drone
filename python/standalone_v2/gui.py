"""
Interface graphique pour la simulation de drones distribués
"""

import dearpygui.dearpygui as dpg
import math
from typing import Optional
from network import DroneNetwork


class DroneSimulationGUI:
    """Interface graphique pour la simulation de drones"""
    
    def __init__(self):
        self.network = DroneNetwork()
        self.simulation_running = False
        self.selected_drone = None
        
        # Paramètres d'affichage
        self.drone_radius = 15
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Variables pour l'interaction
        self.mouse_x = 0
        self.mouse_y = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        """Configurer l'interface graphique"""
        dpg.create_context()
        
        # Fenêtre principale
        with dpg.window(label="Simulation de Drones Distribués", width=1400, height=900, tag="main_window"):
            
            # Contrôles
            with dpg.group(horizontal=True):
                dpg.add_button(label="Générer Réseau Aléatoire", callback=self.generate_random_network)
                dpg.add_button(label="Démarrer/Arrêter Simulation", callback=self.toggle_simulation)
                dpg.add_button(label="Étape Unique", callback=self.single_step)
                dpg.add_button(label="Réinitialiser", callback=self.reset_simulation)
                
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="Nombre de drones", default_value=8, tag="num_drones", width=150)
                dpg.add_input_float(label="Probabilité de connexion", default_value=0.3, tag="connection_prob", width=150, min_value=0.0, max_value=1.0)
                
            dpg.add_separator()
            
            # Zone principale avec canvas et panneau d'informations
            with dpg.group(horizontal=True):
                # Canvas pour l'affichage du réseau
                with dpg.child_window(width=self.canvas_width + 20, height=self.canvas_height + 40):
                    dpg.add_text("Réseau de Drones (Cliquez sur un drone pour voir ses détails)")
                    with dpg.drawlist(width=self.canvas_width, height=self.canvas_height, tag="canvas"):
                        pass
                
                # Panneau d'informations détaillées
                with dpg.child_window(width=550, height=self.canvas_height + 40):
                    dpg.add_text("Informations Détaillées", tag="info_title")
                    dpg.add_separator()
                    
                    with dpg.child_window(height=200, tag="drone_details_window"):
                        dpg.add_text("Sélectionnez un drone pour voir ses détails", tag="drone_details")
                    
                    dpg.add_separator()
                    
                    with dpg.child_window(height=150, tag="critical_edges_window"):
                        dpg.add_text("Arêtes critiques:", tag="critical_edges_title")
                        dpg.add_text("Aucune détectée", tag="critical_edges_info")
                    
                    dpg.add_separator()
                    
                    with dpg.child_window(height=200, tag="network_info_window"):
                        dpg.add_text("État du réseau:", tag="network_info_title")
                        dpg.add_text("", tag="network_info")
                        
            # Configurer les callbacks de souris
            with dpg.handler_registry():
                dpg.add_mouse_move_handler(callback=self.mouse_move_callback)
                dpg.add_mouse_click_handler(callback=self.mouse_click_callback)
                
        dpg.create_viewport(title="Simulation de Drones Distribués", width=1420, height=920)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        
    def mouse_move_callback(self, sender, app_data):
        """Callback pour le mouvement de la souris"""
        self.mouse_x, self.mouse_y = app_data
        
        # Ajuster les coordonnées pour le canvas (approximation)
        canvas_x = self.mouse_x - 30  # Offset approximatif
        canvas_y = self.mouse_y - 120  # Offset approximatif
        
        # Vérifier si la souris survole un drone
        self.network.reset_hover_states()
        hovered_drone = self.network.get_drone_at_position(canvas_x, canvas_y, self.drone_radius)
        
        if hovered_drone is not None:
            self.network.drones[hovered_drone].is_hovered = True
            
        self.update_display()
        
    def mouse_click_callback(self, sender, app_data):
        """Callback pour les clics de souris"""
        if app_data == 0:  # Clic gauche
            # Ajuster les coordonnées pour le canvas
            canvas_x = self.mouse_x - 30
            canvas_y = self.mouse_y - 120
            
            clicked_drone = self.network.get_drone_at_position(canvas_x, canvas_y, self.drone_radius)
            
            if clicked_drone is not None:
                # Désélectionner le drone précédent
                self.network.reset_selection_states()
                
                # Sélectionner le nouveau drone
                self.network.drones[clicked_drone].is_selected = True
                self.selected_drone = clicked_drone
                
                self.update_drone_details()
                self.update_display()
                
    def generate_random_network(self):
        """Générer un nouveau réseau aléatoire"""
        num_drones = dpg.get_value("num_drones")
        connection_prob = dpg.get_value("connection_prob")
        
        self.network.generate_random_network(num_drones, connection_prob)
        self.selected_drone = None
        self.update_display()
        self.update_drone_details()
        
    def toggle_simulation(self):
        """Démarrer/arrêter la simulation"""
        self.simulation_running = not self.simulation_running
        
    def single_step(self):
        """Exécuter une seule étape de simulation"""
        self.network.step()
        self.update_display()
        self.update_drone_details()
        
    def reset_simulation(self):
        """Réinitialiser la simulation"""
        self.simulation_running = False
        for drone in self.network.drones.values():
            drone.reset_algorithms()
        self.update_display()
        self.update_drone_details()
        
    def update_display(self):
        """Mettre à jour l'affichage"""
        dpg.delete_item("canvas", children_only=True)
        
        # Dessiner les connexions
        critical_edges = self.network.get_all_critical_edges()
        
        for conn in self.network.connections:
            drone1 = self.network.drones[conn[0]]
            drone2 = self.network.drones[conn[1]]
            
            # Couleur rouge pour les arêtes critiques, bleu sinon
            color = (255, 0, 0) if conn in critical_edges else (100, 150, 255)
            thickness = 3 if conn in critical_edges else 1
            
            dpg.draw_line(
                parent="canvas",
                p1=(drone1.x, drone1.y),
                p2=(drone2.x, drone2.y),
                color=color,
                thickness=thickness
            )
            
        # Dessiner les connexions en évidence pour le drone sélectionné
        if self.selected_drone is not None:
            selected = self.network.drones[self.selected_drone]
            known_nodes = selected.get_known_nodes()
            
            # Dessiner les connexions vers tous les nœuds connus en vert
            for known_node_id in known_nodes:
                if known_node_id in self.network.drones and known_node_id != self.selected_drone:
                    known_drone = self.network.drones[known_node_id]
                    dpg.draw_line(
                        parent="canvas",
                        p1=(selected.x, selected.y),
                        p2=(known_drone.x, known_drone.y),
                        color=(0, 255, 0),
                        thickness=2
                    )
                    
            # Mettre en évidence tous les nœuds connus avec un cercle vert
            for known_node_id in known_nodes:
                if known_node_id in self.network.drones and known_node_id != self.selected_drone:
                    known_drone = self.network.drones[known_node_id]
                    dpg.draw_circle(
                        parent="canvas",
                        center=(known_drone.x, known_drone.y),
                        radius=self.drone_radius + 8,
                        color=(0, 200, 0),
                        thickness=2
                    )
            
        # Dessiner les drones
        for drone in self.network.drones.values():
            # Couleur selon la phase
            if drone.algorithm_phase == "critical_detection":
                color = (0, 255, 0)  # Vert
            elif drone.algorithm_phase == "connectivity":
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 255, 0)  # Jaune
            
            # Drone sélectionné en bleu
            if drone.is_selected:
                color = (0, 100, 255)
            
            dpg.draw_circle(
                parent="canvas",
                center=(drone.x, drone.y),
                radius=self.drone_radius,
                color=color,
                fill=color
            )
            
            # Cercle de survol jaune
            if drone.is_hovered:
                dpg.draw_circle(
                    parent="canvas",
                    center=(drone.x, drone.y),
                    radius=self.drone_radius + 8,
                    color=(255, 255, 0),
                    thickness=4
                )
            
            # Afficher l'ID du drone
            dpg.draw_text(
                parent="canvas",
                pos=(drone.x - 5, drone.y - 5),
                text=str(drone.id),
                color=(0, 0, 0),
                size=12
            )
            
        # Mettre à jour les informations réseau
        self.update_network_info()
        
    def update_drone_details(self):
        """Mettre à jour les détails du drone sélectionné"""
        if self.selected_drone is None:
            dpg.set_value("drone_details", "Sélectionnez un drone pour voir ses détails")
            return
            
        drone = self.network.drones[self.selected_drone]
        details = drone.get_detailed_info()
        
        detail_text = f"Drone {details['id']} - Détails:\n\n"
        detail_text += f"Position: ({details['position'][0]:.1f}, {details['position'][1]:.1f})\n"
        detail_text += f"Phase: {details['phase']}\n"
        detail_text += f"Itération: {details['iteration']}\n"
        detail_text += f"Estimation n: {details['n_estimate']}\n\n"
        
        detail_text += f"Voisins directs: {details['neighbors']}\n\n"
        
        detail_text += "Valeurs ξ (connectivité):\n"
        for node_id, xi_val in details['xi'].items():
            detail_text += f"  ξ[{details['id']},{node_id}] = {xi_val}\n"
        
        detail_text += "\nValeurs ω (distances):\n"
        for node_id, omega_val in details['omega'].items():
            detail_text += f"  ω[{details['id']},{node_id}] = {omega_val}\n"
            
        detail_text += f"\nNœuds connus: {details['known_nodes']}\n"
        detail_text += f"Arêtes critiques détectées: {details['critical_edges']}\n"
        
        dpg.set_value("drone_details", detail_text)
        
    def update_network_info(self):
        """Mettre à jour les informations du réseau"""
        # Arêtes critiques
        critical_edges = self.network.get_all_critical_edges()
        critical_info = f"Arêtes critiques détectées ({len(critical_edges)}):\n"
        for edge in critical_edges:
            critical_info += f"  {edge[0]} ↔ {edge[1]}\n"
        dpg.set_value("critical_edges_info", critical_info)
        
        # Informations générales du réseau
        network_info = f"Nombre de drones: {len(self.network.drones)}\n"
        network_info += f"Nombre de connexions: {len(self.network.connections)}\n\n"
        
        # État des phases
        phases = {}
        for drone in self.network.drones.values():
            phase = drone.algorithm_phase
            phases[phase] = phases.get(phase, 0) + 1
            
        network_info += "Répartition des phases:\n"
        for phase, count in phases.items():
            network_info += f"  {phase}: {count} drones\n"
        
        dpg.set_value("network_info", network_info)
        
    def run(self):
        """Lancer la simulation"""
        # Générer un réseau initial
        self.generate_random_network()
        
        while dpg.is_dearpygui_running():
            if self.simulation_running:
                self.network.step()
                self.update_display()
                self.update_drone_details()
                
            dpg.render_dearpygui_frame()
            
        dpg.destroy_context()
