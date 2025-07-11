"""
Gestionnaire du réseau de drones
"""

import random
from typing import Dict, Set, List, Tuple
from drone import Drone
from message import Message


class DroneNetwork:
    """Gestionnaire du réseau de drones"""
    
    def __init__(self):
        self.drones: Dict[int, Drone] = {}
        self.connections: Set[Tuple[int, int]] = set()
        self.next_id = 0
        
    def add_drone(self, x: float = None, y: float = None) -> int:
        """Ajouter un drone au réseau"""
        if x is None:
            x = random.uniform(50, 750)
        if y is None:
            y = random.uniform(50, 550)
            
        drone_id = self.next_id
        self.next_id += 1
        
        drone = Drone(drone_id, x, y)
        self.drones[drone_id] = drone
        return drone_id
        
    def remove_drone(self, drone_id: int):
        """Supprimer un drone du réseau"""
        if drone_id not in self.drones:
            return
            
        # Supprimer toutes les connexions
        connections_to_remove = []
        for conn in self.connections:
            if drone_id in conn:
                connections_to_remove.append(conn)
                
        for conn in connections_to_remove:
            self.disconnect_drones(conn[0], conn[1])
            
        del self.drones[drone_id]
        
    def connect_drones(self, drone1_id: int, drone2_id: int):
        """Connecter deux drones"""
        if drone1_id not in self.drones or drone2_id not in self.drones:
            return
            
        connection = tuple(sorted([drone1_id, drone2_id]))
        if connection not in self.connections:
            self.connections.add(connection)
            self.drones[drone1_id].add_neighbor(drone2_id)
            self.drones[drone2_id].add_neighbor(drone1_id)
            
    def disconnect_drones(self, drone1_id: int, drone2_id: int):
        """Déconnecter deux drones"""
        connection = tuple(sorted([drone1_id, drone2_id]))
        if connection in self.connections:
            self.connections.remove(connection)
            self.drones[drone1_id].remove_neighbor(drone2_id)
            self.drones[drone2_id].remove_neighbor(drone1_id)
            
    def generate_random_network(self, num_drones: int, connection_probability: float = 0.3):
        """Générer un réseau aléatoire de drones"""
        self.drones.clear()
        self.connections.clear()
        self.next_id = 0
        
        # Créer les drones
        for _ in range(num_drones):
            self.add_drone()
            
        # Créer les connexions aléatoires
        drone_ids = list(self.drones.keys())
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                if random.random() < connection_probability:
                    self.connect_drones(drone_ids[i], drone_ids[j])
                    
        # S'assurer qu'il y a au moins une connexion
        if not self.connections and len(drone_ids) >= 2:
            self.connect_drones(drone_ids[0], drone_ids[1])
            
    def step(self):
        """Exécuter une étape de simulation pour tous les drones"""
        # Collecter tous les messages
        all_messages = []
        for drone in self.drones.values():
            messages = drone.get_messages_to_send()
            all_messages.extend(messages)
            
        # Distribuer les messages
        for msg in all_messages:
            if msg.receiver_id in self.drones:
                self.drones[msg.receiver_id].receive_message(msg)
                
        # Exécuter une étape pour chaque drone
        for drone in self.drones.values():
            drone.step()
            
    def get_all_critical_edges(self) -> Set[Tuple[int, int]]:
        """Obtenir toutes les arêtes critiques détectées"""
        critical_edges = set()
        for drone in self.drones.values():
            critical_edges.update(drone.critical_edges)
        return critical_edges
        
    def get_drone_at_position(self, x: float, y: float, radius: float = 20) -> int:
        """Trouver un drone à une position donnée"""
        for drone_id, drone in self.drones.items():
            distance = ((drone.x - x) ** 2 + (drone.y - y) ** 2) ** 0.5
            if distance <= radius:
                return drone_id
        return None
        
    def reset_hover_states(self):
        """Réinitialiser tous les états de survol"""
        for drone in self.drones.values():
            drone.is_hovered = False
            
    def reset_selection_states(self):
        """Réinitialiser tous les états de sélection"""
        for drone in self.drones.values():
            drone.is_selected = False
