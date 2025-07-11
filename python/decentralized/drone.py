import numpy as np
import dearpygui.dearpygui as dpg
from target import Target
import copy
# import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from message import Message, MessageType


class Drone:
    def __init__(self, drone_id, position:np.ndarray, target:Target|None, scan_for_drones_method=None, scan_for_targets_method=None):

        self.id = drone_id
        self.position:np.ndarray = position.astype(np.float32)
        self.target:Target|None = target
        
        self.speed:float = 5.0
        self.range:float = 200.0
        
        # visual
        self.drone_radius:float = 10.0  # Radius of the drone for drawing purposes
        self.target_radius = 10
        self.connection_radius = 10
        
        # world query methods
        self.scan_for_drones: callable = scan_for_drones_method 
        self.scan_for_targets: callable = scan_for_targets_method

        if self.scan_for_drones is None:
            print("Warning: No scan_for_drones method provided, connections will not be established.")
            self.scan_for_drones = lambda position, range: []  # Default to empty list if not provided
        if self.scan_for_targets is None:
            print("Warning: No scan_for_targets method provided, targets will not be found.")
            self.scan_for_targets = lambda position, range: []  # Default to empty list if not provided

        # Debug
        self.total_operations:int = 0 # Number of operations performed by this drone, for debugging purposes
        self.frame_operations:int = 0 # Number of operations performed in the current frame, for debugging purposes
        self.frame_count:int = 0 # Number of frames since the last reset, for debugging purposes

        
        # Awareness and connections

        self.neighbors: Set[int] = set()
        
        self.n_estimate = 1
        
        self.xi: Dict[int, int] = {self.id: 1}  
        self.omega: Dict[int, float] = {self.id: 0.0}  

        self.neighbors_omega: Dict[int, Dict[int, float]] = {}

        self.critical_edges: Set[Tuple[int, int]] = set()
        
        # Messages à envoyer et reçus
        self.outgoing_messages: List[Message] = []
        self.incoming_messages: List[Message] = []
        
        # Compteur d'itérations pour les algorithmes
        self.iteration = 0
        self.critical_detection_iteration = 0
        self.algorithm_phase = "neighbor_structure"  # neighbor_structure, connectivity, critical_detection
        
        # État de l'interface utilisateur
        self.is_hovered = False
        self.is_selected = False




    def draw(self, draw_list):

        # Get color based on xi-omega state
        color = self.get_display_color()
        
        # draw the drone
        dpg.draw_circle((self.position[0], self.position[1]), self.drone_radius, color=color, parent=draw_list)

        # range circle
        dpg.draw_circle((self.position[0], self.position[1]), self.range, color=(255, 0, 0, 100), parent=draw_list, fill=True)

        # connections
        for connection in self.connections:
            distance = np.linalg.norm(connection.position - self.position)
            if distance > 0:  # Éviter la division par zéro
                line = (connection.position - self.position)
                direction = line / distance

                link_color = (0, 0, 255, 255)
                # TODO: Add criticality detection if needed
                # if connection in self.criticality:
                #     link_color = (255, 0, 0, 255) 

                dpg.draw_line(
                    self.position + direction * self.drone_radius,
                    self.position + line*0.5,
                    color=link_color, 
                    parent=draw_list
                    )
    
    def draw_notify(self, notify="simple", draw_list=None, value=None):
        """
        Draw a notification on the drone.
        notify: The notification to draw, can be "simple" or "xi".
        """
        if notify == "simple":
            dpg.draw_circle(self.position, self.drone_radius, color=(0,0,255,0), 
                parent=draw_list, fill=(255, 100, 0, 105))
        elif notify == "omega":
            if value is not None:
                dpg.draw_text(self.position - np.array([5, 10]), f"{value}", color=(255, 255, 255), parent=draw_list, size=20)
        elif notify == "delta":
            if value is not None:
                dpg.draw_text(self.position - np.array([0, 0]), f"{value}", color=(255, 255, 255), parent=draw_list, size=20)
        elif notify == "hover":
            dpg.draw_circle(self.position, self.drone_radius+5, color=(255, 255, 0, 255), 
                parent=draw_list, fill=(0, 255, 0, 0))
        elif notify == "xi":
            dpg.draw_circle(self.position, self.drone_radius+5, color=(255, 255, 0, 150), 
                parent=draw_list, fill=(0, 255, 0, 40))


    def update(self, delta_time=0.08):
        """Update the drone's state."""

        self.local_time += delta_time

        # # Execute one step of the xi-omega algorithm
        self.step()

        # # Connection management
        # unknown_drones = self.scan_for_unknown_drones()
        # for drone in unknown_drones:
        #     if self.can_connect(drone):
        #         self.add_connection(drone)
        #         break

        # # Movement
        # if self.target is not None:
        #     self.move_towards_target(delta_time)

    # behaviors

    def clamp_position_to_unbreak_connections(self,position: np.ndarray):
        """Clamp the position to ensure it does not break any connection ranges."""
        for neighbor in self.connections:
            if np.linalg.norm(neighbor.position - position) > neighbor.range:
                # If the position will break the connection range, do not move
                return self.position
        return position

    def move_towards_target(self, delta_time=0.08, stop_distance=10):
        """Move towards the target position, stopping at a certain distance."""
        if self.target is None:
            return
        direction = (self.target.position - self.position)
        if np.linalg.norm(direction) > 0 and np.linalg.norm(direction) > stop_distance:
            direction /= np.linalg.norm(direction)
            movement = direction * self.speed * delta_time
            next_position = self.position + movement

            for neighbor in self.connections:
                if np.linalg.norm(neighbor.position - next_position) > neighbor.range:
                    # If the next position will break the connection range, do not move
                    return

            self.position += movement


    # connection management

    def can_connect(self, drone: int) -> bool:
        """
        Check if this drone can add a connection to another drone.
        Returns True if the connection can be added, False otherwise.

        """
        if drone in self.neighbors:
            return False
        
        # at most 3 connections
        if len(self.neighbors) >= 3:
            return False
        
        # must be in range
        if np.linalg.norm(drone.position - self.position) > self.range:
            return False

        return True

    def can_deconnect(self, drone: int) -> bool:
        """
        Check if this drone can remove a connection to another drone.
        Not depending on the other drone's state.
        """
        if drone not in self.neighbors:
            return False
        
        # must not be the only connection
        if len(self.neighbors) <= 1:
            return False
        
        return True

    def add_connection(self, drone_id: int):
        """ Add a connection to another drone. must be in range """
        # Ajouter la connexion avant de calculer xi pour éviter les incohérences
        if drone_id not in self.neighbors:
            self.neighbors.add(drone_id)
        
        conect_msg = Message(
            sender_id=self.id,
            receiver_id=drone_id,
            msg_type=MessageType.CONNECTION_ADD,
            data={
                "position": self.position.tolist(),
                "sender_id": self.id
            }
        )

        # Envoyer le message de connexion
        self.outgoing_messages.append(conect_msg)

        self.reset_algorithms()

    def remove_connection(self, drone: 'Drone'):
        """Remove a connection to another drone."""
        if drone in self.neighbors:
            self.neighbors.remove(drone)
        if self in drone.neighbors:
            drone.neighbors.remove(self)
        self.reset_algorithms()

    def scan_for_new_connections(self, nb_connections=1) -> list['Drone']:
        """
        Scan for new connections within range.
        nb_connections: Number of connections to establish at most.
        """
        candidates = self.scan_for_drones(self.position, self.range)
        candidates = [drone for drone in candidates if drone != self and drone not in self.connections]
        candidates = sorted(candidates, key=lambda d: np.linalg.norm(d.position - self.position))
        return candidates[:nb_connections]

    def scan_for_unknown_drones(self) -> set:
        """
        Scan for unknown drones within range.
        Returns a set of drones that are not in the known_drones set.
        """
        candidates = self.scan_for_drones(self.position, self.range)
        
        candidates = [drone for drone in candidates if drone != self and drone not in self.connections]
        if len(candidates) == 0:
            return set()
        
        unknown_drones = set(drone for drone in candidates if drone not in self.known_drones and drone != self)

        return unknown_drones


    def reset_algorithms(self):
        """Réinitialiser les algorithmes distribués selon les spécifications du papier"""
        # Conditions initiales selon le papier (k=0)
        self.xi = {self.id: 1}  # ξ[i,j](0) = 1 si j=i, 0 sinon
        self.omega = {self.id: 0}  # ω[i,j](0) = 0 si j=i, ∞ sinon
        
        # Initialiser tous les autres nœuds connus à 0 et ∞
        all_known_nodes = {self.id} | self.neighbors
        for node_id in all_known_nodes:
            if node_id != self.id:
                self.xi[node_id] = 0
                self.omega[node_id] = float('inf')
        
        self.neighbors_omega.clear()
        self.critical_edges.clear()
        self.iteration = 0
        self.critical_detection_iteration = 0
        self.algorithm_phase = "neighbor_structure"
        
        # Estimation du nombre de nœuds basée sur l'index le plus élevé connu
        self.estimate_node_count()
        
    def estimate_node_count(self):
        """Estimation du nombre de noeuds selon les hypothèses du papier"""
        # Selon le papier, chaque noeud connaît le nombre total de noeuds n
        # En pratique, on peut estimer en se basant sur les noeuds découverts
        known_nodes = {self.id} | self.neighbors | set(self.xi.keys()) | set(self.omega.keys())

        # Exclure les noeuds avec des valeurs infinies (non atteignables)
        reachable_nodes = {node for node in known_nodes
                          if self.omega.get(node, float('inf')) != float('inf') or node == self.id}

        # Estimation conservative basée sur l'index le plus élevé trouvé
        if known_nodes:
            max_id = max(known_nodes)
            self.n_estimate = max(max_id + 1, len(reachable_nodes), self.n_estimate)
        else:
            self.n_estimate = max(1, self.n_estimate)
        
    def update_neighbor_structure(self):
        """Algorithme 1: Mise à jour de la structure voisine (xi et omega)"""
        if self.iteration >= self.n_estimate:
            # Transition vers l'algorithme 2 après n étapes
            self.algorithm_phase = "connectivity"
            return
            
        # Créer les messages pour les voisins avec les valeurs actuelles
        for neighbor_id in self.neighbors:
            msg_data = {
                "xi": copy.deepcopy(self.xi),
                "omega": copy.deepcopy(self.omega),
                "iteration": self.iteration,
                "sender_id": self.id
            }
            msg = Message(
                sender_id=self.id,
                receiver_id=neighbor_id,
                msg_type=MessageType.XI_UPDATE,
                data=msg_data
            )
            self.outgoing_messages.append(msg)
            
    def process_xi_update(self, msg: Message):
        """Traiter les mises à jour xi/omega des voisins selon l'algorithme du papier"""
        sender_xi = msg.data["xi"]
        sender_omega = msg.data["omega"]
        sender_id = msg.sender_id
        
        # Stocker les valeurs omega du voisin pour l'algorithme 3
        self.neighbors_omega[sender_id] = copy.deepcopy(sender_omega)
        
        # Appliquer les lois de mise à jour du papier (Algorithme 1)
        old_xi = copy.deepcopy(self.xi)

        # Pour tous les noeuds connus (inclure les nouveaux noeuds découverts)
        all_nodes = set(sender_xi.keys()) | set(self.xi.keys())
        
        for j in all_nodes:
            if j == self.id:
                continue
                
            # ξ[i,j](k+1) = max(ξ[l,j](k)) pour l ∈ N_i ∪ {i}
            # Prendre le max entre notre estimation actuelle et celle du voisin
            current_xi = self.xi.get(j, 0)
            sender_xi_val = sender_xi.get(j, 0)
            new_xi = max(current_xi, sender_xi_val)
            
            # Mettre à jour xi
            old_xi_val = old_xi.get(j, 0)
            self.xi[j] = new_xi
            
            # ω[i,j](k+1) selon les conditions du papier
            if new_xi > old_xi_val:
                # xi a été mis à jour (nouvelle connectivité découverte), recalculer omega
                # ω[i,j](k+1) = min_{l ∈ N_i}(ω[l,j](k) + 1)
                min_omega = float('inf')
                for neighbor_id in self.neighbors:
                    if neighbor_id in self.neighbors_omega:
                        neighbor_omega = self.neighbors_omega[neighbor_id]
                        neighbor_dist = neighbor_omega.get(j, float('inf'))
                        if neighbor_dist != float('inf'):
                            min_omega = min(min_omega, neighbor_dist + 1)

                if min_omega != float('inf'):
                    self.omega[j] = min(self.omega.get(j, float('inf')), min_omega)
            # Si xi n'a pas changé, omega reste inchangé

        # Mettre à jour l'estimation du nombre de noeuds
        self.estimate_node_count()
        
    def process_omega_update(self, msg: Message):
        """Traiter les mises à jour omega des voisins pour l'algorithme 3"""
        sender_omega = msg.data["omega"]
        sender_id = msg.sender_id
        
        # Stocker les valeurs omega du voisin
        self.neighbors_omega[sender_id] = copy.deepcopy(sender_omega)
        
    def check_connectivity_and_proceed(self):
        """Algorithme 2: Vérifier la connectivité et passer à la détection d'arêtes critiques"""
        if self.algorithm_phase == "connectivity":
            # Vérifier si tous les nœuds sont atteignables (connectivité)
            all_connected = True
            for node_id in range(self.n_estimate):
                # Un noeud est considéré comme non atteignable si xi[node_id] = 0
                if node_id != self.id and self.xi.get(node_id, 0) == 0:
                    all_connected = False
                    break
                    
            # Transition vers l'algorithme 3 après 2n étapes ou si tout est connecté
            if all_connected or self.iteration >= 2 * self.n_estimate:
                self.algorithm_phase = "critical_detection"
                # Réinitialiser le compteur d'itération pour l'algorithme 3
                self.critical_detection_iteration = 0
                
    def detect_critical_edges(self):
        """Algorithme 3: Détection des arêtes critiques selon le papier (2 étapes)"""
        if self.algorithm_phase != "critical_detection":
            return
            
        self.critical_edges.clear()
        
        # Étape 2n+1: Échanger les vecteurs omega avec tous les voisins
        if self.critical_detection_iteration == 0:
            for neighbor_id in self.neighbors:
                msg_data = {
                    "omega": copy.deepcopy(self.omega),
                    "sender_id": self.id
                }
                msg = Message(
                    sender_id=self.id,
                    receiver_id=neighbor_id,
                    msg_type=MessageType.OMEGA_UPDATE,
                    data=msg_data
                )
                self.outgoing_messages.append(msg)
                
        # Étape 2n+2: Analyser les arêtes critiques
        elif self.critical_detection_iteration >= 1:
            # Vérifier qu'on a reçu les informations omega de tous nos voisins
            if len(self.neighbors_omega) >= len(self.neighbors):
                for l in self.neighbors:
                    if self.is_critical_edge_theorem2(self.id, l):
                        edge = tuple(sorted([self.id, l]))
                        self.critical_edges.add(edge)
                        
        self.critical_detection_iteration += 1
                
    def is_critical_edge_theorem2(self, i: int, l: int) -> bool:
        """
        Vérifier si une arête est critique selon le Théorème 2 du papier
        
        Une arête e_il est critique ssi pour chaque nœud j et pour tous les voisins
        i' de i et l' de l, la condition suivante est remplie:
        Δ[i,j]^(il) ≠ 0 ET {Δ[i,j]^(ii'), Δ[l,j]^(ll')} ≠ {1,1}
        
        Ceci est la négation du Lemme 1 (conditions de non-criticité)
        """
        if l not in self.neighbors:
            return False
            
        # Vérifier que nous avons les informations omega du voisin l
        if l not in self.neighbors_omega:
            return False
            
        omega_l = self.neighbors_omega[l]
        
        # Pour tous les nœuds j dans le réseau
        all_nodes = set(self.omega.keys()) | set(omega_l.keys())
        
        for j in all_nodes:
            if j == i or j == l:
                continue
                
            # Calculer Δ[i,j]^(il) = ω[i,j](n) - ω[l,j](n)
            omega_i_j = self.omega.get(j, float('inf'))
            omega_l_j = omega_l.get(j, float('inf'))

            # Ignorer les nœuds non atteignables
            if omega_i_j == float('inf') or omega_l_j == float('inf'):
                continue
                
            delta_i_j_il = omega_i_j - omega_l_j
            
            # Condition 1: Δ[i,j]^(il) ≠ 0
            # Si Δ[i,j]^(il) = 0, alors j est équidistant de i et l (Lemme 1, condition 1)
            # Cela indique un chemin alternatif, donc l'arête n'est PAS critique
            if delta_i_j_il == 0:
                return False
                
            # Condition 2: Vérifier qu'il n'existe pas de voisins i' et l' tels que
            # {Δ[i,j]^(ii'), Δ[l,j]^(ll')} = {1,1}
            # (c'est la condition 2 du Lemme 1 pour les chemins alternatifs)
            
            # Pour tous les voisins i' de i
            for i_prime in self.neighbors:
                if i_prime == l or i_prime not in self.neighbors_omega:
                    continue
                    
                omega_i_prime = self.neighbors_omega[i_prime]
                omega_i_prime_j = omega_i_prime.get(j, float('inf'))

                if omega_i_prime_j == float('inf'):
                    continue
                    
                # Calculer Δ[i,j]^(ii')
                delta_i_j_ii_prime = omega_i_j - omega_i_prime_j
                
                # Maintenant on doit vérifier les voisins de l
                # En environnement distribué, on approxime en utilisant nos voisins communs
                # ou les informations disponibles sur l
                for l_prime in self.neighbors:
                    if l_prime == i or l_prime not in self.neighbors_omega:
                        continue
                        
                    # Vérifier si l_prime pourrait être un voisin de l
                    # Cette partie est une approximation car nous n'avons pas
                    # directement accès à la liste des voisins de l
                    omega_l_prime = self.neighbors_omega[l_prime]
                    omega_l_prime_j = omega_l_prime.get(j, float('inf'))

                    if omega_l_prime_j == float('inf'):
                        continue
                        
                    # Calculer Δ[l,j]^(ll') en assumant que l_prime est voisin de l
                    delta_l_j_ll_prime = omega_l_j - omega_l_prime_j
                    
                    # Si {Δ[i,j]^(ii'), Δ[l,j]^(ll')} = {1,1}, alors il existe
                    # un chemin alternatif (Lemme 1, condition 2)
                    if delta_i_j_ii_prime == 1 and delta_l_j_ll_prime == 1:
                        return False
                                        
        # Si toutes les conditions sont satisfaites pour tous les j,
        # l'arête est critique selon le Théorème 2
        return True
        
    def step(self):
        """Exécuter une étape de l'algorithme distribué"""
        # Traiter les messages entrants
        for msg in self.incoming_messages:
            if msg.msg_type == MessageType.XI_UPDATE:
                self.process_xi_update(msg)
            elif msg.msg_type == MessageType.OMEGA_UPDATE:
                self.process_omega_update(msg)
            elif msg.msg_type == MessageType.CONNECTION_ADD:
                # Ajouter une connexion à un nouveau drone
                new_drone_id = msg.data["sender_id"]
                if self.can_connect(new_drone_id):
                    self.add_connection(new_drone_id)
                
        self.incoming_messages.clear()
        
        # Exécuter l'algorithme approprié selon la phase
        if self.algorithm_phase == "neighbor_structure":
            self.update_neighbor_structure()
        elif self.algorithm_phase == "connectivity":
            self.check_connectivity_and_proceed()
        elif self.algorithm_phase == "critical_detection":
            self.detect_critical_edges()
            
        self.iteration += 1
        
    def get_messages_to_send(self) -> List[Message]:
        """Obtenir les messages à envoyer et vider la queue"""
        messages = self.outgoing_messages.copy()
        self.outgoing_messages.clear()
        return messages
        
    def receive_message(self, msg: Message):
        """Recevoir un message"""
        self.incoming_messages.append(msg)
        
    def get_known_nodes(self) -> Set[int]:
        """Obtenir tous les nœuds dont ce drone a connaissance"""
        known_nodes = {self.id}
        known_nodes.update(self.neighbors)
        known_nodes.update(self.xi.keys())
        known_nodes.update(self.omega.keys())
        return known_nodes
        
    def get_detailed_info(self) -> dict:
        """Obtenir les informations détaillées du drone"""
        return {
            'id': self.id,
            'position': (self.x, self.y),
            'neighbors': list(self.neighbors),
            'phase': self.algorithm_phase,
            'iteration': self.iteration,
            'n_estimate': self.n_estimate,
            'xi': dict(self.xi),
            'omega': {k: v if v != float('inf') else '∞' for k, v in self.omega.items()},
            'critical_edges': list(self.critical_edges),
            'known_nodes': list(self.get_known_nodes())
        }










    def __str__(self):
        return f"Drone(pos={self.position}, connections={len(self.connections)}, iteration={self.iteration})"
    
    def __repr__(self):
        return self.__str__()

    # misc

    def __del__(self):
        """Clean up the drone."""
        for connection in self.connections:
            connection.remove_connection(self)
        self.connections.clear()
        self.target = None
        self.scan_for_drones = None
        self.scan_for_targets = None
        print(f"Drone at {self.position} deleted.")
