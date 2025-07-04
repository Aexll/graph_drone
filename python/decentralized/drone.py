import numpy as np
import dearpygui.dearpygui as dpg
from target import Target
import copy
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional


class Drone:
    def __init__(self, position:np.ndarray, target:Target|None, scan_for_drones_method=None, scan_for_targets_method=None):
        
        self.position:np.ndarray = position.astype(np.float32)
        self.target:Target|None = target
        
        self.drone_radius:float = 10.0  # Radius of the drone for drawing purposes

        self.speed:float = 5.0
        self.range:float = 200.0

        # Drone memory 
        self.local_time:float = 0.0


        self.connections: list['Drone'] = []  # List of connected drones

        # Variables locales pour l'algorithme xi-omega (inspired by example 3)
        self.xi = defaultdict(float)  # xi[drone] = valeur de connectivité vers drone
        self.omega = defaultdict(lambda: float('inf'))  # omega[drone] = distance vers drone
        
        # Initialisation : le drone se connaît lui-même
        self.xi[self] = 1.0
        self.omega[self] = 0.0
        
        # Messages à envoyer et reçus
        self.outgoing_messages = {}
        self.incoming_messages = {}
        
        # Historique pour détecter la convergence
        self.previous_xi = {}
        self.previous_omega = {}
        
        # Métadonnées
        self.iteration = 0
        self.converged = False
        self.known_drones = {self}  # Ensemble des drones découverts

        # world query methods
        self.scan_for_drones: callable = scan_for_drones_method 
        self.scan_for_targets: callable = scan_for_targets_method

        if self.scan_for_drones is None:
            print("Warning: No scan_for_drones method provided, connections will not be established.")
            self.scan_for_drones = lambda position, range: []  # Default to empty list if not provided
        if self.scan_for_targets is None:
            print("Warning: No scan_for_targets method provided, targets will not be found.")
            self.scan_for_targets = lambda position, range: []  # Default to empty list if not provided

        # visual
        self.drone_radius = 10
        self.target_radius = 10
        self.connection_radius = 10

        # Debug
        self.total_operations:int = 0 # Number of operations performed by this drone, for debugging purposes
        self.frame_operations:int = 0 # Number of operations performed in the current frame, for debugging purposes
        self.frame_count:int = 0 # Number of frames since the last reset, for debugging purposes
        

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
        self.xi_omega_step()

        # Connection management
        unknown_drones = self.scan_for_unknown_drones()
        for drone in unknown_drones:
            if self.can_connect(drone):
                self.add_connection(drone)
                break

        # Movement
        if self.target is not None:
            self.move_towards_target(delta_time)

        # Debugging 
        if True:
            if self.frame_operations > 1000:
                print(f"Drone at {self.position} performed {self.frame_operations} operations in this frame, Average: {self.total_operations / self.frame_count if self.frame_count > 0 else 0}.")
            self.total_operations += self.frame_operations
            self.frame_count += 1
            self.frame_operations = 0  # Reset frame operations for the next frame



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

    def can_connect(self, drone: 'Drone') -> bool:
        """
        Check if this drone can add a connection to another drone.
        Returns True if the connection can be added, False otherwise.

        """
        if drone in self.connections:
            return False
        
        # at most 3 connections
        if len(self.connections) >= 3:
            return False
        
        # must be in range
        if np.linalg.norm(drone.position - self.position) > self.range:
            return False

        return True

    def can_deconnect(self, drone: 'Drone') -> bool:
        """
        Check if this drone can remove a connection to another drone.
        Not depending on the other drone's state.
        """
        if drone not in self.connections:
            return False
        
        # must not be the only connection
        if len(self.connections) <= 1:
            return False
        
        return True

    def add_connection(self, drone: 'Drone'):
        """ Add a connection to another drone. must be in range """
        # Ajouter la connexion avant de calculer xi pour éviter les incohérences
        if drone not in self.connections:
            self.connections.append(drone)
        if self not in drone.connections:
            drone.connections.append(self)
        
    def remove_connection(self, drone: 'Drone'):
        """Remove a connection to another drone."""
        if drone in self.connections:
            self.connections.remove(drone)
        if self in drone.connections:
            drone.connections.remove(self)

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

    def destroy(self):
        """Clean up the drone's resources."""
        for connection in self.connections:
            connection.remove_connection(self)
        self.connections.clear()
        self.known_drones.clear()
        self.incoming_messages.clear()

    # Xi-Omega Algorithm Implementation (based on examples 3 and 4)

    def xi_omega_step(self):
        """Execute one step of the distributed xi-omega algorithm"""
        # Phase 1: Prepare messages
        messages = self.prepare_messages()
        
        # Phase 2: Send messages to neighbors (simulated)
        for neighbor, message in messages.items():
            neighbor.receive_message(self, message)
        
        # Phase 3: Update xi and omega values
        self.update_xi_omega()
        
        # Debug: print current state
        # print(f"Drone {id(self)} step {self.iteration}: knows {len(self.known_drones)} drones")

    def prepare_messages(self) -> Dict['Drone', Dict]:
        """
        Prépare les messages à envoyer aux voisins
        
        Returns:
            Dictionnaire {neighbor_drone: message}
        """
        messages = {}
        
        # Créer un message contenant l'état xi et omega du drone
        # Utiliser les IDs au lieu des objets pour éviter la récursion infinie
        xi_serializable = {id(drone): value for drone, value in self.xi.items()}
        omega_serializable = {id(drone): value for drone, value in self.omega.items()}
        known_drones_ids = {id(drone) for drone in self.known_drones}
        
        message = {
            'sender_id': id(self),
            'iteration': self.iteration,
            'xi': xi_serializable,
            'omega': omega_serializable,
            'known_drones_ids': known_drones_ids,
            'timestamp': time.time()
        }
        
        # Envoyer le même message à tous les voisins
        for neighbor in self.connections:
            messages[neighbor] = message.copy()  # Copie simple au lieu de deepcopy
        
        return messages

    def receive_message(self, sender: 'Drone', message: Dict):
        """
        Reçoit un message d'un voisin
        
        Args:
            sender: Drone expéditeur
            message: Contenu du message
        """
        if sender not in self.incoming_messages:
            self.incoming_messages[sender] = []
        
        self.incoming_messages[sender].append(message)

    def update_xi_omega(self):
        """
        Met à jour les valeurs xi et omega selon l'algorithme distribué
        Implémentation corrigée basée sur exemple_3.py
        """
        # Sauvegarder l'état précédent
        self.previous_xi = dict(self.xi)
        self.previous_omega = dict(self.omega)
        
        # Découvrir de nouveaux drones à partir des messages
        all_known_drones = set(self.known_drones)
        
        # Ajouter directement les expéditeurs des messages comme nouveaux drones connus
        for sender in self.incoming_messages.keys():
            all_known_drones.add(sender)
        
        # Créer un mapping ID->drone pour les drones que nous connaissons déjà
        drone_id_to_object = {id(drone): drone for drone in all_known_drones}
        
        # Découvrir d'autres drones mentionnés dans les messages
        for sender, messages in self.incoming_messages.items():
            if messages:  # Prendre le message le plus récent
                latest_message = messages[-1]
                known_drone_ids = latest_message.get('known_drones_ids', set())
                
                # Découvrir de nouveaux drones via les IDs dans les messages
                # Ici, on doit demander au monde ou au système de résoudre les IDs
                # Pour l'instant, on va faire une approche simplifiée :
                # Chercher dans tous les drones connectés si on peut résoudre ces IDs
                
                # Méthode de découverte étendue via les connexions
                for connected_drone in self.connections:
                    if connected_drone in self.incoming_messages:
                        for other_drone in connected_drone.known_drones:
                            if other_drone not in all_known_drones:
                                all_known_drones.add(other_drone)
                                drone_id_to_object[id(other_drone)] = other_drone
        
        self.known_drones = all_known_drones
        
        # Étape 1: Calculer xi pour tous les drones connus
        new_xi = defaultdict(float)
        new_omega = defaultdict(lambda: float('inf'))
        
        # Initialiser avec les valeurs actuelles
        for drone in self.known_drones:
            new_xi[drone] = self.xi.get(drone, 0.0)
            new_omega[drone] = self.omega.get(drone, float('inf'))
        
        # Assurer que self reste à 1.0 et 0.0
        new_xi[self] = 1.0
        new_omega[self] = 0.0
        
        # Pour chaque drone target
        for target_drone in self.known_drones:
            if target_drone == self:
                continue
                
            target_id = id(target_drone)
            
            # Équation (1): xi_i,j(k+1) = max{xi_l,j(k) : l ∈ N_i ∪ {i}}
            candidates = [self.xi.get(target_drone, 0.0)]  # xi_i,j(k)
            
            # Ajouter les valeurs des voisins
            for neighbor in self.connections:
                if neighbor in self.incoming_messages:
                    messages = self.incoming_messages[neighbor]
                    if messages:
                        latest_message = messages[-1]
                        neighbor_xi = latest_message.get('xi', {})
                        if target_id in neighbor_xi:
                            candidates.append(neighbor_xi[target_id])
            
            # Si le target est un voisin direct, ajouter sa connectivité
            if target_drone in self.connections:
                candidates.append(1.0)
            
            new_xi[target_drone] = max(candidates)
        
        # Étape 2: Calculer omega pour tous les drones connus
        for target_drone in self.known_drones:
            if target_drone == self:
                continue
                
            target_id = id(target_drone)
            
            # Si xi a changé, recalculer omega
            if new_xi[target_drone] != self.xi.get(target_drone, 0.0):
                # Équation (2): Si xi augmente, omega = min{omega_l,j(k) + 1 : l ∈ N_i, xi_l,j(k) = xi_i,j(k+1)}
                min_distances = []
                
                # Si le target est un voisin direct et xi=1, alors omega=1
                if target_drone in self.connections and new_xi[target_drone] == 1.0:
                    min_distances.append(1.0)
                
                # Chercher parmi les voisins
                for neighbor in self.connections:
                    if neighbor in self.incoming_messages:
                        messages = self.incoming_messages[neighbor]
                        if messages:
                            latest_message = messages[-1]
                            neighbor_xi = latest_message.get('xi', {})
                            neighbor_omega = latest_message.get('omega', {})
                            
                            # Si le voisin a la même valeur xi que celle qu'on vient de calculer
                            if (target_id in neighbor_xi and 
                                abs(neighbor_xi[target_id] - new_xi[target_drone]) < 1e-10):
                                if (target_id in neighbor_omega and 
                                    neighbor_omega[target_id] != float('inf')):
                                    min_distances.append(neighbor_omega[target_id] + 1)
                
                if min_distances:
                    new_omega[target_drone] = min(min_distances)
                else:
                    new_omega[target_drone] = float('inf')
            else:
                # xi n'a pas changé, garder omega
                new_omega[target_drone] = self.omega.get(target_drone, float('inf'))
        
        # Mettre à jour les valeurs
        self.xi = new_xi
        self.omega = new_omega
        
        # Nettoyer les messages traités
        self.incoming_messages.clear()
        self.iteration += 1

    def has_converged(self, tolerance: float = 1e-10) -> bool:
        """
        Vérifie si le drone a convergé (pas de changement dans xi et omega)
        
        Args:
            tolerance: Tolérance pour la convergence
            
        Returns:
            True si convergé, False sinon
        """
        if not self.previous_xi or not self.previous_omega:
            return False
        
        # Vérifier si tous les drones connus ont des valeurs stables
        for drone in self.known_drones:
            if drone == self:
                continue
            
            # Vérifier xi
            prev_xi = self.previous_xi.get(drone, 0)
            curr_xi = self.xi.get(drone, 0)
            if abs(curr_xi - prev_xi) > tolerance:
                return False
            
            # Vérifier omega
            prev_omega = self.previous_omega.get(drone, float('inf'))
            curr_omega = self.omega.get(drone, float('inf'))
            if abs(curr_omega - prev_omega) > tolerance:
                return False
        
        return True

    def get_neighbor_structure(self) -> Dict[int, List['Drone']]:
        """
        Retourne la structure de voisinage locale
        
        Returns:
            Dictionnaire {distance: [liste_des_drones]}
        """
        structure = defaultdict(list)
        
        for drone in self.known_drones:
            if drone != self:
                distance = self.omega.get(drone, float('inf'))
                if distance != float('inf'):
                    structure[int(distance)].append(drone)
        
        return dict(structure)

    def print_local_state(self):
        """Affiche l'état local du drone"""
        print(f"\n--- État du drone {id(self)} ---")
        print(f"Position: {self.position}")
        print(f"Voisins: {[id(d) for d in self.connections]}")
        print(f"Drones connus: {[id(d) for d in self.known_drones]}")
        print(f"Itération: {self.iteration}")
        
        xi_display = {id(drone): value for drone, value in self.xi.items()}
        print(f"Xi: {xi_display}")
        
        omega_display = {}
        for drone, dist in self.omega.items():
            omega_display[id(drone)] = dist if dist != float('inf') else 'inf'
        print(f"Omega: {omega_display}")
        
        print(f"Structure de voisinage: {self.get_neighbor_structure()}")
        print(f"Convergé: {self.has_converged()}")

    def get_display_color(self) -> List[int]:
        """Retourne la couleur d'affichage du drone (inspiré de l'exemple 4)"""
        if self.has_converged():
            return [100, 255, 100]  # Vert si convergé
        elif len(self.incoming_messages) > 0:
            return [255, 100, 100]  # Rouge si actif (traite des messages)
        else:
            return [100, 150, 255]  # Bleu par défaut
   




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
