import numpy as np
import dearpygui.dearpygui as dpg
from target import Target


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

        self.xi_omega:list[dict['Drone',tuple[int, int]]] = []  # Dictionary to store xi and omega values for each drone, as a mapping of drone to (xi, omega) tuple

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

        # draw the drone
        dpg.draw_circle((self.position[0], self.position[1]), self.drone_radius, color=(255, 0, 0), parent=draw_list)

        # range circle
        dpg.draw_circle((self.position[0], self.position[1]), self.range, color=(255, 0, 0, 100), parent=draw_list, fill=True)

        # connections
        for connection in self.connections:
            distance = np.linalg.norm(connection.position - self.position)
            if distance > 0:  # Éviter la division par zéro
                line = (connection.position - self.position)
                direction = line / distance

                link_color = (0, 255, 0, 255)
                if connection in self.criticality:
                    link_color = (255, 0, 0, 255) 

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

        # self.compute_criticality()

        unknown_drones = self.scan_for_unknown_drones()
        for drone in unknown_drones:
            if self.can_add_connection(drone):
                self.add_connection(drone)
                break

        # if len(self.connections) < 1:
        #     self.scan_for_new_connections(nb_connections=1)

        if self.target is not None:
            self.move_towards_target(delta_time)
        else:
            # If no target, do nothing or hover in place
            pass
        pass



        # debugging 
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
        Returns a set of drones that are not in the connections list.
        """
        candidates = self.scan_for_drones(self.position, self.range)
        
        candidates = [drone for drone in candidates if drone != self and drone not in self.connections]
        if len(candidates) == 0:
            return set()
        
        unknown_drones = set(drone for drone in candidates if drone not in self.compute_xi(n=5) and drone != self)

        return unknown_drones


    # xi and omega computation

    def compute_xi_omega(self, max_iter=None) -> dict:

        if max_iter is None:
            max_iter = 10

        xi_omega_history = self.xi_omega.copy()  # Copy current xi_omega state

        for k in range(max_iter):
            xi_new:list[dict['Drone', int]] = {}

        return self.xi_omega
   




    def __str__(self):
        return f"Drone(pos={self.position}, connections={len(self.connections)}, count={self.count})"
    
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
        self.targets = None
        print(f"Drone at {self.position} deleted.")
