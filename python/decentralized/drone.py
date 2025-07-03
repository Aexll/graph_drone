import numpy as np
import dearpygui.dearpygui as dpg
from target import Target


class Drone:
    def __init__(self, position:np.ndarray, target:Target|None, scan_for_drones_method=None, scan_for_targets_method=None):
        self.position :np.ndarray = position.astype(np.float32)
        self.target:Target|None = target

        self.speed:float = 5.0
        self.range:float = 200.0

        # Debug
        self.total_operations:int = 0 # Number of operations performed by this drone, for debugging purposes
        self.frame_operations:int = 0 # Number of operations performed in the current frame, for debugging purposes
        self.frame_count:int = 0 # Number of frames since the last reset, for debugging purposes

        # Drone memory 
        self.local_time:float = 0.0

        self.count:int = 1 # Number of drone known to be connected to this drone, including itself  

        self.connections:list['Drone'] = []
        self.targets = None

        self.xi:set = {self}
        self.xi_n:int = 0  # Number of xi values computed, -1 means not computed yet
        self.xi_time:float = 0.0
        self.xi_deprecation_time:float = 100.0  # Time after which the xi value is considered deprecated (need to be recomputed)

        self.omega:dict = {self: 0}
        self.omega_n:int = 0 
        self.omega_time:float = 0.0
        self.omega_deprecation_time:float = 5.0  # Time after which the omega value is considered deprecated (need to be recomputed)    

        self.delta:dict = {}  # Delta values for other drones
        self.delta_n:dict = {}  # Number of delta values computed, -1 means not computed yet
        self.delta_time:dict = {}  # Time step for updates, can be adjusted as needed
        self.delta_deprecation_time:float = 5.0  # Time after which the delta value is considered deprecated (need to be recomputed)

        self.criticality:dict = {}
        self.criticality_time:dict = {}  # Time step for updates, can be adjusted as needed
        self.criticality_deprecation_time:float = 5.0  # Time after which the criticality value is considered deprecated (need to be recomputed)

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
        

    def draw(self, draw_list):

        # draw the drone
        dpg.draw_circle((self.position[0], self.position[1]), 10, color=(255, 0, 0), parent=draw_list)

        # range circle
        dpg.draw_circle((self.position[0], self.position[1]), self.range, color=(255, 0, 0, 100), parent=draw_list, fill=True)

        # connections
        for connection in self.connections:
            direction = (connection.position - self.position) / np.linalg.norm(connection.position - self.position)
            link_color = (0, 255, 0, 100)
            if connection in self.criticality:
                link_color = (255, 0, 0, 255) 

            dpg.draw_line(
                self.position + direction * self.drone_radius,
                connection.position - direction * self.drone_radius,
                color=link_color, parent=draw_list)
    
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


    def update(self, delta_time=0.08):
        """Update the drone's state."""

        self.local_time += delta_time

        self.compute_criticality()

        unknown_drones = self.scan_for_unknown_drones(8)
        for drone in unknown_drones:
            if self.accept_connection(drone):
                # new_count = drone.count + self.count
                self.add_connection(drone)
                # drone.update_drone_count(new_count, n=new_count)



        if len(self.connections) < 1:
            self.scan_for_new_connections(nb_connections=1)
        if self.target is not None:
            self.move_towards_target(delta_time)
        else:
            # If no target, do nothing or hover in place
            pass
        pass

        self.total_operations += self.frame_operations
        # print(f"Drone at {self.position} performed {self.frame_operations} operations in this frame, Average: {self.total_operations / self.frame_count if self.frame_count > 0 else 0}.")
        self.frame_count += 1
        self.frame_operations = 0  # Reset frame operations for the next frame

    # connection management



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

    def add_connection(self, drone: 'Drone'):
        """Add a connection to another drone. must be in range"""
        if np.linalg.norm(drone.position - self.position) > self.range:
            return
        if drone not in self.compute_xi(n=self.count):
            self.count += drone.count
            self.update_drone_count(self.count, n=self.count)
            drone.update_drone_count(self.count, n=self.count)

        if drone not in self.connections:
            self.connections.append(drone)
        # Ensure the connection is mutual
        if self not in drone.connections:
            drone.connections.append(self)
        
    def remove_connection(self, drone: 'Drone'):
        """Remove a connection to another drone."""
        if drone in self.connections:
            self.connections.remove(drone)
        if self in drone.connections:
            drone.connections.remove(self)
    
    def accept_connection(self, drone: 'Drone') -> bool:
        """
        Check if this drone can accept a connection from another drone.
        """
        res = True

        if drone in self.connections:
            res = False
        
        # at most 3 connections
        if len(self.connections) >= 3:
            res = False

        return res

    def scan_for_new_connections(self, nb_connections=1):
        """
        Scan for new connections within range.
        nb_connections: Number of connections to establish at most.
        """
        candidates = self.scan_for_drones(self.position, self.range)
        candidates = [drone for drone in candidates if drone != self and drone not in self.connections]
        candidates = sorted(candidates, key=lambda d: np.linalg.norm(d.position - self.position))
        for drone in candidates[:nb_connections]:
            self.add_connection(drone)
    
    def scan_for_unknown_drones(self, n=0) -> set:
        """
        Scan for unknown drones within range.
        Returns a set of drones that are not in the connections list.
        """
        candidates = self.scan_for_drones(self.position, self.range)
        unknown_drones = set(drone for drone in candidates if drone not in self.compute_xi(n=n) and drone != self)
        return unknown_drones

    def compute_xi(self, n=0, force_update=False) -> set:
        """
        Compute the xi value for this drone.
        xi is a dictionary of the form {drone: value:bool}.
        """
        self.frame_operations += 1
        if force_update:
            self.xi_n = 0
        if self.xi_time + self.xi_deprecation_time < self.local_time:
            self.xi_n = 0
        if self.xi_n >= n:
            return self.xi
        if n <= 0:
            self.xi = {self}
            self.xi_n = 0
            self.xi_time = self.local_time
            return self.xi
        if n > 0:
            self.xi = set()
            for drone in self.connections:
                self.xi.update(drone.compute_xi(n-1 if not force_update else 2*n, force_update=False))
            self.xi.add(self)
            self.xi_n = n
            self.xi_time = self.local_time
            return self.xi.copy()


    def compute_omega(self, n=0) -> dict:
        """
        Compute the omega value for this drone.
        omega is a dictionary of the form {drone: value:int}.
        """
        self.frame_operations += 1
        if self.omega_time + self.omega_deprecation_time < self.local_time:
            self.omega_n = 0
            self.omega = {self: 0}
            self.omega_time = self.local_time
        if self.omega_n >= n:
            return self.omega.copy()
        if n <= 0:
            return self.omega.copy()    
        if n > 0:
            new_omega = {self: 0}
            for drone in self.connections:
                d_omega = drone.compute_omega(n - 1)
                for key, value in d_omega.items():
                    if key in new_omega:
                        new_omega[key] = min(new_omega[key], value + 1)
                    else:
                        new_omega[key] = value + 1
            self.omega = new_omega
            self.omega_n = n
            self.omega_time = self.local_time
            return self.omega.copy()


    def compute_delta(self, drone: 'Drone', n:int=0) -> dict:
        """
        Compute the delta value for this drone.
        delta is a dictionary of the form {drone: value:int}.
        """
        if drone in self.delta and self.delta_time.get(drone, 0) + self.delta_deprecation_time > self.local_time:
            return self.delta[drone].copy()
        if drone not in self.connections:
            print(f"Drone {self} cannot compute delta for {drone}, not connected.")
            return {}
        delta = self.compute_omega(n=self.count*2).copy()
        for key, value in drone.compute_omega(n=self.count*2).items():
            if key in delta:
                delta[key] = delta[key] - value
            else:
                print(f"Warning: Drone {key} not in delta, adding with value {value}.")
                delta[key] = value

        self.delta[drone] = delta
        self.delta_n[drone] = n
        self.delta_time[drone] = self.local_time
        return delta

    def compute_criticality(self,n:int=0) -> dict:
        """
        Compute the criticality value for this drone.
        Criticality is a measure of how important this drone is in the network.
        Returns a dictionary of the form {drone: value:bool}.
        """
        criticality = {}
        for connection in self.connections:
            criticality[connection] = self.compute_criticality_drone(connection, n=self.count)
        return criticality

    def compute_criticality_drone(self, drone: 'Drone', n:int=0) -> bool:
        """
        Compute the criticality value for this drone.
        Criticality is a measure of how important this drone is in the network.
        """
        # i_drones = self.connections.copy().remove(drone)
        # j_drones = drone.connections.copy().remove(self)

        if drone in self.criticality and self.criticality_time.get(drone, 0) + self.criticality_deprecation_time > self.local_time:
            return self.criticality[drone]
        if drone not in self.connections:
            print(f"Drone {self} cannot compute criticality for {drone}, not connected.")
            return 0.0
        d1 = self.compute_delta(drone, n=n)
        d2 = drone.compute_delta(self, n=n)
        d1 = np.array(list(d1.values()))
        d2 = np.array(list(d2.values()))
        ret = 0 in d1 or 0 in d2

        self.criticality[drone] = ret
        self.criticality_time[drone] = self.local_time
        # print(d1,d2, ret)
        return ret


    def update_drone_count(self, new_count:int, n=0):
        """
        """
        if n<=0:
            self.count = new_count
        else:
            self.count = new_count
            for connection in self.connections:
                connection.update_drone_count(new_count, n=n-1)

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
