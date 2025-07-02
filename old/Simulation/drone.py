import numpy as np
import dearpygui.dearpygui as dpg
import algograph as ag
from message import Message
import random

DRONE_RADIUS = 10
DRONE_SPEED = 200
CONEXION_RADIUS = 400
TARGET_RADIUS = 5
FREE_TICKS = 2 # Every N ticks, the drones will update they omega and xi values
INF = 1000000000


OMEGA_UPDATE_TICK = 10



"""
Note: when a variable start with an underscore it means that the drone is not aware of it, 
it is just for drawing purposes, the drone will not use it to make decisions.
"""

class Drone:
    DRONE_COUNT = 0
    DRONE_REFERENCE = []
    GRAPH = None

    @staticmethod
    def get_graph():
        """
        Returns the graph as matrix using numpy, the matrix is a square matrix of size Drone.DRONE_COUNT
        its values at (i,j) are 1 if the drone i is connected to drone j, 0 otherwise
        """

        # if the graph is already computed, return it
        if Drone.GRAPH is not None:
            return Drone.GRAPH
        
        positions = np.array([drone.position for drone in Drone.DRONE_REFERENCE])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # we exclude self-connections
        np.fill_diagonal(dists, np.inf)
        # we set the value to 1 if the distance is less than CONEXION_RADIUS, 0 otherwise
        ret = (dists < CONEXION_RADIUS).astype(float)
        Drone.GRAPH = ret
        return ret

    @staticmethod
    def omega_zero(id):
        ret = np.ones(Drone.DRONE_COUNT) * INF
        ret[id] = 0
        return ret


    def __init__(self, position, target):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self._neighbors = None
        self.id = Drone.DRONE_COUNT
        self.ticks = 0 # internal ticks counter

        # Omega is a list of every drone distances to the drone
        self.stored_omega = np.ones(Drone.DRONE_COUNT) * INF
        self.stored_omega_range = 0

        Drone.DRONE_COUNT += 1
        Drone.DRONE_REFERENCE.append(self)

        self.sent_messages = []

    def retarget(self):
        """
        Retarget the drone to a random target
        """
        self.target = np.random.rand(2)*800  # Random target within a 800x600 area



    ## Message handling

    def receive_message(self, message: Message):
        """
        Receive a message from another drone
        """
        pass
        
    def send_message(self, msg: Message):
        """
        Send a message to another drone
        """
        self.sent_messages.append(msg)

    def receive_omega(self, new_omega: np.ndarray, n: int):
        """
        Receive an omega message from another drone
        we update our stored_omega with the message
        """
        # Initialize stored_omega if it is not initialized
        if self.stored_omega.size != Drone.DRONE_COUNT:
            self.stored_omega = Drone.omega_zero(self.id)
            self.stored_omega_range = 0

        self.stored_omega = np.minimum(self.stored_omega, new_omega+1)
        self.stored_omega_range = max(self.stored_omega_range, n+1)

        if self.stored_omega_range < Drone.DRONE_COUNT:
            for d in self.get_neighbors_ids():
                self.send_omega(d)



    def send_omega(self,target):
        """
        Send an omega message to another drone
        """
        Drone.DRONE_REFERENCE[target].receive_omega(self.stored_omega, self.stored_omega_range)



    def think(self):

        # Initialize stored_omega if it is not initialized
        if self.stored_omega.size == 0:
            self.stored_omega = Drone.omega_zero(self.id)
            self.stored_omega_range = 0

        # Send omega to neighbors every OMEGA_UPDATE_TICK ticks
        if self.ticks % OMEGA_UPDATE_TICK == 0:
            for neighbor in self.get_neighbors_ids():
                self.send_omega(neighbor)
        self.ticks += 1

    def get_neighbors_ids(self) -> np.ndarray:
        """
        Returns the list of neighbors of the drone
        (using get_graph)
        """
        return np.where(Drone.get_graph()[self.id] == 1)[0]

    def get_neighbors_references(self) -> list:
        """
        Returns the list of neighbors of the drone
        (using get_graph)
        """
        return [Drone.DRONE_REFERENCE[i] for i in self.get_neighbors_ids()]


    def update(self, dt):
        """
        Update the drone position
        dt is the time step
        """

        # Send a message to a random drone
        # if random.random() < 0.001:
        #     to_id = random.randint(0, Drone.DRONE_COUNT-1)
        #     if to_id != self.id and to_id in self.get_neighbors_ids():
        #         self.send_message(Message(self.id, to_id, **{"test": "test"}))

        # update messages
        for message in self.sent_messages:
            to_pos = Drone.DRONE_REFERENCE[message.to_id].position
            from_pos = self.position
            if np.linalg.norm(to_pos - from_pos) > 0:
                message.progress += dt * (Message.SPEED / np.linalg.norm(to_pos - from_pos))
            else:
                message.progress = 1
            if message.progress >= 1:
                Drone.DRONE_REFERENCE[message.to_id].receive_message(message)
                self.sent_messages.remove(message)

        # Move the drone towards the target
        dir = np.linalg.norm(self.target - self.position)
        if dir > 0:
            dir = (self.target - self.position) / dir
        else:
            dir = np.zeros_like(self.position)
        
        # Don't move if the drone is already at the target
        if np.linalg.norm(self.target - self.position) < TARGET_RADIUS:
            dir = np.zeros_like(dir)
        self.position += dir * DRONE_SPEED * dt




        Drone.GRAPH = None

    def draw(self):
        # Draw connections between neighbors
        for neighbor in self.get_neighbors_references():
            if self.id > neighbor.id:
                continue
            direction = neighbor.position - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.zeros_like(direction)
            
            line_start = self.position + direction * DRONE_RADIUS
            line_end = neighbor.position - direction * DRONE_RADIUS
            
            if ag.is_critical_edge(self.id, neighbor.id):
                color = [255, 0, 0, 255]
            else:
                color = [255, 255, 255, 255]

            dpg.draw_line(
                line_start.tolist(), 
                line_end.tolist(), 
                color=color, 
                thickness=2,
                parent="simulation_drawing"
            )
        
    
        # Draw drone as circle  
        dpg.draw_circle(
            list(map(float, self.position)), 
            DRONE_RADIUS, 
            color=[0, 0, 255, 255], 
            fill=[0, 0, 255, 255],
            parent="simulation_drawing"
        )
        
        # Draw target as cross
        target_pos = self.target
        dpg.draw_line(
            [target_pos[0] - 5, target_pos[1] - 5], 
            [target_pos[0] + 5, target_pos[1] + 5], 
            color=[0, 255, 0, 255], 
            thickness=2,
            parent="simulation_drawing"
        )
        dpg.draw_line(
            [target_pos[0] - 5, target_pos[1] + 5], 
            [target_pos[0] + 5, target_pos[1] - 5], 
            color=[0, 255, 0, 255], 
            thickness=2,
            parent="simulation_drawing"
        )
        
        # Draw drone ID
        dpg.draw_text(
            [self.position[0] - 5, self.position[1] - 5], 
            str(self.id), 
            color=[255, 255, 255, 255], 
            size=16,
            parent="simulation_drawing"
        )

        for message in self.sent_messages:
            pos = self.position + (Drone.DRONE_REFERENCE[message.to_id].position - self.position) * message.progress
            message.draw(pos)


        # Draw stored_omega next to the drone
        dpg.draw_text(
            [self.position[0] - 20, self.position[1] - 20], 
            str(self.stored_omega), 
            color=[255, 255, 255, 255], 
            size=16,
            parent="simulation_drawing"
        )
