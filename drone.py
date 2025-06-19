import numpy as np
import dearpygui.dearpygui as dpg
import algograph as ag


DRONE_RADIUS = 10
DRONE_SPEED = 200
CONEXION_RADIUS = 400
TARGET_RADIUS = 5
FREE_TICKS = 2 # Every N ticks, the drones will update they omega and xi values





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
        
        # if the graph is not computed, compute it        
        positions = np.array([drone.position for drone in Drone.DRONE_REFERENCE])
        # we compute the distance between all drones using numpy for efficiency
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # we exclude self-connections
        np.fill_diagonal(dists, np.inf)
        # we set the value to 1 if the distance is less than CONEXION_RADIUS, 0 otherwise
        ret = (dists < CONEXION_RADIUS).astype(float)
        Drone.GRAPH = ret
        return ret




    def __init__(self, position, target):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self._neighbors = None
        self.id = Drone.DRONE_COUNT
        Drone.DRONE_COUNT += 1
        Drone.DRONE_REFERENCE.append(self)

    def retarget(self):
        """
        Retarget the drone to a random target
        """
        self.target = np.random.rand(2)*800  # Random target within a 800x600 area

    def think(self):
        pass

    def get_neighbors_ids(self) -> np.ndarray:
        """
        Returns the list of neighbors of the drone
        (using get_graph)
        """
        return np.where(Drone.get_graph()[self.id] == 1)[0]

    def get_neighbors_references(self) -> set:
        """
        Returns the list of neighbors of the drone
        (using get_graph)
        """
        return set(np.where(Drone.get_graph()[self.id] == 1)[0])


    def update(self, dt):
        """
        Update the drone position
        dt is the time step
        """

        self._neighbors = None  # Reset neighbors to force recalculation next time
        # /!\ if another drone has been updated, neighbors may have changed
        # all drones should be updated in the same tick. no get_neighbors call should be made between updates

        dir = np.linalg.norm(self.target - self.position)
        if dir > 0:
            dir = (self.target - self.position) / dir
        else:
            dir = np.zeros_like(self.position)
        
        # Don't move if the drone is already at the target
        if np.linalg.norm(self.target - self.position) < TARGET_RADIUS:
            dir = np.zeros_like(dir)
        self.position += dir * DRONE_SPEED * dt

        # reset the graph to force recalculation next time
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
                thickness=4,
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


