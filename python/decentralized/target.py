import numpy as np
import dearpygui.dearpygui as dpg

"""
0: static
1: moving
2: dynamic
"""

class Target:
    def __init__(self, position, target_type:int, bounds:np.ndarray):
        self.position :np.ndarray = position.astype(np.float32)
        self.target_type :int = target_type
        self.bounds :np.ndarray = bounds

        self.radius = 10
        self.speed = 5.0

        self.acceptable_distance = 5.0  # Distance within which the target is considered reached
        self.move_towards_target = np.random.rand(2) * self.bounds[1]  # Random target position within bounds

    def draw(self, draw_list):
        dpg.draw_circle((self.position[0], self.position[1]), 10, color=(0, 255, 0), parent=draw_list)

    def update(self, delta_time=0.08):
        if self.target_type == 1:
            # If the target is moving, update its position
            self.move_towards(self.move_towards_target, delta_time=delta_time)

    def move_towards(self, target_position:np.ndarray, delta_time:float):
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        if distance > self.acceptable_distance:
            direction /= distance
            self.position += direction * self.speed * delta_time
        else:
            self.move_towards_target = np.random.rand(2) * self.bounds[1]
        # Ensure the target stays within bounds
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    

