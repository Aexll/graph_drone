"""

Drones will send messages to each other to inform them of their omega and xi values.


"""
import numpy as np
import dearpygui.dearpygui as dpg

class Message:

    SPEED = 800

    def __init__(self, from_id, to_id, name,**kwargs):
        self.from_id = from_id
        self.to_id = to_id
        self.kwargs = kwargs
        self.progress = 0

    
    def __str__(self):
        return f"Message from drone {self.from_id} to drone {self.to_id} with {self.kwargs}"

    def draw(self,pos):
        dpg.draw_circle(
            list(map(float, pos)), 
            5, 
            color=[0, 255, 0, 255], 
            fill=[0, 255, 0, 255],
            parent="simulation_drawing"
        )
        






