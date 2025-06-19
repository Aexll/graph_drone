import numpy as np
import dearpygui.dearpygui as dpg
import sys
import threading
import time

DRONE_RADIUS = 10
DRONE_SPEED = 200
CONEXION_RADIUS = 300
TARGET_RADIUS = 5

"""
Note: when a variable start with an underscore it means that the drone is not aware of it, 
it is just for drawing purposes, the drone will not use it to make decisions.
"""

class Drone:
    DRONE_COUNT = 0
    DRONE_REFERENCE = []
    
    def __init__(self, position, target):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self.id = Drone.DRONE_COUNT
        Drone.DRONE_COUNT += 1
        Drone.DRONE_REFERENCE.append(self)

    def retarget(self):
        self.target = np.random.rand(2) * 500

    def think(self):
        pass

    def get_neighbors(self) -> set:
        """
        Returns the list of neighbors of the drone
        """
        ret = set()
        for i in range(Drone.DRONE_COUNT):
            ref = Drone.DRONE_REFERENCE[i]
            if ref is not self and np.linalg.norm(self.position - ref.position) < CONEXION_RADIUS:
                ret.add(ref)
        return ret

    def update(self, dt):
        dir = np.linalg.norm(self.target - self.position)
        if dir > 0:
            dir = (self.target - self.position) / dir
        else:
            dir = np.zeros_like(self.position)
        
        # Don't move if the drone is already at the target
        if np.linalg.norm(self.target - self.position) < TARGET_RADIUS:
            dir = np.zeros_like(dir)
        self.position += dir * DRONE_SPEED * dt

# Initialize drones
drones = [
    Drone((100, 100), (200, 200)),
    Drone((300, 300), (400, 400)),
    Drone((500, 500), (600, 600)),
]

def N(i):
    """
    returns the set of neighbors of drone i
    """
    return [n.id for n in Drone.DRONE_REFERENCE[i].get_neighbors()]

def ξ(i, j, n):
    """
    returns 1 if drone i is aware of drone j, 0 otherwise
    """
    if n <= 0: 
        return 1 if i == j else 0
    
    return max(ξ(l, j, n-1) for l in N(i) + [i])

inf = 100000000

def ω(i, j, n):
    """
    returns the smallest amount of edges that connects drone i to drone j
    """
    if n <= 0:
        return 0 if i == j else inf
    
    if ξ(i, j, n-1) == ξ(i, j, n): 
        return ω(i, j, n-1)
    else:
        return min(ω(l, j, n-1) + 1 for l in N(i))



# OPTIMIZED omega and xi functions

xi_memory = {}
def ξ_optimized(i, j, n):
    if n<= 0:
        return  1 if i == j else 0
    if (i, j, n) in xi_memory:
        return xi_memory[(i, j, n)]
    else:
        v = ξ(i, j, n)  # Call the original function to populate the memory
        xi_memory[(i, j, n)] = v
        return v
    
omega_memory = {}
def ω_optimized(i, j, n):
    if n <= 0:
        return 0 if i == j else inf
    if (i, j, n) in omega_memory:
        return omega_memory[(i, j, n)]
    else:
        v = ω(i, j, n)  # Call the original function to populate the memory
        omega_memory[(i, j, n)] = v
        return v


def clear_memory():
    """
    Clears the memory of the optimized functions
    """
    global xi_memory, omega_memory
    xi_memory = {}
    omega_memory = {}


# Global variables
TICK_ENABLED = False
tick = 0
running = True

def toggle_tick():
    global TICK_ENABLED
    TICK_ENABLED = not TICK_ENABLED
    print(f"TICK_ENABLED set to {TICK_ENABLED}")

def draw_simulation():
    global tick
    
    # Clear the drawing
    dpg.delete_item("simulation_drawing", children_only=True)
    
    # Retarget drones every 600 ticks (10 seconds at 60 FPS)
    if tick % 600 == 0:
        for drone in drones:
            drone.retarget()
    
    # Update drone thinking
    for drone in drones:
        drone.think()
    
    # Update positions if enabled
    if TICK_ENABLED:
        for drone in drones:
            drone.update(0.016)
    
    # Draw connections between neighbors
    for drone in drones:
        for neighbor in drone.get_neighbors():
            direction = neighbor.position - drone.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.zeros_like(direction)
            
            line_start = drone.position + direction * DRONE_RADIUS
            line_end = neighbor.position - direction * DRONE_RADIUS
            
            dpg.draw_line(
                line_start.tolist(), 
                line_end.tolist(), 
                color=[255, 0, 0, 255], 
                thickness=4,
                parent="simulation_drawing"
            )
    
    # Draw drones and targets
    for drone in drones:
        # Draw drone as circle  
        dpg.draw_circle(
            drone.position.tolist(), 
            DRONE_RADIUS, 
            color=[0, 0, 255, 255], 
            fill=[0, 0, 255, 255],
            parent="simulation_drawing"
        )
        
        # Draw target as cross
        target_pos = drone.target
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
            [drone.position[0] - 5, drone.position[1] - 5], 
            str(drone.id), 
            color=[255, 255, 255, 255], 
            size=16,
            parent="simulation_drawing"
        )
    
    tick += 1

def update_xi_table():
    n = Drone.DRONE_COUNT
    for i in range(n):
        for j in range(n):
            value = ξ_optimized(i, j, n)
            dpg.set_value(f"xi_cell_{i}_{j}", str(value))

def update_omega_table():
    n = Drone.DRONE_COUNT
    for i in range(n):
        for j in range(n):
            value = ω_optimized(i, j, n)
            display_value = str(value) if value < inf else "∞"
            dpg.set_value(f"omega_cell_{i}_{j}", display_value)

def create_xi_table():
    with dpg.table(header_row=True, tag="xi_table"):
        # Create columns
        dpg.add_table_column(label="i\\j")
        for j in range(Drone.DRONE_COUNT):
            dpg.add_table_column(label=f"j={j}")
        
        # Create rows
        for i in range(Drone.DRONE_COUNT):
            with dpg.table_row():
                dpg.add_text(f"i={i}")
                for j in range(Drone.DRONE_COUNT):
                    dpg.add_text("0", tag=f"xi_cell_{i}_{j}")

def create_omega_table():
    with dpg.table(header_row=True, tag="omega_table"):
        # Create columns
        dpg.add_table_column(label="i\\j")
        for j in range(Drone.DRONE_COUNT):
            dpg.add_table_column(label=f"j={j}")
        
        # Create rows
        for i in range(Drone.DRONE_COUNT):
            with dpg.table_row():
                dpg.add_text(f"i={i}")
                for j in range(Drone.DRONE_COUNT):
                    dpg.add_text("0", tag=f"omega_cell_{i}_{j}")

def recreate_xi_table():
    """Recreate the xi table with updated drone count"""
    if dpg.does_item_exist("xi_table"):
        dpg.delete_item("xi_table")
    
    with dpg.table(header_row=True, tag="xi_table", parent="xi_tab"):
        # Create columns
        dpg.add_table_column(label="i\\j")
        for j in range(Drone.DRONE_COUNT):
            dpg.add_table_column(label=f"j={j}")
        
        # Create rows
        for i in range(Drone.DRONE_COUNT):
            with dpg.table_row():
                dpg.add_text(f"i={i}")
                for j in range(Drone.DRONE_COUNT):
                    dpg.add_text("0", tag=f"xi_cell_{i}_{j}")

def recreate_omega_table():
    """Recreate the omega table with updated drone count"""
    if dpg.does_item_exist("omega_table"):
        dpg.delete_item("omega_table")
    
    with dpg.table(header_row=True, tag="omega_table", parent="omega_tab"):
        # Create columns
        dpg.add_table_column(label="i\\j")
        for j in range(Drone.DRONE_COUNT):
            dpg.add_table_column(label=f"j={j}")
        
        # Create rows
        for i in range(Drone.DRONE_COUNT):
            with dpg.table_row():
                dpg.add_text(f"i={i}")
                for j in range(Drone.DRONE_COUNT):
                    dpg.add_text("0", tag=f"omega_cell_{i}_{j}")

def add_drone():
    drones.append(Drone((np.random.rand() * 800, np.random.rand() * 600), (np.random.rand() * 800, np.random.rand() * 600)))
    # Recreate tables to accommodate new drone
    recreate_xi_table()
    recreate_omega_table()
    # Clear memory cache since drone count changed
    clear_memory()

def main():
    dpg.create_context()
    
    # Create main window
    with dpg.window(label="Drone Simulation", tag="main_window"):
        dpg.add_text("Press 'T' to toggle movement, 'ESC' to exit")
        dpg.add_button(label="Toggle Movement", callback=toggle_tick)
        dpg.add_button(label="Add Drone", callback=add_drone)
        
        # Create drawing area
        with dpg.drawlist(width=800, height=600, tag="simulation_drawing"):
            pass
    
    # Create tabbed interface for tables
    with dpg.window(label="Drone Tables", tag="tables_window", pos=[820, 0]):
        with dpg.tab_bar():
            with dpg.tab(label="ξ(i,j) Table", tag="xi_tab"):
                create_xi_table()
            with dpg.tab(label="ω(i,j) Table", tag="omega_tab"):
                create_omega_table()
    
    # Set up keyboard handler
    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_T, callback=lambda: toggle_tick())
        dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())
    
    # Create viewport
    dpg.create_viewport(title="Drone Simulation with DearPyGui", width=1200, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Set primary window
    dpg.set_primary_window("main_window", True)
    
    # Main loop
    while dpg.is_dearpygui_running():
        draw_simulation()
        update_xi_table()
        update_omega_table()
        clear_memory()
        dpg.render_dearpygui_frame()
        time.sleep(1/60)  # 60 FPS
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()