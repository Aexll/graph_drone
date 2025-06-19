import numpy as np
import dearpygui.dearpygui as dpg
import sys
import threading
import time

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
    
    def __init__(self, position, target):
        self.position = np.array(position, dtype=float)
        self.target = np.array(target, dtype=float)
        self._neighbors = None
        self.id = Drone.DRONE_COUNT
        Drone.DRONE_COUNT += 1
        Drone.DRONE_REFERENCE.append(self)

    def retarget(self):
        self.target = np.random.rand(2)*800  # Random target within a 800x600 area

    def think(self):
        pass

    def get_neighbors(self) -> set:
        """
        Returns the list of neighbors of the drone
        """
        if self._neighbors is not None:
            return self._neighbors
        
        ret = set()
        for i in range(Drone.DRONE_COUNT):
            ref = Drone.DRONE_REFERENCE[i]
            if ref is not self and np.linalg.norm(self.position - ref.position) < CONEXION_RADIUS:
                ret.add(ref)
        return ret

    def update(self, dt):

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


    def draw(self):
        # Draw connections between neighbors
        for neighbor in self.get_neighbors():
            if self.id > neighbor.id:
                continue
            direction = neighbor.position - self.position
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.zeros_like(direction)
            
            line_start = self.position + direction * DRONE_RADIUS
            line_end = neighbor.position - direction * DRONE_RADIUS
            
            if is_critical_edge(self.id, neighbor.id):
                color = [255, 0, 0, 255]
            else:
                color = [255, 255, 255, 255]

            line_tag = f"edge_{self.id}_{neighbor.id}"
            dpg.draw_line(
                line_start.tolist(), 
                line_end.tolist(), 
                color=color, 
                thickness=4,
                parent="simulation_drawing",
                tag=line_tag
            )

            # Calculate a rectangle around the line for click area
            # Make the rectangle a bit wider for easier clicking
            rect_width = 10  # pixels

            # Calculate perpendicular vector
            dx, dy = line_end - line_start
            length = np.linalg.norm([dx, dy])
            if length == 0:
                perp = np.array([0, 0])
            else:
                perp = np.array([-dy, dx]) / length * rect_width

            p1 = line_start + perp
            p2 = line_end + perp
            p3 = line_end - perp
            p4 = line_start - perp

            # Overlay a transparent polygon for click detection
            click_tag = f"edge_click_{self.id}_{neighbor.id}"
            dpg.draw_polygon(
                [p1.tolist(), p2.tolist(), p3.tolist(), p4.tolist()],
                color=[0,0,0,0],  # invisible border
                fill=[0,0,0,10],  # almost invisible fill, but clickable
                parent="simulation_drawing",
                tag=click_tag,
                thickness=0
            )
            def make_callback(i, l):
                return lambda s, a: display_critical_edges(i, l)
            dpg.set_item_callback(click_tag, make_callback(self.id, neighbor.id))
    
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


# Initialize 5 drones in a cycle (pentagon)
import math

CYCLE_RADIUS = 250
CYCLE_CENTER = (400, 400)
NUM_DRONES = 5

drones = []
for i in range(NUM_DRONES):
    angle = 2 * math.pi * i / NUM_DRONES
    x = CYCLE_CENTER[0] + CYCLE_RADIUS * math.cos(angle)
    y = CYCLE_CENTER[1] + CYCLE_RADIUS * math.sin(angle)
    # Target is the next point in the cycle
    next_angle = 2 * math.pi * ((i + 1) % NUM_DRONES) / NUM_DRONES
    tx = CYCLE_CENTER[0] + CYCLE_RADIUS * math.cos(next_angle)
    ty = CYCLE_CENTER[1] + CYCLE_RADIUS * math.sin(next_angle)
    drones.append(Drone((x, y), (tx, ty)))





### ALGO ###


# Memory for optimized xi and omega functions

xi_memory = {}
omega_memory = {}


# these are the optimized versions of the xi and omega functions they are used to avoid recalculating the same values multiple times
# they use a dictionary to store the results of previous calculations
# the memory is cleared with the clear_memory function, it should be called every time the drones are updated 

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
    
    if (i, j, n) in xi_memory:
        return xi_memory[(i, j, n)]
    v = max(ξ(l, j, n-1) for l in N(i) + [i])
    xi_memory[(i, j, n)] = v
    return v

inf = 100000000

def ω(i, j, n):
    """
    returns the smallest amount of edges that connects drone i to drone j
    """
    if n <= 0:
        return 0 if i == j else inf
    
    if (i, j, n) in omega_memory:
        return omega_memory[(i, j, n)]
    
    if ξ(i, j, n-1) == ξ(i, j, n): 
        v = ω(i, j, n-1)
        omega_memory[(i, j, n)] = v
        return v
    else:
        v = min(ω(l, j, n-1) + 1 for l in N(i))
        omega_memory[(i, j, n)] = v
        return v



def Δ(i,j,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    v = ω(i, j, n) - ω(j, l, n)
    return max(min(v,1),-1) 




def is_critical_edge(i,l):
    """ returns if the edge i,l is critical """
    n = Drone.DRONE_COUNT
    for j in range(n):
        if Δ(i,j,l,n + 1) == 0:
            return False
        for ii in N(i):
            for ll in N(l):
                if Δ(i,j,ii,n + 1) == 1 and Δ(l,j,ll,n + 1) == 1:
                    return False
    # print(i,l,"critical")
    return True
    
def display_critical_edges(i,l):
    n = Drone.DRONE_COUNT
    if is_critical_edge(i,l):
        print(i,l,"critical")
    else:
        print(i,l,"not critical")
        for j in range(n):
            if Δ(i,j,l,n + 1) == 0:
                print(f"j = {j} : {Δ(i,j,l,n + 1)}")
            else: 
                for ii in N(i):
                    for ll in N(l):
                        if Δ(i,j,ii,n + 1) == 1 and Δ(l,j,ll,n + 1) == 1:
                            print(f"j = {j}, ii = {ii}, ll = {ll} : {Δ(i,j,ii,n + 1)} , {Δ(l,j,ll,n + 1)}")


########################
#### DISPLAY TABLES ####
########################

def clear_memory():
    """
    Clears the memory of the optimized functions
    """
    global xi_memory, omega_memory
    xi_memory = {}
    omega_memory = {}

def update_xi_table():
    n = Drone.DRONE_COUNT
    for i in range(n):
        for j in range(n):
            value = ξ(i, j, n)
            dpg.set_value(f"xi_cell_{i}_{j}", str(value))

def update_omega_table():
    n = Drone.DRONE_COUNT
    for i in range(n):
        for j in range(n):
            value = ω(i, j, n)
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





# Global variables
TICK_ENABLED = False
tick = 0
running = True


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

    # Clear memory for optimized functions
    if tick % FREE_TICKS == 0:
        clear_memory()    

    for drone in drones:
        drone.draw()
    
    tick += 1




## Buttons

def btn_add_drone():
    drones.append(Drone((np.random.rand() * 800, np.random.rand() * 600), (np.random.rand() * 800, np.random.rand() * 600)))
    # Recreate tables to accommodate new drone
    recreate_xi_table()
    recreate_omega_table()
    # Clear memory cache since drone count changed
    clear_memory()

def btn_retarget():
    for drone in drones:
        drone.retarget()

def btn_toggle_tick():
    global TICK_ENABLED
    TICK_ENABLED = not TICK_ENABLED
    print(f"TICK_ENABLED set to {TICK_ENABLED}")




def main():
    dpg.create_context()
    
    # Create main window
    with dpg.window(label="Drone Simulation", tag="main_window"):
        dpg.add_text("Press 'T' to toggle movement, 'ESC' to exit")
        dpg.add_button(label="Toggle Movement", callback=btn_toggle_tick)
        dpg.add_button(label="Add Drone", callback=btn_add_drone)
        dpg.add_button(label="Retarget", callback=btn_retarget)
        
        # Create drawing area
        with dpg.drawlist(width=800, height=800, tag="simulation_drawing"):
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
        dpg.add_key_press_handler(dpg.mvKey_T, callback=lambda: btn_toggle_tick())
        dpg.add_key_press_handler(dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())
    
    # Create viewport
    dpg.create_viewport(title="Drone Simulation with DearPyGui", width=1200, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Set primary window
    dpg.set_primary_window("main_window", True)
    
    # Main loop
    while dpg.is_dearpygui_running():
        draw_simulation()
        update_xi_table()
        update_omega_table()
        dpg.render_dearpygui_frame()
        time.sleep(1/60)  # 60 FPS
        # display_critical_edges(0,4)
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()