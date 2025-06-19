import numpy as np
import dearpygui.dearpygui as dpg
import time
import algograph as ag
from drone import Drone

DRONE_RADIUS = 10
DRONE_SPEED = 200
CONEXION_RADIUS = 400
TARGET_RADIUS = 5
FREE_TICKS = 2 # Every N ticks, the drones will update they omega and xi values





"""
Note: when a variable start with an underscore it means that the drone is not aware of it, 
it is just for drawing purposes, the drone will not use it to make decisions.
"""



drones = [
    Drone((100, 100), (200, 200)),
    Drone((300, 300), (400, 400)),
    Drone((500, 500), (600, 600)),
    Drone((700, 700), (800, 800)),
    Drone((400, 100), (400, 200)),
]



drones.append(Drone((100, 100), (200, 200)))
print(Drone.get_graph())




ag.GRAPH = Drone.get_graph()

print(ag.N(1))

quit()


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
    return v

def Δ_array(i,l,n):
    """
    returns the distance between drone i and drone j, witout passing by the edge i,l
    """
    ret = []
    for j in range(Drone.DRONE_COUNT):
        ret.append(Δ(i,j,l,n))
    return ret




def is_critical_edge(i,l):
    """ returns if the edge i,l is critical """
    n = Drone.DRONE_COUNT
    for j in range(n):
        if Δ(i,j,l,n + 1) == 0:
            return False
        for ii in N(i):
            for ll in N(l):
                if ii == l or ll == i:
                    continue
                if Δ(i,j,ii,n + 1) == 1 and Δ(l,j,ll,n + 1) == 1:
                    return False
    return True
    


########################
#### DISPLAY TABLES ####
########################



# Global variables
TICK_ENABLED = False
tick = 0
running = True


def update():
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
    # if tick % FREE_TICKS == 0:
    #     clear_memory()    

    for drone in drones:
        drone.draw()
    
    tick += 1




## Buttons

def btn_add_drone():
    drones.append(Drone((np.random.rand() * 800, np.random.rand() * 600), (np.random.rand() * 800, np.random.rand() * 600)))
    # Recreate tables to accommodate new drone
    # recreate_xi_table()
    # recreate_omega_table()
    # Clear memory cache since drone count changed
    # clear_memory()

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
                # create_xi_table()
                pass
            with dpg.tab(label="ω(i,j) Table", tag="omega_tab"):
                # create_omega_table()
                pass
            with dpg.tab(label="Neighbors Table", tag="neighbors_tab"):
                # neighbors_table()
                pass
    
    
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
        update()
        # update_xi_table()
        # update_omega_table()
        dpg.render_dearpygui_frame()
        time.sleep(1/60)  # 60 FPS
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()