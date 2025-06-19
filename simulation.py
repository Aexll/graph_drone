import numpy as np
import dearpygui.dearpygui as dpg
import time
import algograph as ag
from drone import Drone


drones = [
    Drone((100, 100), (200, 200)),
    Drone((300, 300), (400, 400)),
    Drone((500, 500), (600, 600)),
    Drone((700, 700), (800, 800)),
    Drone((400, 100), (400, 200)),
]

ag.update_graph(Drone.get_graph())

# Global variables
TICK_ENABLED = False
tick = 0


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

    # Draw drones
    for drone in drones:
        drone.draw()
    
    tick += 1




## Buttons

def btn_add_drone():
    drones.append(Drone((np.random.rand() * 800, np.random.rand() * 600), (np.random.rand() * 800, np.random.rand() * 600)))

def btn_retarget():
    for drone in drones:
        drone.retarget()

def btn_toggle_tick():
    global TICK_ENABLED
    TICK_ENABLED = not TICK_ENABLED
    # print(f"TICK_ENABLED set to {TICK_ENABLED}")


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
    # with dpg.window(label="Drone Tables", tag="tables_window", pos=[820, 0]):
    #     with dpg.tab_bar():
    #         with dpg.tab(label="ξ(i,j) Table", tag="xi_tab"):
    #             # create_xi_table()
    #             pass
    #         with dpg.tab(label="ω(i,j) Table", tag="omega_tab"):
    #             # create_omega_table()
    #             pass
    #         with dpg.tab(label="Neighbors Table", tag="neighbors_tab"):
    #             # neighbors_table()
    #             pass
    
    
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
        ag.update_graph(Drone.get_graph())
        dpg.render_dearpygui_frame()
        time.sleep(1/60)  # 60 FPS
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()