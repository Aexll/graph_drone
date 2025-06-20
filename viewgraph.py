import dearpygui.dearpygui as dpg
import math
import random
import numpy as np
from errorcalc import calculate_error_graph, calculate_graph_connectivity, error_linear, error_square, er_lin, er_sq
from learning import optimize_nodes, make_error_function, mutate_nodes
from renforcement import GraphEnv
from stable_baselines3 import PPO

# Example set of coordinates (x, y)
coordinates:np.ndarray = np.array([
    [100, 100],
    [200, 150],
    [300, 100],
    [400, 200],
    [250, 300],
    [120, 250]
]).astype(np.float32)

# Targets
target_coordinates:np.ndarray = np.array([
    [320, 200],
    [200, 400],
    [320, 400],
    [100, 500],
    [200, 600],
    [320, 700]
]).astype(np.float32)

history = []

def randomize_targets():
    global target_coordinates
    target_coordinates = np.array([
        [random.randint(0, 900), random.randint(0, 900)]
        for _ in range(6)
    ])
    draw_graph(None, None)


def draw_dotted_line(p1, p2, color, thickness, parent, dot_length=5, gap_length=5):
    # Calculate the total distance and direction
    dist = distance(p1, p2)
    if dist == 0:
        return
    dx = (p2[0] - p1[0]) / dist
    dy = (p2[1] - p1[1]) / dist

    num_dots = int(dist // (dot_length + gap_length))
    for i in range(num_dots + 1):
        start_x = p1[0] + (dot_length + gap_length) * i * dx
        start_y = p1[1] + (dot_length + gap_length) * i * dy
        end_x = start_x + dot_length * dx
        end_y = start_y + dot_length * dy

        # Clamp the end point to not overshoot
        if distance((start_x, start_y), p2) < dot_length:
            end_x, end_y = p2
        dpg.draw_line((start_x, start_y), (end_x, end_y), color=color, thickness=thickness, parent=parent)


# Distance threshold for connecting nodes
DIST_THRESHOLD = 150

# Node drawing parameters
NODE_RADIUS = 10

SHOW_TARGET_LINES = False
SHOW_RAYS = False

# Drag state
_dragging_node = {'index': None, 'offset': (0, 0)}


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def draw_graph(sender, app_data):
    dpg.delete_item("graph_draw", children_only=True)
    # Draw links
    for i, p1 in enumerate(coordinates):
        for j, p2 in enumerate(coordinates):
            if i < j and distance(p1, p2) <= DIST_THRESHOLD:
                dpg.draw_line(tuple(p1), tuple(p2), color=(150, 150, 255, 255), thickness=2, parent="graph_draw")
    

    barycenter = np.mean(target_coordinates, axis=0)

    if SHOW_RAYS:
        # Draw ray from barycenter to targets
        for i, p1 in enumerate(target_coordinates):
            d = np.linalg.norm(barycenter - target_coordinates[i])
            dpg.draw_line((float(barycenter[0]), float(barycenter[1])), (float(target_coordinates[i][0]), float(target_coordinates[i][1])), color=(150, 150, 255, 255), thickness=2, parent="graph_draw")


    

    # Draw nodes
    for idx, (x, y) in enumerate(coordinates):
        dpg.draw_circle((float(x), float(y)), NODE_RADIUS, color=(0, 0, 0, 255), fill=(100, 200, 255, 255), parent="graph_draw")
        dpg.draw_text((float(x) - 4, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw targets
    for idx, (x, y) in enumerate(target_coordinates):
        dpg.draw_circle((float(x), float(y)), NODE_RADIUS, color=(0, 0, 0, 255), fill=(255, 100, 50, 255), parent="graph_draw")
        dpg.draw_text((float(x) - 4, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent="graph_draw")

    if SHOW_TARGET_LINES:
    # Draw links between nodes and targets (dotted lines)
        for i, p1 in enumerate(coordinates):
            for j, p2 in enumerate(target_coordinates):
                if i==j:
                    d = distance(p1, p2)
                    cd = 255-max(min(d/2, 255),0)
                    draw_dotted_line(tuple(p1), tuple(p2), color=(150, 150, 255, cd), thickness=2, parent="graph_draw")

    # Draw barycenter of targets
    dpg.draw_circle((float(barycenter[0]), float(barycenter[1])), NODE_RADIUS/2, color=(0, 0, 0, 255), fill=(255, 100, 50, 255), parent="graph_draw")
    dpg.draw_text((float(barycenter[0]) - 4, float(barycenter[1]) - 8), "", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw barycenter of nodes
    barycenter = np.mean(coordinates, axis=0)
    dpg.draw_circle((float(barycenter[0]), float(barycenter[1])), NODE_RADIUS/2, color=(0, 0, 0, 255), fill=(100, 200, 255, 255), parent="graph_draw")
    dpg.draw_text((float(barycenter[0]) - 4, float(barycenter[1]) - 8), "", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw history as a line from the first to the last point
    for i in range(len(history) - 1):
        p1 = history[i]
        p2 = history[i + 1]
        dpg.draw_line((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), color=(0, 0, 255, 100), thickness=2, parent="graph_draw")

    update_error(None, None)

def mouse_handler(sender, app_data):
    mouse_pos = dpg.get_mouse_pos(local=False)
    drawlist_pos = dpg.get_item_rect_min("graph_draw")
    rel_mouse = (mouse_pos[0] - drawlist_pos[0], mouse_pos[1] - drawlist_pos[1])
    mouse_down = dpg.is_mouse_button_down(0)
    mouse_clicked = dpg.is_mouse_button_clicked(0)
    mouse_released = dpg.is_mouse_button_released(0)

    if _dragging_node['index'] is not None:
        if mouse_down:
            idx = _dragging_node['index']
            offset = _dragging_node['offset']
            coordinates[idx][0] = rel_mouse[0] - offset[0]
            coordinates[idx][1] = rel_mouse[1] - offset[1]
            draw_graph(None, None)
        if mouse_released:
            _dragging_node['index'] = None
    elif mouse_clicked:
        # Check if mouse is over a node
        for idx, (x, y) in enumerate(coordinates):
            if distance((x, y), rel_mouse) <= NODE_RADIUS:
                _dragging_node['index'] = idx
                _dragging_node['offset'] = (rel_mouse[0] - x, rel_mouse[1] - y)
                break


def update_dist_threshold(sender, app_data):
            global DIST_THRESHOLD
            try:
                DIST_THRESHOLD = float(app_data)
            except ValueError:
                pass
            draw_graph(None, None)

ERROR_FUNCTION = error_linear

def update_error(sender, app_data):
    dpg.set_value("error_text", "Error Linear: " + str(er_lin(coordinates, target_coordinates)))
    dpg.set_value("error_square_text", "Error Square: " + str(er_sq(coordinates, target_coordinates)))
    dpg.set_value("connectivity_text", "Connectivity: " + str(calculate_graph_connectivity(coordinates, DIST_THRESHOLD)))

def optimize_nodes_callback(sender, app_data):
    global coordinates
    best_nodes = optimize_nodes(coordinates, target_coordinates, make_error_function(DIST_THRESHOLD, ERROR_FUNCTION ))
    coordinates = best_nodes
    draw_graph(None, None)
    update_error(None, None)

def mutate_nodes_callback(sender, app_data):
    global coordinates
    coordinates, hist = mutate_nodes(coordinates, target_coordinates, make_error_function(DIST_THRESHOLD, ERROR_FUNCTION ))
    history.extend(hist)
    draw_graph(None, None)
    update_error(None, None)

def main():
    dpg.create_context()
    dpg.create_viewport(title='Graph Visualizer', width=1400, height=800)

    with dpg.window(label="Graph", width=1000, height=800):
        dpg.add_drawlist(width=1000, height=800, tag="graph_draw")
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=mouse_handler)
            dpg.add_mouse_click_handler(callback=mouse_handler)
            dpg.add_mouse_drag_handler(callback=mouse_handler)
            dpg.add_mouse_release_handler(callback=mouse_handler)

    with dpg.window(label="Info", width=400, height=800, pos=[1000, 0]):
        dpg.add_text("Options")
        dpg.add_separator()
        dpg.add_button(label="Redraw Graph", callback=draw_graph)
        dpg.add_button(label="Randomize Targets", callback=randomize_targets)
        dpg.add_input_text(label="Distance Threshold",default_value=str(DIST_THRESHOLD),callback=update_dist_threshold)
        dpg.add_text(label="Error",tag="error_text", parent="Info")
        dpg.add_text(label="Error Square",tag="error_square_text", parent="Info")
        dpg.add_text(label="Connectivity",tag="connectivity_text", parent="Info")
        dpg.add_button(label="Optimize",callback=optimize_nodes_callback)
        dpg.add_button(label="Mutate",callback=mutate_nodes_callback)
        dpg.add_checkbox(label="Autostep", tag="autostep_checkbox", default_value=False)

        dpg.add_button(label="Step", callback=step_env)

        # Timer for autostep
        def autostep_callback():
            if dpg.get_value("autostep_checkbox"):
                step_env()
                dpg.set_frame_callback(dpg.get_frame_count() + int(0.1 / dpg.get_delta_time()), autostep_callback)

        def autostep_toggle(sender, app_data):
            if app_data:
                dpg.set_frame_callback(dpg.get_frame_count() + int(0.1 / dpg.get_delta_time()), autostep_callback)

        dpg.set_item_callback("autostep_checkbox", autostep_toggle)
        

    
    draw_graph(None, None)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


current_env = GraphEnv(n_nodes=len(coordinates), dist_threshold=int(DIST_THRESHOLD), nodes=coordinates, targets=target_coordinates)
model = PPO.load("ppo_graph_env")
def make_env():
    global current_env
    current_env = GraphEnv(n_nodes=len(coordinates), dist_threshold=int(DIST_THRESHOLD), nodes=coordinates, targets=target_coordinates)
    return current_env

def step_env():
    global current_env, coordinates
    action, _ = model.predict(current_env._get_obs())
    # action = current_env.action_space.sample()
    obs, reward, terminated, truncated, info = current_env.step(action)
    coordinates = current_env.nodes
    draw_graph(None, None)
    update_error(None, None)
    return obs, reward, terminated, truncated, info
    


if __name__ == "__main__":
    main()












