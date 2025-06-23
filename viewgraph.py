import dearpygui.dearpygui as dpg
import math
import random
import numpy as np
from errorcalc import calculate_graph_connectivity, cout_snt, cout_total, cout_min
from learning import optimize_nodes, make_error_function, mutate_nodes

# Window sizes
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
WINDOW_INFO_WH = (400, 800)
WINDOW_GRAPH_WH = (1000, 800)

DRONE_COUNT = 10
DRONE_SPAWN_RADIUS = 100

# options
auto_mutate_rate = 100 # ms 
auto_mutate_n = 1
error_power = 1

# Example set of coordinates (x, y)
coordinates:np.ndarray = np.random.uniform(WINDOW_GRAPH_WH[0]/2-DRONE_SPAWN_RADIUS, WINDOW_GRAPH_WH[0]/2+DRONE_SPAWN_RADIUS, (DRONE_COUNT, 2)).astype(np.float32)
# Targets
target_coordinates:np.ndarray = np.random.uniform(0, WINDOW_GRAPH_WH[1], (DRONE_COUNT, 2)).astype(np.float32)

history = []

def randomize_targets():
    offset = 100
    global target_coordinates
    target_coordinates = np.array([
        [random.randint(offset, WINDOW_GRAPH_WH[1]-offset), random.randint(offset, WINDOW_GRAPH_WH[1]-offset)]
        for _ in range(len(coordinates))
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

    barycenter_targets = np.mean(target_coordinates, axis=0)
    barycenter_nodes = np.mean(coordinates, axis=0)

    # Draw links
    for i, p1 in enumerate(coordinates):
        for j, p2 in enumerate(coordinates):
            if i < j and distance(p1, p2) <= DIST_THRESHOLD:
                dpg.draw_line(tuple(p1), tuple(p2), color=(150, 150, 255, 255), thickness=2, parent="graph_draw")
    
    # Draw ray from barycenter to targets
    if SHOW_RAYS:
        for i, p1 in enumerate(target_coordinates):
            d = np.linalg.norm(barycenter_targets - target_coordinates[i]) * 0.5
            dpg.draw_line((float(barycenter_targets[0]), float(barycenter_targets[1])), (float(target_coordinates[i][0]), float(target_coordinates[i][1])), color=(150, 150, 255, 255), thickness=2, parent="graph_draw")

    # Draw links between nodes and targets (dotted lines)
    if SHOW_TARGET_LINES:
        for i, p1 in enumerate(coordinates):
            for j, p2 in enumerate(target_coordinates):
                if i==j:
                    d = distance(p1, p2)
                    cd = 255-max(min(d/2, 255),0)
                    draw_dotted_line(tuple(p1), tuple(p2), color=(150, 150, 255, cd), thickness=2, parent="graph_draw")

    # Draw nodes
    for idx, (x, y) in enumerate(coordinates):
        dpg.draw_circle((float(x), float(y)), NODE_RADIUS, color=(0, 0, 0, 255), fill=(100, 200, 255, 255), parent="graph_draw")
        dpg.draw_text((float(x) - 4, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw targets
    for idx, (x, y) in enumerate(target_coordinates):
        dpg.draw_circle((float(x), float(y)), NODE_RADIUS, color=(0, 0, 0, 255), fill=(255, 100, 50, 255), parent="graph_draw")
        dpg.draw_text((float(x) - 4, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw barycenter of targets
    dpg.draw_circle((float(barycenter_targets[0]), float(barycenter_targets[1])), NODE_RADIUS/2, color=(0, 0, 0, 255), fill=(255, 100, 50, 255), parent="graph_draw")
    dpg.draw_text((float(barycenter_targets[0]) - 4, float(barycenter_targets[1]) - 8), "", size=15, color=(0,0,0,255), parent="graph_draw")

    # Draw barycenter of nodes
    dpg.draw_circle((float(barycenter_nodes[0]), float(barycenter_nodes[1])), NODE_RADIUS/2, color=(0, 0, 0, 255), fill=(100, 200, 255, 255), parent="graph_draw")
    dpg.draw_text((float(barycenter_nodes[0]) - 4, float(barycenter_nodes[1]) - 8), "", size=15, color=(0,0,0,255), parent="graph_draw")

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

def update_error(sender, app_data):
    dpg.set_value("error_linear_text", "Error Linear: " + str(cout_snt(coordinates, target_coordinates)))
    dpg.set_value("error_square_text", "Error Square: " + str(cout_total(coordinates, target_coordinates)))
    dpg.set_value("error_max_text", "Error Max: " + str(cout_min(coordinates, target_coordinates)))
    dpg.set_value("connectivity_text", "Connectivity: " + str(calculate_graph_connectivity(coordinates, DIST_THRESHOLD)))

def optimize_nodes_callback(sender, app_data):
    global coordinates
    best_nodes = optimize_nodes(coordinates, target_coordinates, make_error_function(DIST_THRESHOLD, error_function, error_power))
    coordinates = best_nodes
    draw_graph(None, None)
    update_error(None, None)

def mutate_nodes_callback(sender, app_data):
    global coordinates
    coordinates, hist = mutate_nodes(coordinates, target_coordinates, make_error_function(DIST_THRESHOLD, error_function, error_power))
    history.extend(hist)
    draw_graph(None, None)
    update_error(None, None)

def auto_mutate_callback(sender, app_data):
    if dpg.get_value("auto_mutate_checkbox"):
        global coordinates
        coordinates, hist = mutate_nodes(coordinates, target_coordinates, make_error_function(DIST_THRESHOLD, error_function, error_power), n_mutations=auto_mutate_n)
        history.extend(hist)
        draw_graph(None, None)
        update_error(None, None)
        dpg.set_frame_callback(dpg.get_frame_count() + int(auto_mutate_rate*0.001 / dpg.get_delta_time()), auto_mutate_callback)


error_function = cout_total
def update_error_function(sender, app_data):
    global error_function
    print(app_data)
    functions = {
        "SNT": cout_snt,
        "Total": cout_total,
        "Min": cout_min
    }
    error_function = functions[app_data]
    # make_error_function(DIST_THRESHOLD, functions[app_data], error_power)
    draw_graph(None, None)
    update_error(None, None)

def update_auto_mutate_n(sender, app_data):
    global auto_mutate_n
    auto_mutate_n = app_data

def update_auto_mutate_rate(sender, app_data):
    global auto_mutate_rate
    auto_mutate_rate = app_data

def update_error_power(sender, app_data):
    global error_power
    error_power = app_data

def main():
    dpg.create_context()
    dpg.create_viewport(title='Graph Visualizer', width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    with dpg.window(label="Graph", width=WINDOW_GRAPH_WH[0], height=WINDOW_GRAPH_WH[1]):
        dpg.add_drawlist(width=WINDOW_GRAPH_WH[0], height=WINDOW_GRAPH_WH[1], tag="graph_draw")
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=mouse_handler)
            dpg.add_mouse_click_handler(callback=mouse_handler)
            dpg.add_mouse_drag_handler(callback=mouse_handler)
            dpg.add_mouse_release_handler(callback=mouse_handler)

    with dpg.window(label="Info", width=WINDOW_INFO_WH[0], height=WINDOW_INFO_WH[1], pos=[WINDOW_GRAPH_WH[0], 0]):
        dpg.add_text("Options")
        dpg.add_separator()
        dpg.add_button(label="Randomize Targets", callback=randomize_targets)
        dpg.add_input_text(label="Distance Threshold",default_value=str(DIST_THRESHOLD),callback=update_dist_threshold)
        dpg.add_text(label="Error Linear",tag="error_linear_text", parent="Info")
        dpg.add_text(label="Error Square",tag="error_square_text", parent="Info")
        dpg.add_text(label="Error Max",tag="error_max_text", parent="Info")
        dpg.add_text(label="Connectivity",tag="connectivity_text", parent="Info")
        dpg.add_button(label="Mutate",callback=mutate_nodes_callback)
        dpg.add_input_int(label="Auto Mutate N",default_value=auto_mutate_n,callback=update_auto_mutate_n)
        dpg.add_input_int(label="Auto Mutate Rate",default_value=auto_mutate_rate,callback=update_auto_mutate_rate)
        dpg.add_input_int(label="Error power",default_value=1,callback=update_error_power)

        dpg.add_combo(
            label="Error Function",
            items=["SNT", "Total", "Min"],
            default_value="Total",
            tag="error_function_combo",
            callback=update_error_function
        )

        # toggle mutation
        dpg.add_checkbox(label="Auto Mutate", tag="auto_mutate_checkbox", default_value=False)
        dpg.set_item_callback("auto_mutate_checkbox", auto_mutate_callback)

    draw_graph(None, None)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()












