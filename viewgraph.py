import dearpygui.dearpygui as dpg
import math

# Example set of coordinates (x, y)
coordinates = [
    (100, 100),
    (200, 150),
    (300, 100),
    (400, 200),
    (250, 300),
    (120, 250)
]

# Distance threshold for connecting nodes
DIST_THRESHOLD = 300

# Node drawing parameters
NODE_RADIUS = 10


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def draw_graph(sender, app_data):
    dpg.delete_item("graph_draw", children_only=True)
    # Draw links
    for i, p1 in enumerate(coordinates):
        for j, p2 in enumerate(coordinates):
            if i < j and distance(p1, p2) <= DIST_THRESHOLD:
                dpg.draw_line(p1, p2, color=(150, 150, 255, 255), thickness=2, parent="graph_draw")
    # Draw nodes
    for idx, (x, y) in enumerate(coordinates):
        dpg.draw_circle((x, y), NODE_RADIUS, color=(0, 0, 0, 255), fill=(100, 200, 255, 255), parent="graph_draw")
        dpg.draw_text((x + NODE_RADIUS + 2, y - NODE_RADIUS - 2), f"{idx}", size=15, color=(0,0,0,255), parent="graph_draw")


def main():
    dpg.create_context()
    dpg.create_viewport(title='Graph Visualizer', width=1200, height=800)

    with dpg.window(label="Graph", width=1000, height=800):
        dpg.add_button(label="Redraw Graph", callback=draw_graph)
        dpg.add_drawlist(width=580, height=350, tag="graph_draw")

    draw_graph(None, None)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
