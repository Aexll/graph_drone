import dearpygui.dearpygui as dpg
import numpy as np
import math


class GraphVisualizer:
    def __init__(self, nodes, targets, dist_threshold=150):
        self.nodes = nodes
        self.targets = targets
        self.dist_threshold = dist_threshold

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def draw_dotted_line(self, p1, p2, color, thickness, parent, dot_length=5, gap_length=5):
        dist = self.distance(p1, p2)
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
            if self.distance((start_x, start_y), p2) < dot_length:
                end_x, end_y = p2
            dpg.draw_line((start_x, start_y), (end_x, end_y), color=color, thickness=thickness, parent=parent)

    def show(self):
        width, height = 1000, 800
        node_radius = 10
        node_color = (100, 150, 255, 255)
        node_outline = (0, 0, 0, 255)
        target_color = (255, 100, 100, 255)
        target_outline = (0, 0, 0, 255)
        line_color = (150, 150, 255, 255)
        dotted_color = (150, 150, 255, 180)


        # To center the graph in the window, compute the centroid of all points (nodes and targets),
        # then compute the offset needed to move the centroid to the center of the window,
        # and apply this offset to all node and target positions for drawing.

        # Compute centroid of all points
        all_points = np.vstack([self.nodes, self.targets])
        centroid = np.mean(all_points, axis=0)

        # Compute center of the window
        window_center = np.array([width / 2, height / 2])

        # Compute offset
        offset = window_center - centroid

        # Apply offset to nodes and targets for drawing
        nodes_centered = self.nodes + offset
        targets_centered = self.targets + offset

        self.nodes = nodes_centered
        self.targets = targets_centered


        with dpg.window(label="Graph Visualizer", width=width+40, height=height+60):
            with dpg.drawlist(width=width, height=height) as drawlist_tag:
                # Draw dotted lines between each node and its target
                for i, (p1, p2) in enumerate(zip(self.nodes, self.targets)):
                    self.draw_dotted_line(tuple(p1), tuple(p2), color=dotted_color, thickness=2, parent=drawlist_tag)
                # Draw lines between nodes if close enough
                for i, p1 in enumerate(self.nodes):
                    for j, p2 in enumerate(self.nodes):
                        if i < j and self.distance(p1, p2) <= self.dist_threshold:
                            dpg.draw_line(tuple(p1), tuple(p2), color=line_color, thickness=2, parent=drawlist_tag)
                # Draw nodes
                for idx, (x, y) in enumerate(self.nodes):
                    dpg.draw_circle((float(x), float(y)), node_radius, color=node_outline, fill=node_color, parent=drawlist_tag)
                    dpg.draw_text((float(x) - 6, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent=drawlist_tag)
                # Draw targets
                for idx, (x, y) in enumerate(self.targets):
                    dpg.draw_circle((float(x), float(y)), node_radius, color=target_outline, fill=target_color, parent=drawlist_tag)
                    dpg.draw_text((float(x) - 6, float(y) - 8), f"{idx}", size=15, color=(0,0,0,255), parent=drawlist_tag)




def create_graph_visualizer(nodes, targets):
    dpg.create_context()
    visualizer = GraphVisualizer(nodes, targets)
    visualizer.show()
    dpg.create_viewport(title="Graph Drawing", width=1100, height=900, resizable=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    nb_nodes = 6
    nodes = np.random.uniform(100, 200, (nb_nodes, 2))
    targets = np.random.uniform(100, 600, (nb_nodes, 2))
    create_graph_visualizer(nodes, targets)