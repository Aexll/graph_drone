import dearpygui.dearpygui as dpg
import numpy as np
from typing import Dict, Set, Tuple, List, Any
import graphx as gx 







class GraphNetworkVisualizer:
    def __init__(self, shapes_dict: Dict[str, Dict[str, Any]], 
                 transition_history: Set[Tuple[str, str]],
                 get_image_func,
                 targets: np.ndarray,
                 dist_threshold: float,
                 parent):
        self.shapes_dict = shapes_dict
        self.transition_history = transition_history
        self.get_image_func = get_image_func
        self.targets = targets
        self.dist_threshold = dist_threshold
        self.parent = parent
        
        self.selected_graph = None
        self.graph_positions = {}
        self.graph_textures = {}
        self.connection_lines = []
        
    def organize_by_error_layers(self, num_layers: int = 6) -> Dict[int, List[str]]:
        """Organize graphs into layers based on error scores"""
        if not self.shapes_dict:
            return {}
            
        scores = [info['score'] for info in self.shapes_dict.values()]
        min_score, max_score = min(scores), max(scores)
        
        if max_score == min_score:
            # Tous les scores sont identiques, tout dans la première couche
            return {0: list(self.shapes_dict.keys()), **{i: [] for i in range(1, num_layers)}}
        
        layers = {i: [] for i in range(num_layers)}
        layer_width = (max_score - min_score) / num_layers
        
        for shape_key, info in self.shapes_dict.items():
            score = info['score']
            layer_idx = min(int((score - min_score) / layer_width), num_layers - 1)
            layers[layer_idx].append(shape_key)
            
        return layers
    
    def calculate_positions(self, layers: Dict[int, List[str]], 
                          image_size: int = 100, 
                          layer_spacing: int = 200,
                          vertical_spacing: int = 120) -> Dict[str, Tuple[int, int]]:
        """Calculate positions for each graph to avoid overlaps"""
        positions = {}
        
        for layer_idx, graphs in layers.items():
            if not graphs:
                continue
                
            x = 50 + layer_idx * layer_spacing
            
            # Calculate vertical positions to minimize link crossings
            # Simple approach: spread evenly with some randomization
            total_height = len(graphs) * vertical_spacing
            start_y = 50
            
            for i, graph_key in enumerate(graphs):
                y = start_y + i * vertical_spacing
                positions[graph_key] = (x, y)
                
        return positions
    
    def create_graph_textures(self, image_size: int = 100, skin: str = "default"):
        """Create textures for all graphs"""
        with dpg.texture_registry():
            for shape_key, info in self.shapes_dict.items():
                nodes = info['graph']
                
                # Generate image using the provided function
                img_array = self.get_image_func(
                    nodes, self.targets, self.dist_threshold, 
                    size=image_size/50, skin=skin, error=info['score'],
                    age_min=info['age_min'],
                    age_max=info['age_max'],
                    age=info['age'],
                    shape_key=shape_key,
                )
                
                # Convert to format suitable for DearPyGUI
                # img_array is RGBA format from matplotlib
                height, width = img_array.shape[:2]
                
                # Ensure we have the right data type and range
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                
                # Flatten the array in the correct order for DearPyGUI
                texture_data = img_array.flatten().astype(np.float32) / 255.0
                
                # Create texture
                if dpg.does_item_exist(f"texture_{shape_key}"):
                    dpg.delete_item(f"texture_{shape_key}")
                texture_id = dpg.add_raw_texture(
                    width=width, 
                    height=height, 
                    default_value=texture_data, 
                    format=dpg.mvFormat_Float_rgba,
                    tag=f"texture_{shape_key}"
                )
                self.graph_textures[shape_key] = texture_id
    
    def get_connected_graphs(self, selected_key: str) -> Set[str]:
        """Get all graphs connected to the selected one"""
        connected = set()
        for link in self.transition_history:
            if link[0] == selected_key:
                connected.add(link[1])
            elif link[1] == selected_key:
                connected.add(link[0])
        return connected
    
    def get_all_graph_at_distance_map(self, selected_key: str, distance_max: int) -> Dict[int, List[str]]:
        """Get a map of int:list of graph keys at that distance"""
        connected = {0: [selected_key]}
        distance = 1
        reached = set()
        while distance <= distance_max:
            connected[distance] = []
            for key in connected[distance - 1]:
                for link in self.transition_history:
                    if link[0] == key and link[1] not in reached:
                        connected[distance].append(link[1])
                    elif link[1] == key and link[0] not in reached:
                        connected[distance].append(link[0])
            reached.update(connected[distance])
            distance += 1
        return connected
    
    def on_graph_click(self, sender, app_data, user_data):
        """Handle graph click events"""
        clicked_graph = user_data
        
        if self.selected_graph == clicked_graph:
            # Deselect if clicking the same graph
            self.selected_graph = None
            self.clear_connections()
            self.reset_graph_opacity()
        else:
            # Select new graph
            self.selected_graph = clicked_graph
            self.update_visualization()
    
    def clear_connections(self):
        """Clear all connection lines"""
        for line_id in self.connection_lines:
            if dpg.does_item_exist(line_id):
                dpg.delete_item(line_id)
        self.connection_lines.clear()
    
    def update_visualization(self):
        """Update the visualization based on selected graph"""
        if not self.selected_graph:
            return
            
        self.clear_connections()
        
        # Get connected graphs
        connected = self.get_connected_graphs(self.selected_graph)

        max_distance = 8

        # Get distance dict 
        distance_dict = self.get_all_graph_at_distance_map(self.selected_graph, max_distance)

        
        # Since we can't change tint_color of draw_image, we'll redraw with different opacity
        # Clear the canvas and redraw everything
        dpg.delete_item("canvas")
        
        with dpg.drawlist(width=self.canvas_width, height=self.canvas_height, 
                         tag="canvas", parent=self.parent):
            # Redraw all graph images with appropriate opacity
            for shape_key, pos in self.graph_positions.items():
                x, y = pos
                
                # Determine opacity based on selection
                if shape_key == self.selected_graph:
                    # Full opacity - draw normally
                    dpg.draw_image(
                        f"texture_{shape_key}",
                        (x, y), (x + self.image_size, y + self.image_size),
                        tag=f"image_{shape_key}"
                    )
                else:

                    opacity = 0.96

                    for i in range(1, max_distance + 1):
                        if shape_key in distance_dict[i]:
                            opacity = 0.96 - (i - 1) * 1/max_distance
                            break


                    # Reduced opacity - draw with overlay
                    dpg.draw_image(
                        f"texture_{shape_key}",
                        (x, y), (x + self.image_size, y + self.image_size),
                        tag=f"image_{shape_key}"
                    )
                    # Add semi-transparent overlay to simulate reduced opacity
                    dpg.draw_rectangle(
                        (x, y), (x + self.image_size, y + self.image_size),
                        color=(0, 0, 0, int(opacity * 255)),
                        fill=(0, 0, 0, int(opacity * 255))
                    )
                
                # Add score text
                score = self.shapes_dict[shape_key]['score']
                dpg.draw_text(
                    (x, y + self.image_size + 5), 
                    f"{score:.2f}", 
                    color=(0, 0, 0, 255), 
                    size=12
                )
        
        # Draw connections
        # self.draw_connections(connected)
    
    def reset_graph_opacity(self):
        """Reset all graphs to full opacity"""
        # Redraw canvas without overlays
        dpg.delete_item("canvas")
        
        with dpg.drawlist(width=self.canvas_width, height=self.canvas_height, 
                         tag="canvas", parent=self.parent):
            # Add all graph images
            for shape_key, pos in self.graph_positions.items():
                x, y = pos
                
                # Create image
                dpg.draw_image(
                    f"texture_{shape_key}",
                    (x, y), (x + self.image_size, y + self.image_size),
                    tag=f"image_{shape_key}"
                )
                
                # Add score text
                score = self.shapes_dict[shape_key]['score']
                dpg.draw_text(
                    (x, y + self.image_size + 5), 
                    f"{score:.2f}", 
                    color=(0, 0, 0, 255), 
                    size=12
                )
    
    def draw_connections(self, connected_graphs: Set[str]):
        """Draw curved connections between selected graph and connected ones"""
        if not self.selected_graph or self.selected_graph not in self.graph_positions:
            return
            
        selected_pos = self.graph_positions[self.selected_graph]
        
        with dpg.draw_layer(tag="connections_layer", parent="canvas"):
            for connected_key in connected_graphs:
                if connected_key in self.graph_positions:
                    connected_pos = self.graph_positions[connected_key]
                    
                    # Draw curved line using bezier curve
                    start_x, start_y = selected_pos
                    end_x, end_y = connected_pos
                    
                    # Calculate control points for curve
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # Add some curve by offsetting control points
                    control_offset = 50
                    if abs(end_x - start_x) > abs(end_y - start_y):
                        # Horizontal curve
                        ctrl1_x, ctrl1_y = start_x + control_offset, start_y
                        ctrl2_x, ctrl2_y = end_x - control_offset, end_y
                    else:
                        # Vertical curve
                        ctrl1_x, ctrl1_y = start_x, start_y + control_offset
                        ctrl2_x, ctrl2_y = end_x, end_y - control_offset
                    
                    line_id = dpg.draw_bezier_cubic(
                        (start_x + 50, start_y + 50),  # Center of image
                        (ctrl1_x + 50, ctrl1_y + 50),
                        (ctrl2_x + 50, ctrl2_y + 50),
                        (end_x + 50, end_y + 50),
                        color=(255, 100, 100, 200),
                        thickness=3
                    )
                    self.connection_lines.append(line_id)
    
    def on_mouse_click(self, sender, app_data):
        """Handle mouse clicks on the canvas"""
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        
        # Get the canvas position offset
        canvas_pos = dpg.get_item_rect_min("canvas")
        relative_x = mouse_x - canvas_pos[0]
        relative_y = mouse_y - canvas_pos[1]
        
        # Check which graph was clicked
        for shape_key, pos in self.graph_positions.items():
            x, y = pos
            # Use stored image size for hit detection
            if x <= relative_x <= x + self.image_size and y <= relative_y <= y + self.image_size:
                self.on_graph_click(None, None, shape_key)
                break

    def create_interface(self, num_layers: int = 12, image_size: int = 100, layer_spacing: int = 200, vertical_spacing: int = 120, skin: str = "default"):
        """Create the main interface"""
        self.image_size = image_size  # Store for hit detection
        
        if dpg.does_item_exist(self.parent):
            dpg.delete_item(self.parent, children_only=True)

        # Organize graphs into layers
        layers = self.organize_by_error_layers(num_layers)
        self.graph_positions = self.calculate_positions(layers, image_size, layer_spacing, vertical_spacing)
        
        # Create textures
        self.create_graph_textures(image_size, skin=skin)
        
        # Calculate window size
        max_x = max([pos[0] for pos in self.graph_positions.values()]) + image_size + 100
        max_y = max([pos[1] for pos in self.graph_positions.values()]) + image_size + 100
        self.canvas_width = max_x
        self.canvas_height = max_y
        
        # Create main window
        with dpg.drawlist(width=max_x, height=max_y, tag="canvas", parent=self.parent):
            # Add all graph images
            for shape_key, pos in self.graph_positions.items():
                x, y = pos
                
                # Create image
                image_id = dpg.draw_image(
                    f"texture_{shape_key}",
                    (x, y), (x + image_size, y + image_size),
                    tag=f"image_{shape_key}"
                )
                
                # Add score text
                score = self.shapes_dict[shape_key]['score']
                dpg.draw_text(
                    (x, y + image_size + 5), 
                    f"{score:.2f}", 
                    color=(0, 0, 0, 255), 
                    size=12
                )
        
        # Add mouse click handler to the main window
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self.on_mouse_click)
        
        # Instructions
        # with dpg.window(label="Instructions", width=300, height=150, pos=(10, 10)):
        #     dpg.add_text("Click on any graph to see its connections")
        #     dpg.add_text("Connected graphs will remain visible")
        #     dpg.add_text("Others will become transparent")
        #     dpg.add_text("Click the same graph again to deselect")
    
    def run(self, num_layers: int = 6, image_size: int = 60, layer_spacing: int = 200, vertical_spacing: int = 120, skin: str = "default"):
        """Create the visualizer interface in the given parent window."""
        self.create_interface(num_layers, image_size, layer_spacing, vertical_spacing, skin)


# Usage example:
def spectral_decomposition(shapes_dict, transition_history, get_image_func, 
                           targets, dist_threshold, parent, num_layers=12, image_size=60, layer_spacing=200, vertical_spacing=120, skin="default"):
    """
    Create the graph network visualizer in the given parent window.
    """
    visualizer = GraphNetworkVisualizer(
        shapes_dict, transition_history, get_image_func, targets, dist_threshold, parent
    )
    visualizer.create_interface(num_layers, image_size, layer_spacing, vertical_spacing, skin)
    return visualizer  # Optionnel, si tu veux manipuler l'objet ensuite


# Example usage (uncomment when you have your data):
"""
# After your existing code:
# graphs_histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, 0.1, 10000, 10, False)
# gd, gh = histories_to_shapes_dict_and_transition_history(graphs_histories, targets, dist_threshold)
# gd, gh = filter_shapes_dict(gd, gh, lambda shape_key, info: info['score'] < 3.2)

# Run the visualizer:
spectral_decomposition(
    shapes_dict=gd,
    transition_history=gh,
    get_image_func=get_mini_graph_image,
    targets=targets,
    dist_threshold=dist_threshold,
    parent="spectra_window",
    num_layers=6,
    image_size=100,
    layer_spacing=200,
    vertical_spacing=120
)
"""


def filter_shapes_dict(shapes_dict, transition_history, filter_func=lambda shape_key, info: True):

        new_shapes_dict = {}
        new_shapes_transition_history = set()

        for shape_key, info in shapes_dict.items():
            if filter_func(shape_key, info):
                new_shapes_dict[shape_key] = info

        for transition in transition_history:
            if transition[0] in new_shapes_dict and transition[1] in new_shapes_dict:
                new_shapes_transition_history.add(transition)

        return new_shapes_dict, new_shapes_transition_history

def histories_to_shapes_dict_and_transition_history(histories, targets, dist_threshold):
        shapes_dict = {}
        transition_history = set()
        for history in histories:
            gh = gx.get_shape_string_transition_history(history, dist_threshold)
            gd = gx.decompose_history_by_shape(history, targets, dist_threshold)
            for shape_key, info in gd.items():
                if shape_key not in shapes_dict:
                    shapes_dict[shape_key] = info
                else:
                    if info['score'] < shapes_dict[shape_key]['score']:
                        shapes_dict[shape_key] = info
            transition_history.update(gh)
        return shapes_dict, transition_history



if __name__ == "__main__":

    from getimg import get_mini_graph_image

    targets = np.array([
        np.array([0,0]),
        np.array([3,0]),
        np.array([0,3]),
        np.array([3,4]),
        np.array([3.2,3.4]),
        np.array([1.5,2.2]),
        np.array([3.8,1.2]),
    ]).astype(np.float32)

    # nodes at the barycenter of the targets
    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))

    dist_threshold = 1.1

    graphs_histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, 0.1, 100000,100,False)

    gd,gh = histories_to_shapes_dict_and_transition_history(graphs_histories, targets, dist_threshold)
    gd,gh = filter_shapes_dict(gd, gh, lambda shape_key, info: info['score'] < 3)


    # Run the visualizer:
    spectral_decomposition(
        shapes_dict=gd,
        transition_history=gh,
        get_image_func=get_mini_graph_image,
        targets=targets,
        dist_threshold=dist_threshold,
        parent="spectra_window",
        num_layers=10,
        image_size=100,
        layer_spacing=120,
        vertical_spacing=120,
        skin="stick_dark"
    )
