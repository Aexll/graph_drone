import numpy as np
import dearpygui.dearpygui as dpg
import os
from drone import Drone
from target import Target


class WorldConfiguration:
    """Configuration for the world, such as map size and bounds."""
    MAP_SIZE = np.array([800, 800], dtype=np.float32)
    BOUNDS = np.array([[0, 0], MAP_SIZE], dtype=np.float32)
    DRONES: list[tuple] = [(300, 400),
                           (400, 400),
                           (500, 400)]
    TARGETS: list[tuple|None] = [(100, 100),
                None,
                None]
    
    def __init__(self, map_size:np.ndarray = MAP_SIZE, bounds:np.ndarray = BOUNDS, drones:list[tuple|None] = DRONES, targets:list[tuple|None] = TARGETS):
        self.map_size = map_size
        self.bounds = bounds
        self.drones = drones
        self.targets = targets






class World:

    MAP_SIZE = np.array([800, 800], dtype=np.float32)

    def __init__(self, config: WorldConfiguration|None = None, parent=None, bounds: np.ndarray = None):
        self.parent = parent
        self.bounds = bounds if bounds is not None else np.array([[0, 0], self.MAP_SIZE])
        self.drones: list['Drone'] = []
        self.targets: list['Target'] = []
        
        if config:
            self.map_size = config.map_size
            self.bounds = config.bounds
            
            # Validation des listes de configuration
            max_len = max(len(config.drones), len(config.targets))
            drones_list = config.drones + [None] * (max_len - len(config.drones))
            targets_list = config.targets + [None] * (max_len - len(config.targets))
            
            for i, (drone_pos, target_pos) in enumerate(zip(drones_list, targets_list)):
                if drone_pos is not None:
                    try:
                        drone = Drone(position=np.array(drone_pos), target=None, 
                                    scan_for_drones_method=self.get_drones_in_range, 
                                    scan_for_targets_method=self.get_targets_in_range)
                        if target_pos is not None:
                            target = Target(position=np.array(target_pos), target_type=0, bounds=self.bounds)
                            drone.target = target
                            self.add_target(target)
                        self.add_drone(drone)
                    except Exception as e:
                        print(f"Error creating drone {i}: {e}")

        self.global_time: float = 0.0
        self.delta_time: float = 0.08  # Time step for updates, can be adjusted as needed


    def draw(self, draw_list):
        for drone in self.drones:
            drone.draw(draw_list)
        for target in self.targets:
            target.draw(draw_list)

    def update(self):
        for target in self.targets:
            target.update(self.delta_time)
        for drone in self.drones:
            drone.update(self.delta_time)
        
        # Exécuter l'algorithme xi-omega distribué
        # self.xi_omega_step()
    
    def xi_omega_step(self):
        """Exécute une étape de l'algorithme xi-omega distribué pour tous les drones"""
        # Phase 1: Tous les drones préparent leurs messages
        all_messages = {}
        for drone in self.drones:
            messages = drone.prepare_messages()
            all_messages[drone] = messages
        
        # Phase 2: Distribuer les messages
        for sender, messages in all_messages.items():
            for receiver, message in messages.items():
                receiver.receive_message(sender, message)
        
        # Phase 3: Tous les drones mettent à jour leur état
        for drone in self.drones:
            drone.update_xi_omega()

    def add_drone(self, drone:Drone):
        self.drones.append(drone)

    def add_target(self, target:Target):
        self.targets.append(target)

    def remove_drone(self, idx: int):
        """Remove a drone by index with proper error handling."""
        if 0 <= idx < len(self.drones):
            drone = self.drones.pop(idx)
            # Supprimer les références au drone dans les connexions des autres drones
            drone.destroy()  # Nettoyage local
            self._cleanup_drone_references(drone)  # Nettoyage global
        else:
            print(f"Error: Invalid drone index {idx}. Valid range: 0-{len(self.drones)-1}")

    def remove_target(self, idx: int):
        """Remove a target by index with proper error handling."""
        if 0 <= idx < len(self.targets):
            target = self.targets.pop(idx)
            # Supprimer les références au target dans les drones
            for drone in self.drones:
                if drone.target == target:
                    drone.target = None
        else:
            print(f"Error: Invalid target index {idx}. Valid range: 0-{len(self.targets)-1}")

    def add_drone_with_target(self, drone:Drone, target:Target):
        """Add a drone with a target."""
        drone.target = target
        if drone not in self.drones:
            drone.scan_for_drones_method = self.get_drones_in_range
            drone.scan_for_targets_method = self.get_targets_in_range
            drone.targets = self.targets
            self.add_drone(drone)
        if target not in self.targets:
            self.add_target(target)
    
    def generate_random_drone(self) -> Drone:
        # Utiliser les bounds correctement pour générer une position
        position = np.random.rand(2) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return Drone(position=position, target=None, 
                    scan_for_drones_method=self.get_drones_in_range, 
                    scan_for_targets_method=self.get_targets_in_range)

    def generate_random_target(self) -> Target:
        # Utiliser les bounds correctement pour générer une position
        position = np.random.rand(2) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        target_type = np.random.choice([0, 1, 2])  # Inclure le type 2 (dynamic)
        return Target(position=position, target_type=target_type, bounds=self.bounds)

    def generate_random_drone_with_target(self):
        """Generate a random drone with a target."""
        drone = self.generate_random_drone()
        target = self.generate_random_target()
        drone.target = target
        if drone not in self.drones:
            drone.scan_for_drones_method = self.get_drones_in_range
            drone.scan_for_targets_method = self.get_targets_in_range
            drone.targets = self.targets
            self.add_drone(drone)
        if target not in self.targets:
            self.add_target(target)

    # world query methods
    def get_drones_in_range(self, position:np.ndarray, range:float) -> list[Drone]:
        """Get all drones within a certain range of a position."""
        return [drone for drone in self.drones if np.linalg.norm(drone.position - position) < range]
    
    def get_targets_in_range(self, position:np.ndarray, range:float) -> list[Target]:
        """Get all targets within a certain range of a position."""
        return [target for target in self.targets if np.linalg.norm(target.position - position) < range]
    
    def _cleanup_drone_references(self, removed_drone):
        """Nettoie toutes les références au drone supprimé dans les autres drones."""
        for drone in self.drones:
            # Retirer des connexions (normalement déjà fait)
            if removed_drone in drone.connections:
                drone.connections.remove(removed_drone)
            # Retirer des drones connus
            if removed_drone in drone.known_drones:
                drone.known_drones.discard(removed_drone)
            # Retirer des dictionnaires xi et omega
            if removed_drone in drone.xi:
                del drone.xi[removed_drone]
            if removed_drone in drone.omega:
                del drone.omega[removed_drone]
            # Nettoyer les messages en attente
            if removed_drone in drone.incoming_messages:
                del drone.incoming_messages[removed_drone]

if __name__ == "__main__":
    dpg.create_context()

    # === Load Unicode Font ===
    font_dir = os.path.dirname(__file__)
    font_path = os.path.join(font_dir, "FiraMonoNerdFont-Regular.otf")  # Place FiraCode-Regular.ttf in the same folder as this script
    with dpg.font_registry():
        if os.path.exists(font_path):
            unicode_font = dpg.add_font(font_path, 18)
            dpg.bind_font(unicode_font)
        else:
            print(f"Warning: {font_path} not found. Unicode characters may not display correctly.")
    dpg.create_viewport(title='World', width=1400, height=1000)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    hovered_drone = None

    def on_drawlist_hover(sender, app_data, user_data):
        """Handle hover events on the draw list."""
        global hovered_drone
        mouse_pos = dpg.get_mouse_pos()
        hovered_drone = None
        for drone in world.drones:
            if np.linalg.norm(drone.position - np.array(mouse_pos)) < drone.drone_radius*3:
                # print(f"Hovered over drone at {drone.position}")
                hovered_drone = drone
    
    def on_drawlist_click(sender, app_data, user_data):
        """Handle click events on the draw list."""
        global hovered_drone
        if hovered_drone is not None:
            print(f"=== Clicked on Drone at {hovered_drone.position} ===")
            hovered_drone.print_local_state()
            print("=" * 50)
            hovered_drone.update_xi_omega()


    with dpg.window(label="Simulation", width=800, height=800) as sim_window:
        draw_list = dpg.add_drawlist(width=800, height=800)
        with dpg.item_handler_registry() as drawlist_handler_registry:
            dpg.add_item_hover_handler(callback=on_drawlist_hover)
            dpg.add_item_clicked_handler(callback=on_drawlist_click)
        dpg.bind_item_handler_registry(draw_list, drawlist_handler_registry)
    
    # Information window
    with dpg.window(label="Algorithm Info", width=400, height=800, pos=(1000, 0)):
        dpg.add_text("Xi-Omega Algorithm Status")
        dpg.add_separator()
        
        status_text = dpg.add_text("Initializing...")
        drone_info_text = dpg.add_text("")
        
        dpg.add_separator()
        dpg.add_text("Network Topology:")
        topology_text = dpg.add_text("")
        
        dpg.add_separator()
        dpg.add_text("Convergence Progress:")
        convergence_text = dpg.add_text("")
    
    with dpg.window(label="Controls", width=200, height=800, pos=(800, 0)):
        dpg.add_text("Xi-Omega Controls")
        dpg.add_separator()
        
        # Basic controls
        dpg.add_button(label="Add Drone", callback=lambda: world.add_drone(Drone(position=np.random.rand(2) * 700, target=None, scan_for_drones_method=world.get_drones_in_range, scan_for_targets_method=world.get_targets_in_range)))
        dpg.add_button(label="Add Target", callback=lambda: world.add_target(Target(position=np.random.rand(2) * 700, target_type=1, bounds=np.array([[0, 0], world.MAP_SIZE]))))
        dpg.add_button(label="Add Drone with Target", callback=lambda: world.generate_random_drone_with_target())
        dpg.add_button(label="Remove Drone", callback=lambda: world.remove_drone(0) if world.drones else None)
        dpg.add_button(label="Remove Target", callback=lambda: world.remove_target(0) if world.targets else None)
        
        dpg.add_separator()
        dpg.add_text("Xi-Omega Algorithm")
        
        # Algorithm control
        def reset_algorithm():
            for drone in world.drones:
                drone.xi.clear()
                drone.omega.clear()
                drone.xi[drone] = 1.0
                drone.omega[drone] = 0.0
                drone.known_drones = {drone}
                drone.iteration = 0
                drone.incoming_messages.clear()
                drone.previous_xi.clear()
                drone.previous_omega.clear()
        
        def step_algorithm():
            for drone in world.drones:
                drone.xi_omega_step()
        
        def print_convergence_status():
            converged = sum(1 for drone in world.drones if drone.has_converged())
            total = len(world.drones)
            print(f"Convergence: {converged}/{total} drones converged")
            for i, drone in enumerate(world.drones):
                print(f"  Drone {i}: iteration {drone.iteration}, converged: {drone.has_converged()}")
        
        dpg.add_button(label="Reset Xi-Omega", callback=reset_algorithm)
        dpg.add_button(label="Step Algorithm", callback=step_algorithm)
        dpg.add_button(label="Print Status", callback=print_convergence_status)
        
        # Display options
        dpg.add_separator()
        dpg.add_text("Display Options")
        show_ranges = dpg.add_checkbox(label="Show Ranges", default_value=True)
        show_xi_values = dpg.add_checkbox(label="Show Xi Values", default_value=True)
        show_omega_values = dpg.add_checkbox(label="Show Omega Values", default_value=False)
        show_iterations = dpg.add_checkbox(label="Show Iterations", default_value=True)

    config = WorldConfiguration(
        map_size=np.array([800, 800]),
        bounds=np.array([[0, 0], [800, 800]]),
        drones=[(150, 150),
                (250, 150),
                (350, 150),
                (450, 150),
                (250, 250),
                ],
        targets=[None,
                 None,
                 None,
                 None,
                 (400, 400),
                ]
        )
    
    world = World(config=config, bounds=np.array([[0, 0], [800, 800]]), parent=None)

    # Establish some initial connections for testing
    if len(world.drones) >= 4:
        world.drones[0].add_connection(world.drones[1])
        world.drones[1].add_connection(world.drones[2])
        world.drones[2].add_connection(world.drones[3])
        if len(world.drones) >= 5:
            world.drones[1].add_connection(world.drones[4])  # Create a branch

    focusing_drone = world.drones[0] if world.drones else None

    while dpg.is_dearpygui_running():
        dpg.delete_item(draw_list, children_only=True)
        world.global_time += world.delta_time
        world.update()
        world.draw(draw_list)
        
        # hovered drone
        if hovered_drone is not None:
            hovered_drone.draw_notify("hover", draw_list=draw_list)
            
            # Show xi-omega information for hovered drone
            if dpg.get_value(show_xi_values):
                for i, (drone, xi_value) in enumerate(hovered_drone.xi.items()):
                    if drone != hovered_drone and xi_value > 0:
                        # Draw line to known drones with xi value
                        # dpg.draw_line(hovered_drone.position, drone.position, 
                        #             color=(255, 255, 0, 100), thickness=2, parent=draw_list)
                        # Show xi value
                        # mid_point = (hovered_drone.position + drone.position) / 2
                        # dpg.draw_text(mid_point, f"{xi_value:.0f}", 
                                    # color=(255, 255, 0), size=20, parent=draw_list)
                        dpg.draw_circle(drone.position, radius=hovered_drone.drone_radius*3,
                                    color=(255, 255, 0, 100), parent=draw_list)
                        text_pos = drone.position + np.array([0, 15])
                        dpg.draw_text(text_pos, f"{xi_value:.0f}", 
                                    color=(255, 255, 0), size=20, parent=draw_list)
            # Show omega information for hovered drone
            if dpg.get_value(show_omega_values):
                for drone, omega_value in hovered_drone.omega.items():
                    if drone != hovered_drone and omega_value != float('inf'):
                        # Show omega value below drone
                        text_pos = drone.position + np.array([0, 15])
                        dpg.draw_text(text_pos, f"{omega_value:.0f}", 
                                    color=(0, 255, 255), size=20, parent=draw_list)

        # Global information display
        dpg.draw_text((10, 10), f"Global Time: {world.global_time:.2f}", color=(255, 255, 255), parent=draw_list, size=20)
        
        # Convergence status
        converged_count = sum(1 for drone in world.drones if drone.has_converged())
        total_drones = len(world.drones)
        dpg.draw_text((10, 30), f"Converged: {converged_count}/{total_drones}", 
                     color=(0, 255, 0) if converged_count == total_drones else (255, 255, 0), 
                     parent=draw_list
                     , size=20)
        
        # Show iterations for each drone
        if dpg.get_value(show_iterations):
            for i, drone in enumerate(world.drones):
                iteration_pos = drone.position + np.array([-15, -25])
                color = (0, 255, 0) if drone.has_converged() else (255, 255, 255)
                dpg.draw_text(iteration_pos, f"#{drone.iteration}", 
                            color=color, size=15, parent=draw_list)
        
        # Show known drones count
        for drone in world.drones:
            known_count = len(drone.known_drones) - 1  # Exclude self
            count_pos = drone.position + np.array([15, -25])
            dpg.draw_text(count_pos, f"K:{known_count}", 
                        color=(255, 200, 100), size=10, parent=draw_list)

        # Instructions for the user
        dpg.draw_text((10, 70), "Instructions:", color=(200, 200, 200), parent=draw_list , size=15)
        dpg.draw_text((10, 90), "> Hover over drones to see xi-omega info", color=(200, 200, 200), parent=draw_list , size=15)
        dpg.draw_text((10, 110), "> Click on drones to print detailed state", color=(200, 200, 200), parent=draw_list , size=15)
        dpg.draw_text((10, 130), "> Use controls to manage algorithm", color=(200, 200, 200), parent=draw_list , size=15)

        # Update info window
        converged_count = sum(1 for drone in world.drones if drone.has_converged())
        total_drones = len(world.drones)
        
        # Status update
        if total_drones == 0:
            status = "No drones in simulation"
        elif converged_count == total_drones:
            status = f"[v] Algorithm CONVERGED! ({total_drones} drones)"
        else:
            status = f"[s] Running... ({converged_count}/{total_drones} converged)"
        dpg.set_value(status_text, status)
        
        # Drone info update
        if hovered_drone:
            drone_idx = world.drones.index(hovered_drone)
            known_count = len(hovered_drone.known_drones) - 1
            drone_info = f"Selected: Drone {drone_idx}\n"
            drone_info += f"Position: ({hovered_drone.position[0]:.0f}, {hovered_drone.position[1]:.0f})\n"
            drone_info += f"Connections: {len(hovered_drone.connections)}\n"
            drone_info += f"Known drones: {known_count}\n"
            drone_info += f"Iteration: {hovered_drone.iteration}\n"
            drone_info += f"Converged: {'Yes' if hovered_drone.has_converged() else 'No'}"
        else:
            drone_info = "Hover over a drone to see details"
        dpg.set_value(drone_info_text, drone_info)
        
        # Topology info
        topology_info = f"Total drones: {total_drones}\n"
        topology_info += f"Total connections: {sum(len(d.connections) for d in world.drones) // 2}\n"
        if world.drones:
            avg_connections = sum(len(d.connections) for d in world.drones) / len(world.drones)
            topology_info += f"Avg connections/drone: {avg_connections:.1f}\n"
            max_known = max(len(d.known_drones) - 1 for d in world.drones)
            topology_info += f"Max known drones: {max_known}"
        dpg.set_value(topology_text, topology_info)
        
        # Convergence progress
        if world.drones:
            iterations = [d.iteration for d in world.drones]
            convergence_info = f"Min iterations: {min(iterations)}\n"
            convergence_info += f"Max iterations: {max(iterations)}\n"
            convergence_info += f"Avg iterations: {sum(iterations)/len(iterations):.1f}\n"
            convergence_info += f"Progress: {converged_count}/{total_drones} ({100*converged_count/total_drones:.1f}%)"
        else:
            convergence_info = "No drones to analyze"
        dpg.set_value(convergence_text, convergence_info)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()