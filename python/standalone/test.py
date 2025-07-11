import dearpygui.dearpygui as dpg
import threading
import time
import random
import math
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import json

class MessageType(Enum):
    XI_OMEGA_UPDATE = "xi_omega_update"
    NETWORK_DISCOVERY = "network_discovery"
    CRITICAL_EDGE_CHECK = "critical_edge_check"
    HEARTBEAT = "heartbeat"
    NODE_COUNT_ESTIMATE = "node_count_estimate"

@dataclass
class Message:
    sender_id: int
    message_type: MessageType
    data: dict
    timestamp: float

class Drone:
    def __init__(self, drone_id: int, x: float, y: float):
        self.id = drone_id
        self.x = x
        self.y = y
        
        # Neighbors (directly connected drones)
        self.neighbors: Set[int] = set()
        
        # Message queue
        self.message_queue: List[Message] = []
        self.message_lock = threading.Lock()
        
        # Algorithm state
        self.xi = {}  # xi[j] = 1 if node j is reachable, 0 otherwise
        self.omega = {}  # omega[j] = distance to node j
        self.delta = {}  # For critical edge detection
        
        # Network state
        self.n_estimate = 1  # Estimated number of nodes
        self.known_nodes: Set[int] = {self.id}
        self.iteration = 0
        self.max_iterations = 50  # Safety limit
        
        # Critical edges
        self.critical_neighbors: Set[int] = set()
        
        # Convergence detection
        self.last_xi = {}
        self.last_omega = {}
        self.convergence_counter = 0
        self.convergence_threshold = 3  # Stable for 3 iterations
        
        # Initialize xi and omega
        self.reset_algorithm_state()
        
        # Status
        self.is_running = True
        
    def reset_algorithm_state(self):
        """Reset algorithm state for all known nodes"""
        for node_id in self.known_nodes:
            if node_id == self.id:
                self.xi[node_id] = 1
                self.omega[node_id] = 0
            else:
                self.xi[node_id] = 0
                self.omega[node_id] = float('inf')
    
    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor and update network state"""
        if neighbor_id not in self.neighbors:
            self.neighbors.add(neighbor_id)
            self.known_nodes.add(neighbor_id)
            
            # Initialize new node in xi and omega
            if neighbor_id not in self.xi:
                self.xi[neighbor_id] = 0
                self.omega[neighbor_id] = float('inf')
            
            # Update network size estimate
            self.n_estimate = max(self.n_estimate, len(self.known_nodes))
            
            # Send discovery message to inform about known nodes
            self.send_network_discovery()
    
    def remove_neighbor(self, neighbor_id: int):
        """Remove a neighbor"""
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
            # Don't remove from known_nodes as node might still be reachable
            # The algorithm will naturally update reachability
    
    def send_message(self, neighbor_id: int, message_type: MessageType, data: dict):
        """Send a message to a specific neighbor"""
        # This would be implemented by the network manager
        pass
    
    def receive_message(self, message: Message):
        """Receive a message from another drone"""
        with self.message_lock:
            self.message_queue.append(message)
    
    def process_messages(self):
        """Process all pending messages"""
        with self.message_lock:
            messages = self.message_queue.copy()
            self.message_queue.clear()
        
        for message in messages:
            self.handle_message(message)
    
    def handle_message(self, message: Message):
        """Handle a specific message based on its type"""
        if message.message_type == MessageType.XI_OMEGA_UPDATE:
            self.handle_xi_omega_update(message)
        elif message.message_type == MessageType.NETWORK_DISCOVERY:
            self.handle_network_discovery(message)
        elif message.message_type == MessageType.CRITICAL_EDGE_CHECK:
            self.handle_critical_edge_check(message)
        elif message.message_type == MessageType.NODE_COUNT_ESTIMATE:
            self.handle_node_count_estimate(message)
    
    def handle_xi_omega_update(self, message: Message):
        """Handle xi and omega updates from neighbors"""
        sender_xi = message.data.get('xi', {})
        sender_omega = message.data.get('omega', {})
        
        # Update known nodes
        for node_id in sender_xi.keys():
            if node_id not in self.known_nodes:
                self.known_nodes.add(node_id)
                self.xi[node_id] = 0
                self.omega[node_id] = float('inf')
        
        # Update network size estimate
        self.n_estimate = max(self.n_estimate, len(self.known_nodes))
    
    def handle_network_discovery(self, message: Message):
        """Handle network discovery messages"""
        sender_known_nodes = set(message.data.get('known_nodes', []))
        
        # Update known nodes
        new_nodes = sender_known_nodes - self.known_nodes
        for node_id in new_nodes:
            self.known_nodes.add(node_id)
            self.xi[node_id] = 0
            self.omega[node_id] = float('inf')
        
        # Update network size estimate
        self.n_estimate = max(self.n_estimate, len(self.known_nodes))
    
    def handle_critical_edge_check(self, message: Message):
        """Handle critical edge check messages"""
        pass  # Implement critical edge detection logic
    
    def handle_node_count_estimate(self, message: Message):
        """Handle node count estimate messages"""
        sender_estimate = message.data.get('n_estimate', 1)
        self.n_estimate = max(self.n_estimate, sender_estimate)
    
    def send_network_discovery(self):
        """Send network discovery message to all neighbors"""
        for neighbor_id in self.neighbors:
            self.send_message(neighbor_id, MessageType.NETWORK_DISCOVERY, {
                'known_nodes': list(self.known_nodes),
                'n_estimate': self.n_estimate
            })
    
    def update_xi_omega(self):
        """Update xi and omega values (Algorithm 1 from paper)"""
        if not self.is_running:
            return
        
        # Store previous values for convergence detection
        self.last_xi = self.xi.copy()
        self.last_omega = self.omega.copy()
        
        new_xi = {}
        new_omega = {}
        
        for j in self.known_nodes:
            if j == self.id:
                new_xi[j] = 1
                new_omega[j] = 0
            else:
                # Max consensus for xi
                max_xi = self.xi.get(j, 0)
                for neighbor_id in self.neighbors:
                    # This would get the neighbor's xi value through message passing
                    # For now, using current values
                    max_xi = max(max_xi, self.xi.get(j, 0))
                
                new_xi[j] = max_xi
                
                # Update omega based on xi changes
                if new_xi[j] > self.xi.get(j, 0):
                    # Node j became reachable, update distance
                    min_distance = float('inf')
                    for neighbor_id in self.neighbors:
                        # This would get the neighbor's omega value through message passing
                        neighbor_omega = self.omega.get(j, float('inf'))
                        if neighbor_omega != float('inf'):
                            min_distance = min(min_distance, neighbor_omega + 1)
                    new_omega[j] = min_distance if min_distance != float('inf') else float('inf')
                else:
                    new_omega[j] = self.omega.get(j, float('inf'))
        
        self.xi = new_xi
        self.omega = new_omega
        self.iteration += 1
        
        # Check for convergence
        self.check_convergence()
        
        # Send updates to neighbors
        self.send_xi_omega_update()
    
    def send_xi_omega_update(self):
        """Send xi and omega updates to all neighbors"""
        for neighbor_id in self.neighbors:
            self.send_message(neighbor_id, MessageType.XI_OMEGA_UPDATE, {
                'xi': self.xi,
                'omega': self.omega,
                'iteration': self.iteration
            })
    
    def check_convergence(self):
        """Check if algorithm has converged"""
        if self.xi == self.last_xi and self.omega == self.last_omega:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
    
    def has_converged(self):
        """Check if algorithm has converged"""
        return self.convergence_counter >= self.convergence_threshold
    
    def detect_critical_edges(self):
        """Detect critical edges using Algorithm 3 from paper"""
        if not self.has_converged():
            return
        
        self.critical_neighbors.clear()
        
        for neighbor_id in self.neighbors:
            if self.is_critical_edge(neighbor_id):
                self.critical_neighbors.add(neighbor_id)
    
    def is_critical_edge(self, neighbor_id: int) -> bool:
        """Check if edge to neighbor is critical"""
        # Simplified critical edge detection
        # In practice, this would implement the full algorithm from the paper
        
        # For now, just check if removing this edge would disconnect the network
        # This is a simplified heuristic
        return len(self.neighbors) == 1 and neighbor_id in self.neighbors
    
    def get_status(self) -> dict:
        """Get current status of the drone"""
        return {
            'id': self.id,
            'position': (self.x, self.y),
            'neighbors': list(self.neighbors),
            'xi': dict(self.xi),
            'omega': {k: v if v != float('inf') else -1 for k, v in self.omega.items()},
            'critical_neighbors': list(self.critical_neighbors),
            'n_estimate': self.n_estimate,
            'iteration': self.iteration,
            'converged': self.has_converged(),
            'known_nodes': list(self.known_nodes)
        }

class DroneNetworkManager:
    def __init__(self):
        self.drones: Dict[int, Drone] = {}
        self.connections: Set[Tuple[int, int]] = set()
        self.next_drone_id = 1
        self.is_running = False
        self.update_thread = None
        self.update_frequency = 2.0  # Updates per second
        
    def add_drone(self, x: float, y: float) -> int:
        """Add a new drone to the network"""
        drone_id = self.next_drone_id
        self.next_drone_id += 1
        
        drone = Drone(drone_id, x, y)
        self.drones[drone_id] = drone
        
        return drone_id
    
    def remove_drone(self, drone_id: int):
        """Remove a drone from the network"""
        if drone_id in self.drones:
            # Remove all connections to this drone
            connections_to_remove = []
            for conn in self.connections:
                if drone_id in conn:
                    connections_to_remove.append(conn)
            
            for conn in connections_to_remove:
                self.remove_connection(conn[0], conn[1])
            
            # Remove the drone
            del self.drones[drone_id]
    
    def add_connection(self, drone1_id: int, drone2_id: int):
        """Add a connection between two drones"""
        if drone1_id in self.drones and drone2_id in self.drones:
            conn = tuple(sorted([drone1_id, drone2_id]))
            if conn not in self.connections:
                self.connections.add(conn)
                self.drones[drone1_id].add_neighbor(drone2_id)
                self.drones[drone2_id].add_neighbor(drone1_id)
    
    def remove_connection(self, drone1_id: int, drone2_id: int):
        """Remove a connection between two drones"""
        conn = tuple(sorted([drone1_id, drone2_id]))
        if conn in self.connections:
            self.connections.remove(conn)
            self.drones[drone1_id].remove_neighbor(drone2_id)
            self.drones[drone2_id].remove_neighbor(drone1_id)
    
    def send_message(self, sender_id: int, receiver_id: int, message_type: MessageType, data: dict):
        """Send a message between drones"""
        if receiver_id in self.drones:
            message = Message(sender_id, message_type, data, time.time())
            self.drones[receiver_id].receive_message(message)
    
    def start_simulation(self):
        """Start the simulation"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop for the simulation"""
        while self.is_running:
            self.update_network()
            time.sleep(1.0 / self.update_frequency)
    
    def update_network(self):
        """Update the entire network for one iteration"""
        # Process messages for all drones
        for drone in self.drones.values():
            drone.process_messages()
        
        # Update xi and omega for all drones
        for drone in self.drones.values():
            drone.update_xi_omega()
        
        # Detect critical edges
        for drone in self.drones.values():
            drone.detect_critical_edges()
        
        # Override message sending for actual network communication
        for drone in self.drones.values():
            drone.send_message = lambda neighbor_id, msg_type, data, sender=drone.id: self.send_message(sender, neighbor_id, msg_type, data)
    
    def get_network_status(self) -> dict:
        """Get the status of the entire network"""
        return {
            'drones': {drone_id: drone.get_status() for drone_id, drone in self.drones.items()},
            'connections': list(self.connections),
            'total_drones': len(self.drones)
        }

# Global network manager
network_manager = DroneNetworkManager()

def create_gui():
    """Create the GUI for the drone network visualization"""
    dpg.create_context()
    
    # Global state
    selected_drone = None
    drag_mode = False
    
    def add_drone_callback():
        x = random.uniform(50, 750)
        y = random.uniform(50, 550)
        drone_id = network_manager.add_drone(x, y)
        update_display()
    
    def remove_drone_callback():
        if selected_drone is not None:
            network_manager.remove_drone(selected_drone)
            update_display()
    
    def start_simulation_callback():
        network_manager.start_simulation()
        dpg.set_item_label("start_btn", "Simulation Running")
        dpg.configure_item("start_btn", enabled=False)
        dpg.configure_item("stop_btn", enabled=True)
    
    def stop_simulation_callback():
        network_manager.stop_simulation()
        dpg.set_item_label("start_btn", "Start Simulation")
        dpg.configure_item("start_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)
    
    def update_display():
        """Update the visual display"""
        dpg.delete_item("drawing_canvas", children_only=True)
        
        status = network_manager.get_network_status()
        
        # Draw connections
        for conn in status['connections']:
            drone1_id, drone2_id = conn
            if drone1_id in status['drones'] and drone2_id in status['drones']:
                drone1 = status['drones'][drone1_id]
                drone2 = status['drones'][drone2_id]
                
                # Check if connection is critical
                is_critical = (drone2_id in drone1['critical_neighbors'] or 
                              drone1_id in drone2['critical_neighbors'])
                
                color = [255, 0, 0] if is_critical else [100, 100, 100]
                thickness = 3 if is_critical else 1
                
                dpg.draw_line(
                    parent="drawing_canvas",
                    p1=drone1['position'],
                    p2=drone2['position'],
                    color=color,
                    thickness=thickness
                )
        
        # Draw drones
        for drone_id, drone_data in status['drones'].items():
            x, y = drone_data['position']
            
            # Color based on convergence status
            if drone_data['converged']:
                color = [0, 255, 0]  # Green for converged
            else:
                color = [255, 255, 0]  # Yellow for not converged
            
            dpg.draw_circle(
                parent="drawing_canvas",
                center=(x, y),
                radius=15,
                color=color,
                fill=color,
                tag=f"drone_{drone_id}"
            )
            
            # Label with drone ID
            dpg.draw_text(
                parent="drawing_canvas",
                pos=(x-5, y-5),
                text=str(drone_id),
                color=[0, 0, 0],
                size=12
            )
        
        # Update status panel
        update_status_panel()
    
    def update_status_panel():
        """Update the status information panel"""
        dpg.delete_item("status_panel", children_only=True)
        
        status = network_manager.get_network_status()
        
        with dpg.group(parent="status_panel"):
            dpg.add_text(f"Total Drones: {status['total_drones']}")
            dpg.add_text(f"Total Connections: {len(status['connections'])}")
            
            if selected_drone and selected_drone in status['drones']:
                drone_data = status['drones'][selected_drone]
                dpg.add_separator()
                dpg.add_text(f"Selected Drone: {selected_drone}")
                dpg.add_text(f"Position: ({drone_data['position'][0]:.1f}, {drone_data['position'][1]:.1f})")
                dpg.add_text(f"Neighbors: {drone_data['neighbors']}")
                dpg.add_text(f"Network Size Estimate: {drone_data['n_estimate']}")
                dpg.add_text(f"Iteration: {drone_data['iteration']}")
                dpg.add_text(f"Converged: {drone_data['converged']}")
                dpg.add_text(f"Critical Neighbors: {drone_data['critical_neighbors']}")
                
                # Xi values
                dpg.add_separator()
                dpg.add_text("Xi values (reachability):")
                for node_id, xi_val in drone_data['xi'].items():
                    dpg.add_text(f"  Node {node_id}: {xi_val}")
                
                # Omega values
                dpg.add_text("Omega values (distances):")
                for node_id, omega_val in drone_data['omega'].items():
                    dist_str = str(omega_val) if omega_val != -1 else "∞"
                    dpg.add_text(f"  Node {node_id}: {dist_str}")
    
    def canvas_click_callback(sender, app_data):
        nonlocal selected_drone
        mouse_pos = dpg.get_mouse_pos(local=False)
        canvas_pos = dpg.get_item_pos("drawing_canvas")
        relative_pos = (mouse_pos[0] - canvas_pos[0], mouse_pos[1] - canvas_pos[1])
        
        # Check if we clicked on a drone
        clicked_drone = None
        status = network_manager.get_network_status()
        for drone_id, drone_data in status['drones'].items():
            x, y = drone_data['position']
            distance = math.sqrt((relative_pos[0] - x)**2 + (relative_pos[1] - y)**2)
            if distance <= 15:  # Within drone radius
                clicked_drone = drone_id
                break
        
        if clicked_drone:
            if selected_drone is None:
                selected_drone = clicked_drone
            elif selected_drone == clicked_drone:
                selected_drone = None
            else:
                # Create/remove connection
                if (selected_drone, clicked_drone) in network_manager.connections or \
                   (clicked_drone, selected_drone) in network_manager.connections:
                    network_manager.remove_connection(selected_drone, clicked_drone)
                else:
                    network_manager.add_connection(selected_drone, clicked_drone)
                selected_drone = None
        else:
            selected_drone = None
        
        update_display()
    
    # Create main window
    with dpg.window(label="Distributed Drone Network", width=1200, height=800, tag="main_window"):
        with dpg.menu_bar():
            with dpg.menu(label="Network"):
                dpg.add_menu_item(label="Add Drone", callback=add_drone_callback)
                dpg.add_menu_item(label="Remove Selected Drone", callback=remove_drone_callback)
                dpg.add_separator()
                dpg.add_menu_item(label="Start Simulation", callback=start_simulation_callback, tag="start_btn")
                dpg.add_menu_item(label="Stop Simulation", callback=stop_simulation_callback, tag="stop_btn", enabled=False)
        
        with dpg.group(horizontal=True):
            # Drawing area
            with dpg.child_window(width=800, height=600, tag="canvas_container"):
                with dpg.drawlist(width=800, height=600, tag="drawing_canvas"):
                    pass
                
                # Handle mouse clicks
                with dpg.handler_registry():
                    dpg.add_mouse_click_handler(callback=canvas_click_callback)
            
            # Status panel
            with dpg.child_window(width=350, height=600, tag="status_panel"):
                dpg.add_text("Network Status")
                dpg.add_separator()
        
        # Instructions
        dpg.add_text("Instructions:")
        dpg.add_text("• Click 'Add Drone' to add a new drone at random position")
        dpg.add_text("• Click a drone to select it (becomes highlighted)")
        dpg.add_text("• Click another drone to create/remove connection")
        dpg.add_text("• Click 'Remove Selected Drone' to remove selected drone")
        dpg.add_text("• Green drones have converged, yellow are still computing")
        dpg.add_text("• Red connections are critical edges")
    
    # Setup viewport
    dpg.create_viewport(title="Distributed Drone Network", width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # Update display periodically
    def update_loop():
        while dpg.is_dearpygui_running():
            update_display()
            time.sleep(0.5)
    
    # Start update thread
    update_thread = threading.Thread(target=update_loop)
    update_thread.daemon = True
    update_thread.start()
    
    # Start DearPyGui
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    create_gui()