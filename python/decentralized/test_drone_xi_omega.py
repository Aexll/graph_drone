#!/usr/bin/env python3
"""
Test script for the xi-omega algorithm implementation in drone.py
"""

import numpy as np
import time
from drone import Drone
from target import Target


def create_simple_network():
    """Create a simple network of drones to test xi-omega algorithm"""
    
    # Create targets (can be None for this test)
    bounds = np.array([400, 400])  # Simple bounds for targets
    target1 = Target(np.array([100, 100]), 0, bounds)  # Static target
    target2 = Target(np.array([200, 200]), 0, bounds)  # Static target
    
    # Create drones in a line: drone1 -- drone2 -- drone3 -- drone4
    drone1 = Drone(np.array([50, 50]), target1)
    drone2 = Drone(np.array([150, 50]), None)
    drone3 = Drone(np.array([250, 50]), None)
    drone4 = Drone(np.array([350, 50]), target2)
    
    all_drones = [drone1, drone2, drone3, drone4]
    
    # Set up scan methods for each drone
    def make_scan_method(drones_list):
        def scan_for_drones(position, range_limit):
            candidates = []
            for drone in drones_list:
                distance = np.linalg.norm(drone.position - position)
                if distance <= range_limit:
                    candidates.append(drone)
            return candidates
        return scan_for_drones
    
    def scan_for_targets(position, range_limit):
        return []  # No target scanning for this test
    
    # Assign scan methods to all drones
    for drone in all_drones:
        drone.scan_for_drones = make_scan_method(all_drones)
        drone.scan_for_targets = scan_for_targets
    
    # Manually establish connections (line topology)
    drone1.add_connection(drone2)
    drone2.add_connection(drone3)
    drone3.add_connection(drone4)
    
    return all_drones


def simulate_xi_omega_steps(drones, max_steps=20):
    """Simulate xi-omega algorithm for a given number of steps"""
    
    print("=== Starting Xi-Omega Algorithm Simulation ===\n")
    
    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        
        # Execute one xi-omega step for all drones
        for drone in drones:
            drone.xi_omega_step()
        
        # Check convergence
        converged_count = sum(1 for drone in drones if drone.has_converged())
        print(f"Drones converged: {converged_count}/{len(drones)}")
        
        # Print state of first drone as example
        if step < 5 or step % 5 == 0:  # Print first few steps and every 5th step
            print(f"\nExample - Drone 1 state:")
            drones[0].print_local_state()
        
        # Check global convergence
        if converged_count == len(drones):
            print(f"\n🎉 Global convergence achieved after {step + 1} steps!")
            break
        
        time.sleep(0.1)  # Small delay for readability
    
    # Print final states
    print("\n" + "="*60)
    print("FINAL STATES")
    print("="*60)
    
    for i, drone in enumerate(drones):
        print(f"\nDrone {i+1}:")
        drone.print_local_state()


def test_xi_omega_basic():
    """Basic test of xi-omega algorithm"""
    
    print("Creating drone network...")
    drones = create_simple_network()
    
    print(f"Created {len(drones)} drones")
    print("Network topology: 1 -- 2 -- 3 -- 4")
    print(f"Drone positions: {[tuple(d.position) for d in drones]}")
    
    # Show initial state
    print("\n--- Initial States ---")
    for i, drone in enumerate(drones):
        print(f"Drone {i+1}: {len(drone.connections)} connections, knows {len(drone.known_drones)} drones")
    
    # Run simulation
    simulate_xi_omega_steps(drones, max_steps=15)
    
    # Analyze results
    print("\n--- Analysis ---")
    for i, drone in enumerate(drones):
        print(f"Drone {i+1}:")
        print(f"  - Knows {len(drone.known_drones)} total drones")
        print(f"  - Converged: {drone.has_converged()}")
        print(f"  - Iterations: {drone.iteration}")
        
        # Show distances to other drones
        distances = {}
        for other_drone in drone.known_drones:
            if other_drone != drone:
                dist = drone.omega.get(other_drone, float('inf'))
                distances[f"Drone{drones.index(other_drone)+1}"] = dist if dist != float('inf') else '∞'
        print(f"  - Distances: {distances}")


if __name__ == "__main__":
    test_xi_omega_basic()
