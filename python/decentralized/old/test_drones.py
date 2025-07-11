#!/usr/bin/env python3
"""
Test simple pour vérifier que l'algorithme xi-omega fonctionne avec les drones
"""

import numpy as np
import sys
import os

# Ajouter le chemin parent pour importer les modules
sys.path.append(os.path.dirname(__file__))

from drone import Drone
from target import Target
from world import World, WorldConfiguration


def test_simple_drone_network():
    """
    Test simple avec 3 drones en ligne pour vérifier l'algorithme xi-omega
    """
    print("=== TEST SIMPLE DU RÉSEAU DE DRONES ===\n")
    
    # Configuration simple : 3 drones en ligne
    config = WorldConfiguration(
        map_size=np.array([400, 400]),
        bounds=np.array([[0, 0], [400, 400]]),
        drones=[(100, 200), (200, 200), (300, 200)],  # 3 drones alignés
        targets=[None, None, None]
    )
    
    world = World(config=config)
    
    print(f"Monde créé avec {len(world.drones)} drones")
    for i, drone in enumerate(world.drones):
        print(f"  Drone {i}: position {drone.position}")
    
    # Établir des connexions manuellement (drone 0 <-> drone 1 <-> drone 2)
    if len(world.drones) >= 2:
        world.drones[0].add_connection(world.drones[1])
        print("Connexion établie: Drone 0 <-> Drone 1")
    
    if len(world.drones) >= 3:
        world.drones[1].add_connection(world.drones[2])
        print("Connexion établie: Drone 1 <-> Drone 2")
    
    print("\n--- ÉTAT INITIAL ---")
    print_drone_states(world.drones)
    
    # Simuler l'algorithme xi-omega étape par étape
    print("\n--- SIMULATION ÉTAPE PAR ÉTAPE ---")
    
    for step in range(6):  # Suffisant pour la convergence
        print(f"\n>>> ÉTAPE {step + 1} <<<")
        
        # Phase 1: Préparer les messages
        all_messages = {}
        for i, drone in enumerate(world.drones):
            messages = drone.prepare_messages()
            all_messages[drone] = messages
            if messages:
                neighbor_indices = [world.drones.index(neighbor) for neighbor in messages.keys()]
                print(f"  Drone {i} envoie à: {neighbor_indices}")
        
        # Phase 2: Distribuer les messages
        for sender, messages in all_messages.items():
            for receiver, message in messages.items():
                receiver.receive_message(sender, message)
        
        # Phase 3: Mettre à jour tous les drones
        for drone in world.drones:
            drone.update_xi_omega()
        
        # Afficher l'état après cette étape
        print("\nÉtat après cette étape:")
        for i, drone in enumerate(world.drones):
            known_indices = [world.drones.index(d) for d in drone.known_drones if d != drone]
            xi_values = {world.drones.index(d): v for d, v in drone.xi.items() if d != drone}
            omega_values = {world.drones.index(d): v for d, v in drone.omega.items() if d != drone and v != float('inf')}
            print(f"  Drone {i}: connaît {known_indices}, xi={xi_values}, omega={omega_values}")
        
        # Vérifier convergence
        converged = all(drone.has_converged() for drone in world.drones)
        if converged:
            print(f"\n*** CONVERGENCE ATTEINTE après {step + 1} étapes ***")
            break
    
    print("\n--- ÉTAT FINAL DÉTAILLÉ ---")
    print_drone_states(world.drones)


def print_drone_states(drones):
    """Affiche l'état détaillé de tous les drones"""
    for i, drone in enumerate(drones):
        print(f"\n--- Drone {i} ---")
        print(f"  Position: {drone.position}")
        print(f"  Connexions: {[drones.index(c) for c in drone.connections]}")
        print(f"  Itération: {drone.iteration}")
        print(f"  Convergé: {drone.has_converged()}")
        
        # Afficher xi et omega pour les autres drones
        known_others = [d for d in drone.known_drones if d != drone]
        if known_others:
            print("  Xi (connectivité):")
            for other in known_others:
                other_idx = drones.index(other)
                xi_val = drone.xi.get(other, 0.0)
                print(f"    vers Drone {other_idx}: {xi_val}")
            
            print("  Omega (distance):")
            for other in known_others:
                other_idx = drones.index(other)
                omega_val = drone.omega.get(other, float('inf'))
                if omega_val != float('inf'):
                    print(f"    vers Drone {other_idx}: {omega_val}")
        else:
            print("  Aucun autre drone connu")


if __name__ == "__main__":
    test_simple_drone_network()
