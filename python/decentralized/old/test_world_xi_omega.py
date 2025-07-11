#!/usr/bin/env python3
"""
Test simple pour vérifier le fonctionnement de l'algorithme xi-omega dans world.py
sans interface graphique complexe
"""

import numpy as np
from drone import Drone
from target import Target
from world import World, WorldConfiguration

def test_xi_omega_simple():
    """Test simple de l'algorithme xi-omega"""
    
    print("=== Test de l'algorithme Xi-Omega ===\n")
    
    # Configuration simple
    config = WorldConfiguration(
        map_size=np.array([800, 800]),
        bounds=np.array([[0, 0], [800, 800]]),
        drones=[(100, 100),   # Drone 0
                (200, 100),   # Drone 1 
                (300, 100),   # Drone 2
                (400, 100),   # Drone 3
                (200, 200)],  # Drone 4 (branch)
        targets=[None, None, None, None, (500, 500)]
    )
    
    # Créer le monde
    world = World(config=config, bounds=np.array([[0, 0], [800, 800]]))
    
    print(f"Monde créé avec {len(world.drones)} drones")
    
    # Établir les connexions manuellement (topologie en ligne avec branche)
    if len(world.drones) >= 5:
        world.drones[0].add_connection(world.drones[1])  # 0-1
        world.drones[1].add_connection(world.drones[2])  # 1-2
        world.drones[2].add_connection(world.drones[3])  # 2-3
        world.drones[1].add_connection(world.drones[4])  # 1-4 (branche)
    
    print("Connexions établies:")
    for i, drone in enumerate(world.drones):
        connections = [world.drones.index(c) for c in drone.connections]
        print(f"  Drone {i}: connecté à {connections}")
    
    # État initial
    print("\n--- État initial ---")
    for i, drone in enumerate(world.drones):
        print(f"Drone {i}: connaît {len(drone.known_drones)} drones (lui-même)")
    
    # Exécuter l'algorithme step par step
    print("\n--- Exécution de l'algorithme Xi-Omega ---")
    for step in range(10):
        print(f"\nÉtape {step + 1}:")
        
        # Exécuter une étape pour tous les drones
        for drone in world.drones:
            drone.xi_omega_step()
        
        # Vérifier la convergence
        converged_count = sum(1 for drone in world.drones if drone.has_converged())
        print(f"  Convergés: {converged_count}/{len(world.drones)}")
        
        # Afficher l'état du premier drone comme exemple
        drone0 = world.drones[0]
        known_count = len(drone0.known_drones) - 1
        print(f"  Drone 0: itération {drone0.iteration}, connaît {known_count} autres drones")
        
        # Arrêter si convergence globale
        if converged_count == len(world.drones):
            print(f"\n🎉 Convergence globale atteinte à l'étape {step + 1}!")
            break
    
    # État final détaillé
    print("\n" + "="*60)
    print("ÉTAT FINAL")
    print("="*60)
    
    for i, drone in enumerate(world.drones):
        print(f"\nDrone {i} (position {drone.position}):")
        print(f"  - Itération: {drone.iteration}")
        print(f"  - Convergé: {drone.has_converged()}")
        print(f"  - Connexions: {[world.drones.index(c) for c in drone.connections]}")
        print(f"  - Drones connus: {len(drone.known_drones) - 1} (+ lui-même)")
        
        # Afficher les distances omega vers les autres drones
        distances = {}
        for other_drone in drone.known_drones:
            if other_drone != drone:
                idx = world.drones.index(other_drone)
                dist = drone.omega.get(other_drone, float('inf'))
                distances[f"Drone{idx}"] = dist if dist != float('inf') else '∞'
        
        if distances:
            print(f"  - Distances (ω): {distances}")
    
    # Vérification de la cohérence
    print("\n--- Vérification de la cohérence ---")
    print("Les distances doivent être cohérentes entre tous les drones...")
    
    # Vérifier que tous les drones connaissent le même nombre de drones
    known_counts = [len(d.known_drones) for d in world.drones]
    if len(set(known_counts)) == 1:
        print(f"✅ Tous les drones connaissent {known_counts[0]} drones (incluant eux-mêmes)")
    else:
        print(f"❌ Incohérence: les drones connaissent {known_counts} drones respectivement")
    
    print(f"\nTest terminé. Algorithme Xi-Omega {'✅ RÉUSSI' if converged_count == len(world.drones) else '❌ INCOMPLET'}")


if __name__ == "__main__":
    test_xi_omega_simple()
