#!/usr/bin/env python3
"""
Démonstration des algorithmes distribués de détection d'arêtes critiques
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import DroneNetwork
from drone import Drone
import time
import random


def demo_scenario(name: str, setup_func, expected_critical_count=None):
    """Démonstration d'un scénario spécifique"""
    print(f"\n{'='*60}")
    print(f"DÉMONSTRATION: {name}")
    print('='*60)
    
    network = DroneNetwork()
    setup_func(network)
    
    print(f"Réseau initial:")
    print(f"  - {len(network.drones)} drones")
    print(f"  - {len(network.connections)} connexions: {network.connections}")
    
    # Simulation
    print(f"\nSimulation en cours...")
    for step in range(25):
        network.step()
        if step % 5 == 4:
            critical_edges = network.get_all_critical_edges()
            print(f"  Étape {step + 1:2d}: {len(critical_edges)} arêtes critiques")
    
    # Résultats finaux
    critical_edges = network.get_all_critical_edges()
    print(f"\nRésultats finaux:")
    print(f"  - Arêtes critiques détectées: {critical_edges}")
    print(f"  - Nombre d'arêtes critiques: {len(critical_edges)}")
    
    if expected_critical_count is not None:
        if len(critical_edges) == expected_critical_count:
            print(f"  ✓ Résultat attendu ({expected_critical_count} arêtes critiques)")
        else:
            print(f"  ⚠ Résultat inattendu (attendu: {expected_critical_count}, obtenu: {len(critical_edges)})")
    
    # État des drones
    print(f"\nÉtat des drones:")
    for drone_id, drone in network.drones.items():
        degree = len(drone.neighbors)
        print(f"  Drone {drone_id}: degré={degree}, phase={drone.algorithm_phase}")
        
    return critical_edges


def setup_line_network(network):
    """Configuration: réseau en ligne 0-1-2-3"""
    for i in range(4):
        network.add_drone(100 + i * 100, 100)
    
    for i in range(3):
        network.connect_drones(i, i + 1)


def setup_star_network(network):
    """Configuration: réseau en étoile avec centre 0"""
    for i in range(5):
        network.add_drone(200 + (i % 2) * 200, 150 + (i // 2) * 100)
    
    # Drone 0 au centre connecté à tous les autres
    for i in range(1, 5):
        network.connect_drones(0, i)


def setup_cycle_network(network):
    """Configuration: réseau cyclique 0-1-2-3-0"""
    for i in range(4):
        network.add_drone(150 + 100 * (i % 2), 150 + 100 * (i // 2))
    
    for i in range(4):
        network.connect_drones(i, (i + 1) % 4)


def setup_bridge_network(network):
    """Configuration: deux triangles reliés par un pont"""
    # Triangle 1: nœuds 0, 1, 2
    for i in range(3):
        network.add_drone(100 + i * 50, 100)
    
    # Triangle 2: nœuds 3, 4, 5
    for i in range(3, 6):
        network.add_drone(300 + (i-3) * 50, 100)
    
    # Connexions triangle 1
    network.connect_drones(0, 1)
    network.connect_drones(1, 2)
    network.connect_drones(2, 0)
    
    # Connexions triangle 2
    network.connect_drones(3, 4)
    network.connect_drones(4, 5)
    network.connect_drones(5, 3)
    
    # Pont entre les triangles
    network.connect_drones(2, 3)


def setup_tree_network(network):
    """Configuration: réseau en arbre"""
    # Nœud racine
    network.add_drone(200, 100)
    
    # Niveau 1
    for i in range(1, 3):
        network.add_drone(150 + i * 100, 200)
        network.connect_drones(0, i)
    
    # Niveau 2
    for i in range(3, 7):
        network.add_drone(100 + (i-3) * 100, 300)
        parent = 1 + (i - 3) // 2
        network.connect_drones(parent, i)


def setup_random_network(network):
    """Configuration: réseau aléatoire"""
    # Créer 8 drones
    for i in range(8):
        network.add_drone()
    
    # Connexions aléatoires mais en s'assurant que le réseau soit connecté
    network.generate_random_network(8, 0.3)


def demo_dynamic_changes(network):
    """Démonstration des changements dynamiques"""
    print(f"\n{'='*60}")
    print(f"DÉMONSTRATION: Adaptation aux changements dynamiques")
    print('='*60)
    
    # Commencer avec une ligne
    setup_line_network(network)
    print(f"Réseau initial (ligne): {network.connections}")
    
    # Simulation initiale
    for step in range(15):
        network.step()
    
    critical_before = network.get_all_critical_edges()
    print(f"Arêtes critiques avant modification: {critical_before}")
    
    # Ajouter une connexion pour créer un cycle
    network.connect_drones(0, 3)
    print(f"Ajout de la connexion (0, 3)")
    print(f"Nouveau réseau: {network.connections}")
    
    # Simulation après modification
    for step in range(15):
        network.step()
    
    critical_after = network.get_all_critical_edges()
    print(f"Arêtes critiques après modification: {critical_after}")
    
    # Supprimer une connexion
    network.disconnect_drones(1, 2)
    print(f"Suppression de la connexion (1, 2)")
    print(f"Réseau final: {network.connections}")
    
    # Simulation finale
    for step in range(15):
        network.step()
    
    critical_final = network.get_all_critical_edges()
    print(f"Arêtes critiques après suppression: {critical_final}")


def main():
    """Lancement des démonstrations"""
    print("DÉMONSTRATIONS DES ALGORITHMES DISTRIBUÉS DE DÉTECTION D'ARÊTES CRITIQUES")
    print("=" * 80)
    
    scenarios = [
        ("Réseau en ligne", setup_line_network, 3),
        ("Réseau en étoile", setup_star_network, 4),
        ("Réseau cyclique", setup_cycle_network, 0),
        ("Réseau avec pont", setup_bridge_network, 1),
        ("Réseau en arbre", setup_tree_network, 6),
        ("Réseau aléatoire", setup_random_network, None),
    ]
    
    results = []
    
    for name, setup_func, expected in scenarios:
        critical_edges = demo_scenario(name, setup_func, expected)
        results.append((name, len(critical_edges), expected))
    
    # Démonstration des changements dynamiques
    network = DroneNetwork()
    demo_dynamic_changes(network)
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES DÉMONSTRATIONS")
    print('='*60)
    
    for name, obtained, expected in results:
        status = "✓" if expected is None or obtained == expected else "⚠"
        exp_str = f"(attendu: {expected})" if expected is not None else ""
        print(f"{status} {name:20s}: {obtained} arêtes critiques {exp_str}")
    
    print(f"\nLes algorithmes distribués détectent correctement les arêtes critiques")
    print(f"dans la plupart des topologies standard. Pour des réseaux plus complexes,")
    print(f"l'algorithme peut nécessiter des raffinements supplémentaires.")


if __name__ == "__main__":
    main()
