#!/usr/bin/env python3
"""
Tests pour vérifier le bon fonctionnement des algorithmes distribués
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from drone import Drone
from network import DroneNetwork
from message import MessageType
import time


def test_basic_network():
    """Test basique avec 3 drones connectés en ligne"""
    print("=== Test Réseau Basique (3 drones en ligne) ===")
    
    network = DroneNetwork()
    
    # Créer 3 drones
    id1 = network.add_drone(100, 100)  # Drone 0
    id2 = network.add_drone(200, 100)  # Drone 1  
    id3 = network.add_drone(300, 100)  # Drone 2
    
    # Les connecter en ligne : 0-1-2
    network.connect_drones(id1, id2)
    network.connect_drones(id2, id3)
    
    print(f"Réseau créé: Drones {id1}, {id2}, {id3}")
    print(f"Connexions: {network.connections}")
    
    # Simuler plusieurs étapes
    for step in range(10):
        print(f"\n--- Étape {step + 1} ---")
        network.step()
        
        # Afficher l'état des drones
        for drone_id, drone in network.drones.items():
            print(f"Drone {drone_id}:")
            print(f"  Phase: {drone.algorithm_phase}")
            print(f"  Itération: {drone.iteration}")
            print(f"  n_estimate: {drone.n_estimate}")
            print(f"  xi: {drone.xi}")
            print(f"  omega: {drone.omega}")
            print(f"  Arêtes critiques: {drone.critical_edges}")
            
    # Vérifier les arêtes critiques détectées
    critical_edges = network.get_all_critical_edges()
    print(f"\n=== Résultats ===")
    print(f"Arêtes critiques détectées: {critical_edges}")
    
    # Dans une topologie en ligne, toutes les arêtes devraient être critiques
    expected_critical = {(0, 1), (1, 2)}
    if critical_edges == expected_critical:
        print("✓ Test réussi : toutes les arêtes sont bien détectées comme critiques")
    else:
        print("✗ Test échoué : arêtes critiques incorrectes")
        
    return critical_edges == expected_critical


def test_cycle_network():
    """Test avec un cycle de 4 drones"""
    print("\n=== Test Réseau Cyclique (4 drones en cycle) ===")
    
    network = DroneNetwork()
    
    # Créer 4 drones
    ids = []
    for i in range(4):
        drone_id = network.add_drone(100 + i * 100, 100)
        ids.append(drone_id)
    
    # Les connecter en cycle : 0-1-2-3-0
    for i in range(4):
        network.connect_drones(ids[i], ids[(i + 1) % 4])
    
    print(f"Réseau créé: Drones {ids}")
    print(f"Connexions: {network.connections}")
    
    # Simuler plusieurs étapes
    for step in range(15):
        network.step()
        
    # Vérifier les arêtes critiques détectées
    critical_edges = network.get_all_critical_edges()
    print(f"\n=== Résultats ===")
    print(f"Arêtes critiques détectées: {critical_edges}")
    
    # Dans un cycle simple, aucune arête ne devrait être critique
    if len(critical_edges) == 0:
        print("✓ Test réussi : aucune arête critique dans le cycle")
        return True
    else:
        print("✗ Test échoué : des arêtes sont incorrectement détectées comme critiques")
        return False


def test_bridge_network():
    """Test avec deux triangles connectés par un pont"""
    print("\n=== Test Réseau avec Pont (2 triangles + pont) ===")
    
    network = DroneNetwork()
    
    # Créer 6 drones
    ids = []
    for i in range(6):
        drone_id = network.add_drone(100 + (i % 3) * 100, 100 + (i // 3) * 150)
        ids.append(drone_id)
    
    # Premier triangle : 0-1-2-0
    network.connect_drones(ids[0], ids[1])
    network.connect_drones(ids[1], ids[2])
    network.connect_drones(ids[2], ids[0])
    
    # Deuxième triangle : 3-4-5-3
    network.connect_drones(ids[3], ids[4])
    network.connect_drones(ids[4], ids[5])
    network.connect_drones(ids[5], ids[3])
    
    # Pont entre les triangles : 1-4
    network.connect_drones(ids[1], ids[4])
    
    print(f"Réseau créé: Drones {ids}")
    print(f"Connexions: {network.connections}")
    
    # Simuler plusieurs étapes
    for step in range(20):
        network.step()
        
    # Vérifier les arêtes critiques détectées
    critical_edges = network.get_all_critical_edges()
    print(f"\n=== Résultats ===")
    print(f"Arêtes critiques détectées: {critical_edges}")
    
    # Le pont (1,4) devrait être l'unique arête critique
    expected_bridge = tuple(sorted([ids[1], ids[4]]))
    if len(critical_edges) == 1 and expected_bridge in critical_edges:
        print("✓ Test réussi : le pont est correctement détecté comme arête critique")
        return True
    else:
        print("✗ Test échoué : détection incorrecte des arêtes critiques")
        return False


def test_dynamic_network():
    """Test de l'adaptation à une topologie changeante"""
    print("\n=== Test Réseau Dynamique ===")
    
    network = DroneNetwork()
    
    # Créer 4 drones
    ids = []
    for i in range(4):
        drone_id = network.add_drone(100 + i * 100, 100)
        ids.append(drone_id)
    
    # Commencer avec une ligne : 0-1-2-3
    for i in range(3):
        network.connect_drones(ids[i], ids[i + 1])
    
    print(f"Réseau initial (ligne): {network.connections}")
    
    # Simuler quelques étapes
    for step in range(10):
        network.step()
    
    critical_before = network.get_all_critical_edges()
    print(f"Arêtes critiques avant modification: {critical_before}")
    
    # Ajouter une connexion pour former un cycle partiel : 0-3
    network.connect_drones(ids[0], ids[3])
    print(f"Connexion ajoutée: ({ids[0]}, {ids[3]})")
    print(f"Nouveau réseau: {network.connections}")
    
    # Simuler plus d'étapes après la modification
    for step in range(15):
        network.step()
    
    critical_after = network.get_all_critical_edges()
    print(f"Arêtes critiques après modification: {critical_after}")
    
    # Vérifier que le système s'adapte
    if len(critical_after) < len(critical_before):
        print("✓ Test réussi : le système s'adapte aux changements de topologie")
        return True
    else:
        print("✗ Test partiel : adaptation non optimale détectée")
        return False


def main():
    """Lancer tous les tests"""
    print("Lancement des tests des algorithmes distribués...\n")
    
    tests = [
        ("Réseau basique", test_basic_network),
        ("Réseau cyclique", test_cycle_network),
        ("Réseau avec pont", test_bridge_network),
        ("Réseau dynamique", test_dynamic_network)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Erreur dans {test_name}: {e}")
            results.append((test_name, False))
        
        print("\n" + "="*60 + "\n")
    
    # Résumé des résultats
    print("=== RÉSUMÉ DES TESTS ===")
    passed = 0
    for test_name, result in results:
        status = "RÉUSSI" if result else "ÉCHOUÉ"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests réussis: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 Tous les tests sont passés avec succès!")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez l'implémentation.")


if __name__ == "__main__":
    main()
