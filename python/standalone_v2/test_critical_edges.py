#!/usr/bin/env python3
"""
Tests pour valider l'algorithme de détection des arêtes critiques
"""

from network import DroneNetwork
from simulation_core import SimulationCore


def test_simple_bridge_network():
    """Tester avec un réseau simple ayant des arêtes critiques évidentes"""
    print("Test: Réseau avec pont évident")
    print("="*40)
    
    network = DroneNetwork()
    
    # Créer un réseau en forme de pont: 0-1-2 avec 1 comme pont
    network.add_drone(x=100, y=300)  # Drone 0
    network.add_drone(x=300, y=300)  # Drone 1 (pont)
    network.add_drone(x=500, y=300)  # Drone 2
    
    # Connecter pour former un pont
    network.connect_drones(0, 1)
    network.connect_drones(1, 2)
    
    print(f"Réseau créé: {len(network.drones)} drones")
    print(f"Connexions: {list(network.connections)}")
    
    # Calculer le nombre d'étapes nécessaires
    n = len(network.drones)
    total_steps = 2 * n + 2
    print(f"Exécution sur {total_steps} étapes...")
    
    # Simuler
    for step in range(total_steps):
        network.step()
        if step % 3 == 0:
            critical_edges = network.get_all_critical_edges()
            print(f"Étape {step}: {len(critical_edges)} arêtes critiques")
            
    # Résultats finaux
    critical_edges = network.get_all_critical_edges()
    print(f"\nRésultats finaux:")
    print(f"Arêtes critiques détectées: {critical_edges}")
    print(f"Arêtes attendues comme critiques: {[(0, 1), (1, 2)]}")
    
    return critical_edges


def test_cycle_network():
    """Tester avec un réseau en cycle (aucune arête critique)"""
    print("\n\nTest: Réseau en cycle (pas d'arêtes critiques)")
    print("="*50)
    
    network = DroneNetwork()
    
    # Créer un cycle: 0-1-2-0
    network.add_drone(x=200, y=200)  # Drone 0
    network.add_drone(x=400, y=200)  # Drone 1
    network.add_drone(x=300, y=400)  # Drone 2
    
    # Connecter en cycle
    network.connect_drones(0, 1)
    network.connect_drones(1, 2)
    network.connect_drones(2, 0)
    
    print(f"Réseau créé: {len(network.drones)} drones")
    print(f"Connexions: {list(network.connections)}")
    
    # Simuler
    n = len(network.drones)
    total_steps = 2 * n + 2
    print(f"Exécution sur {total_steps} étapes...")
    
    for step in range(total_steps):
        network.step()
        
    # Résultats finaux
    critical_edges = network.get_all_critical_edges()
    print(f"\nRésultats finaux:")
    print(f"Arêtes critiques détectées: {critical_edges}")
    print(f"Attendu: aucune arête critique (cycle)")
    
    return critical_edges


def test_complex_network():
    """Tester avec un réseau plus complexe"""
    print("\n\nTest: Réseau complexe")
    print("="*30)
    
    sim = SimulationCore()
    result = sim.run_complete_algorithm_simulation(
        num_drones=6, 
        connection_prob=0.3, 
        verbose=False
    )
    
    critical_edges = result['final_results']['critical_edges']
    print(f"Réseau de 6 drones avec probabilité 0.3")
    print(f"Connexions initiales: {result['initial_network']['connections']}")
    print(f"Arêtes critiques détectées: {critical_edges}")
    
    return critical_edges


def run_validation_tests():
    """Lancer tous les tests de validation"""
    print("TESTS DE VALIDATION DE L'ALGORITHME")
    print("="*60)
    
    # Test 1: Pont simple
    bridge_result = test_simple_bridge_network()
    
    # Test 2: Cycle
    cycle_result = test_cycle_network()
    
    # Test 3: Réseau complexe
    complex_result = test_complex_network()
    
    print("\n\nRÉSUMÉ DES TESTS")
    print("="*30)
    print(f"Test pont: {len(bridge_result)} arêtes critiques")
    print(f"Test cycle: {len(cycle_result)} arêtes critiques")
    print(f"Test complexe: {len(complex_result)} arêtes critiques")
    
    # Validation des résultats attendus
    print("\nValidation:")
    if len(bridge_result) > 0:
        print("✓ Réseau pont: arêtes critiques détectées (attendu)")
    else:
        print("✗ Réseau pont: aucune arête critique détectée (inattendu)")
        
    if len(cycle_result) == 0:
        print("✓ Réseau cycle: aucune arête critique (attendu)")
    else:
        print("✗ Réseau cycle: arêtes critiques détectées (inattendu)")


if __name__ == "__main__":
    run_validation_tests()
