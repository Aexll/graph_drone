#!/usr/bin/env python3
"""
Cœur de simulation pour les drones distribués

Ce module contient les fonctions principales de simulation et d'analyse
pour les algorithmes distribués de détection d'arêtes critiques.
"""

import time
from typing import Dict, List, Tuple, Set
from network import DroneNetwork
from drone import Drone


class SimulationCore:
    """Classe principale pour exécuter les simulations de drones distribués"""
    
    def __init__(self):
        self.network = DroneNetwork()
        self.simulation_history = []
        
    def run_complete_algorithm_simulation(self, num_drones: int = 8, 
                                         connection_prob: float = 0.4,
                                         verbose: bool = True) -> Dict:
        """
        Exécuter une simulation complète des algorithmes 1, 2 et 3
        selon les spécifications du papier
        """
        if verbose:
            print(f"Démarrage simulation avec {num_drones} drones, prob_connexion={connection_prob}")
            
        # Générer le réseau initial
        self.network.generate_random_network(num_drones, connection_prob)
        
        if verbose:
            print(f"Réseau généré: {len(self.network.drones)} drones, {len(self.network.connections)} connexions")
            print(f"Connexions: {sorted(self.network.connections)}")
            
        # Calculer le nombre total d'étapes attendues: (2n + 2)
        n = len(self.network.drones)
        total_steps = 2 * n + 2
        
        if verbose:
            print(f"Exécution des algorithmes sur {total_steps} étapes...")
            
        results = {
            'initial_network': {
                'drones': list(self.network.drones.keys()),
                'connections': list(self.network.connections)
            },
            'algorithm_phases': [],
            'critical_edges_evolution': [],
            'final_results': {}
        }
        
        # Exécuter la simulation étape par étape
        for step in range(total_steps):
            step_info = self._execute_simulation_step(step, verbose)
            results['algorithm_phases'].append(step_info)
            
            # Capturer l'évolution des arêtes critiques
            critical_edges = self.network.get_all_critical_edges()
            results['critical_edges_evolution'].append({
                'step': step,
                'critical_edges': list(critical_edges)
            })
            
        # Analyser les résultats finaux
        results['final_results'] = self._analyze_final_results()
        
        if verbose:
            self._print_final_summary(results)
            
        return results
        
    def _execute_simulation_step(self, step: int, verbose: bool = False) -> Dict:
        """Exécuter une étape de simulation et collecter les informations"""
        
        # État avant l'étape
        phase_distribution = self._get_phase_distribution()
        
        # Exécuter l'étape
        self.network.step()
        
        # État après l'étape
        new_phase_distribution = self._get_phase_distribution()
        critical_edges = self.network.get_all_critical_edges()
        
        step_info = {
            'step': step,
            'phase_before': phase_distribution,
            'phase_after': new_phase_distribution,
            'critical_edges_count': len(critical_edges),
            'drone_states': {}
        }
        
        # Collecter l'état détaillé de chaque drone
        for drone_id, drone in self.network.drones.items():
            step_info['drone_states'][drone_id] = {
                'phase': drone.algorithm_phase,
                'iteration': drone.iteration,
                'n_estimate': drone.n_estimate,
                'known_nodes_count': len(drone.get_known_nodes()),
                'critical_edges': list(drone.critical_edges)
            }
            
        if verbose and step % 5 == 0:
            print(f"Étape {step}: phases={new_phase_distribution}, "
                  f"arêtes_critiques={len(critical_edges)}")
            
        return step_info
        
    def _get_phase_distribution(self) -> Dict[str, int]:
        """Obtenir la distribution des phases des drones"""
        phases = {}
        for drone in self.network.drones.values():
            phase = drone.algorithm_phase
            phases[phase] = phases.get(phase, 0) + 1
        return phases
        
    def _analyze_final_results(self) -> Dict:
        """Analyser les résultats finaux de la simulation"""
        critical_edges = self.network.get_all_critical_edges()
        
        # Vérifier la cohérence des détections entre drones
        edge_detections = {}
        for drone in self.network.drones.values():
            for edge in drone.critical_edges:
                if edge not in edge_detections:
                    edge_detections[edge] = []
                edge_detections[edge].append(drone.id)
                
        # Analyser les valeurs omega finales
        omega_analysis = {}
        for drone_id, drone in self.network.drones.items():
            omega_analysis[drone_id] = {
                'reachable_nodes': len([d for d in drone.omega.values() if d != float('inf')]),
                'max_distance': max([d for d in drone.omega.values() if d != float('inf')] + [0])
            }
            
        return {
            'critical_edges': list(critical_edges),
            'edge_detections': edge_detections,
            'omega_analysis': omega_analysis,
            'final_phases': self._get_phase_distribution(),
            'network_connectivity': self._analyze_network_connectivity()
        }
        
    def _analyze_network_connectivity(self) -> Dict:
        """Analyser la connectivité du réseau"""
        # Utiliser l'état des drones pour estimer la connectivité
        connectivity_info = {}
        
        for drone_id, drone in self.network.drones.items():
            reachable_nodes = [node for node, dist in drone.omega.items() 
                             if dist != float('inf')]
            connectivity_info[drone_id] = {
                'reachable_count': len(reachable_nodes),
                'total_known': len(drone.get_known_nodes()),
                'is_fully_connected': len(reachable_nodes) == len(self.network.drones)
            }
            
        return connectivity_info
        
    def _print_final_summary(self, results: Dict):
        """Afficher un résumé final de la simulation"""
        print("\n" + "="*60)
        print("RÉSUMÉ FINAL DE LA SIMULATION")
        print("="*60)
        
        final_results = results['final_results']
        critical_edges = final_results['critical_edges']
        
        print(f"Arêtes critiques détectées: {len(critical_edges)}")
        for edge in critical_edges:
            print(f"  {edge[0]} ↔ {edge[1]}")
            
        print(f"\nPhases finales des drones:")
        for phase, count in final_results['final_phases'].items():
            print(f"  {phase}: {count} drones")
            
        print(f"\nAnalyse de connectivité:")
        for drone_id, info in final_results['network_connectivity'].items():
            print(f"  Drone {drone_id}: {info['reachable_count']}/{len(self.network.drones)} nœuds atteignables")
            
    def run_performance_test(self, num_tests: int = 10) -> Dict:
        """Exécuter plusieurs tests pour analyser les performances"""
        print(f"Exécution de {num_tests} tests de performance...")
        
        results = []
        
        for test_num in range(num_tests):
            start_time = time.time()
            
            # Varier les paramètres
            num_drones = 6 + (test_num % 5)  # 6 à 10 drones
            conn_prob = 0.2 + (test_num % 3) * 0.1  # 0.2 à 0.4
            
            test_result = self.run_complete_algorithm_simulation(
                num_drones, conn_prob, verbose=False
            )
            
            end_time = time.time()
            
            test_result['test_params'] = {
                'num_drones': num_drones,
                'connection_prob': conn_prob,
                'execution_time': end_time - start_time
            }
            
            results.append(test_result)
            print(f"Test {test_num + 1}/{num_tests} terminé en {end_time - start_time:.3f}s")
            
        return {
            'num_tests': num_tests,
            'individual_results': results,
            'summary': self._analyze_performance_results(results)
        }
        
    def _analyze_performance_results(self, results: List[Dict]) -> Dict:
        """Analyser les résultats de performance"""
        execution_times = [r['test_params']['execution_time'] for r in results]
        critical_edge_counts = [len(r['final_results']['critical_edges']) for r in results]
        
        return {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_critical_edges': sum(critical_edge_counts) / len(critical_edge_counts),
            'total_tests': len(results)
        }


def run_interactive_simulation():
    """Lancer une simulation interactive en mode texte"""
    sim = SimulationCore()
    
    print("Simulateur de Drones Distribués - Mode Interactif")
    print("="*50)
    
    while True:
        print("\nOptions disponibles:")
        print("1. Simulation simple")
        print("2. Simulation avec paramètres personnalisés")
        print("3. Tests de performance")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == '1':
            sim.run_complete_algorithm_simulation()
            
        elif choice == '2':
            try:
                num_drones = int(input("Nombre de drones (3-15): "))
                conn_prob = float(input("Probabilité de connexion (0.1-0.8): "))
                sim.run_complete_algorithm_simulation(num_drones, conn_prob)
            except ValueError:
                print("Paramètres invalides!")
                
        elif choice == '3':
            try:
                num_tests = int(input("Nombre de tests (1-20): "))
                results = sim.run_performance_test(num_tests)
                summary = results['summary']
                print(f"\nRésultats de performance:")
                print(f"Temps d'exécution moyen: {summary['avg_execution_time']:.3f}s")
                print(f"Arêtes critiques moyennes: {summary['avg_critical_edges']:.1f}")
            except ValueError:
                print("Nombre de tests invalide!")
                
        elif choice == '4':
            break
            
        else:
            print("Choix invalide!")


if __name__ == "__main__":
    run_interactive_simulation()
