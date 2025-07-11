#!/usr/bin/env python3
"""
Simulateur de Drones Distribués - Point d'entrée principal

Ce simulateur implémente les algorithmes distribués pour la détection d'arêtes critiques
et l'augmentation de connectivité dans un réseau de drones, basé sur les algorithmes
décrits dans le document de recherche.

Modes d'utilisation:
- python main.py : Interface graphique (par défaut)
- python main.py --text : Mode texte simple
- python main.py --interactive : Mode interactif
- python main.py --test : Tests de performance
"""

import sys
from gui import DroneSimulationGUI
from simulation_core import SimulationCore, run_interactive_simulation


def main():
    """Fonction principale"""
    
    # Analyser les arguments de ligne de commande
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "--text":
            # Mode texte pour une simulation simple
            print("Mode texte activé - Simulation selon l'algorithme du papier")
            
            sim = SimulationCore()
            sim.run_complete_algorithm_simulation(
                num_drones=8, 
                connection_prob=0.4, 
                verbose=True
            )
            
        elif mode == "--interactive":
            # Mode interactif
            run_interactive_simulation()
            
        elif mode == "--test":
            # Tests de performance
            print("Mode test de performance")
            sim = SimulationCore()
            results = sim.run_performance_test(5)
            
            summary = results['summary']
            print(f"\nRésultats de {results['num_tests']} tests:")
            print(f"Temps d'exécution moyen: {summary['avg_execution_time']:.3f}s")
            print(f"Arêtes critiques moyennes: {summary['avg_critical_edges']:.1f}")
            
        elif mode == "--help":
            print(__doc__)
            return
            
        else:
            print(f"Mode inconnu: {mode}")
            print("Utilisez --help pour voir les options disponibles")
            return
            
    else:
        # Mode interface graphique par défaut
        try:
            print("Lancement de l'interface graphique...")
            app = DroneSimulationGUI()
            app.run()
        except Exception as e:
            print(f"Erreur lors du lancement de l'interface graphique: {e}")
            print("Essayez avec --text pour le mode texte ou --help pour l'aide")


if __name__ == "__main__":
    main()
