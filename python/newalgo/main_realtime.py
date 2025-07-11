#!/usr/bin/env python3
"""
Interface en temps réel pour l'algorithme de verrouillage décentralisé
Version console interactive avec mise à jour en temps réel
"""

import networkx as nx
import numpy as np
import time
import threading
import os
import sys
from enum import Enum
from typing import Dict, List, Tuple, Optional

# Configuration du terminal pour l'affichage dynamique
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

np.random.seed(21)

class MessageType(Enum):
    LOCK = "lock"
    UNLOCK = "unlock"

class Message:
    def __init__(self, sender, destination, message_type, edge=None):
        self.sender = sender
        self.destination = destination
        self.message_type = message_type
        self.edge = edge

    def __str__(self):
        return f"Msg({self.sender}→{self.destination}: {self.message_type.value}, {self.edge})"

class Node:
    pending_messages = []

    def __init__(self, id, neighbors):
        self.id = id
        self.neighbors = neighbors
        self.is_locked = False
        self.locked_edge = None
        self.unlock_status = {}
        self.outbox = []
        
        for neighbor in neighbors:
            self.unlock_status[neighbor] = False
    
    def add_to_outbox(self, message):
        self.outbox.append(message)
    
    def send_messages(self):
        for message in self.outbox:
            Node.pending_messages.append(message)
        self.outbox.clear()
    
    def process_message(self, message):
        if message.message_type == MessageType.LOCK:
            if not self.is_locked:
                self.is_locked = True
                self.locked_edge = message.edge
                for neighbor in self.unlock_status:
                    self.unlock_status[neighbor] = False
                
                for neighbor in self.neighbors:
                    lock_message = Message(self.id, neighbor, MessageType.LOCK, message.edge)
                    self.add_to_outbox(lock_message)
            else:
                unlock_message = Message(self.id, message.sender, MessageType.UNLOCK, self.locked_edge)
                self.add_to_outbox(unlock_message)
        
        elif message.message_type == MessageType.UNLOCK:
            if self.is_locked and message.edge == self.locked_edge:
                self.unlock_status[message.sender] = True
                
                if all(self.unlock_status.values()):
                    self.unlock()
    
    def unlock(self):
        self.is_locked = False
        self.locked_edge = None
        for neighbor in self.unlock_status:
            self.unlock_status[neighbor] = False
    
    def initiate_lock(self, edge):
        if not self.is_locked:
            self.is_locked = True
            self.locked_edge = edge
            for neighbor in self.unlock_status:
                self.unlock_status[neighbor] = False
            
            for neighbor in self.neighbors:
                lock_message = Message(self.id, neighbor, MessageType.LOCK, edge)
                self.add_to_outbox(lock_message)
    
    def get_status_symbol(self, target_edge=None):
        """Retourne un symbole pour représenter l'état du nœud"""
        if not self.is_locked:
            return "○"  # Libre
        elif target_edge and self.locked_edge == target_edge:
            if all(self.unlock_status.values()):
                return "✓"  # Peut supprimer l'arête en sécurité
            else:
                return "●"  # Verrouillé sur l'arête cible
        else:
            return "◐"  # Verrouillé sur une autre arête

class RealTimeSimulation:
    def __init__(self):
        self.G = None
        self.nodes = {}
        self.edges_list = []
        self.is_running = False
        self.timestep = 0
        self.target_edge = None
        self.simulation_speed = 1.0
        self.log_messages = []
        self.auto_mode = False
        
    def create_graph(self):
        """Crée un graphe aléatoire"""
        self.G = nx.Graph()
        
        # Ajouter des nœuds
        for i in range(7):
            self.G.add_node(i)
        
        # Ajouter des arêtes aléatoires
        edges = [
            tuple(np.random.choice(list(self.G.nodes()), size=2, replace=False))
            for _ in range(10)
        ]
        self.G.add_edges_from(edges)
        
        # Créer les objets Node
        self.nodes = {}
        for node_id in self.G.nodes():
            neighbors = list(self.G.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)
        
        self.edges_list = list(self.G.edges())
        self.log_message("Nouveau graphe créé")
        
    def simulate_timestep(self):
        """Simule un pas de temps"""
        if not self.is_running:
            return False
            
        self.timestep += 1
        
        # Traiter les messages
        messages_to_process = Node.pending_messages.copy()
        Node.pending_messages.clear()
        
        for message in messages_to_process:
            if message.destination in self.nodes:
                self.nodes[message.destination].process_message(message)
                self.log_message(f"T{self.timestep}: {message}")
        
        # Envoyer les nouveaux messages
        for node in self.nodes.values():
            node.send_messages()
        
        # Vérifier si la simulation doit s'arrêter
        if not Node.pending_messages and all(not node.outbox for node in self.nodes.values()):
            self.is_running = False
            self.log_message("Simulation terminée")
            return False
        
        return True
    
    def start_lock_simulation(self, edge_index):
        """Démarre une simulation de verrouillage"""
        if not self.edges_list or edge_index >= len(self.edges_list):
            return False
            
        self.target_edge = self.edges_list[edge_index]
        initiator_node = self.target_edge[0]
        
        # Réinitialiser l'état
        Node.pending_messages.clear()
        for node in self.nodes.values():
            node.is_locked = False
            node.locked_edge = None
            node.outbox.clear()
            for neighbor in node.unlock_status:
                node.unlock_status[neighbor] = False
        
        self.timestep = 0
        self.is_running = True
        
        # Initier le verrouillage
        self.nodes[initiator_node].initiate_lock(self.target_edge)
        self.log_message(f"Verrouillage initié sur {self.target_edge} par nœud {initiator_node}")
        
        return True
    
    def stop_simulation(self):
        self.is_running = False
        self.auto_mode = False
    
    def log_message(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_messages.append(f"[{timestamp}] {message}")
        if len(self.log_messages) > 20:
            self.log_messages.pop(0)
    
    def get_status_display(self):
        """Retourne l'affichage de l'état de la simulation"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"ALGORITHME DE VERROUILLAGE DÉCENTRALISÉ - Pas de temps: {self.timestep}")
        lines.append("=" * 80)
        
        if not self.G:
            lines.append("Aucun graphe chargé. Utilisez 'c' pour créer un graphe.")
            return "\n".join(lines)
        
        # Affichage du graphe
        lines.append("\nGRAPHE:")
        lines.append("Arêtes: " + ", ".join([f"{e[0]}-{e[1]}" for e in self.edges_list]))
        
        if self.target_edge:
            lines.append(f"Arête cible: {self.target_edge[0]}-{self.target_edge[1]}")
        
        # État des nœuds
        lines.append("\nÉTAT DES NŒUDS:")
        node_display = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            symbol = node.get_status_symbol(self.target_edge)
            status = f"Nœud {node_id}: {symbol}"
            
            if node.is_locked:
                status += f" [Verrouillé sur {node.locked_edge}]"
                if node.unlock_status:
                    unlocked = [str(k) for k, v in node.unlock_status.items() if v]
                    pending = [str(k) for k, v in node.unlock_status.items() if not v]
                    if unlocked:
                        status += f" ✓:{','.join(unlocked)}"
                    if pending:
                        status += f" ⏳:{','.join(pending)}"
            
            node_display.append(status)
        
        lines.extend(node_display)
        
        # Statistiques
        lines.append(f"\nSTATISTIQUES:")
        lines.append(f"- Simulation: {'En cours' if self.is_running else 'Arrêtée'}")
        lines.append(f"- Messages en attente: {len(Node.pending_messages)}")
        lines.append(f"- Mode: {'Automatique' if self.auto_mode else 'Manuel'}")
        
        # Nœuds pouvant supprimer l'arête
        safe_nodes = [str(n.id) for n in self.nodes.values() 
                     if n.is_locked and all(n.unlock_status.values())]
        if safe_nodes:
            lines.append(f"- Nœuds sûrs pour suppression: {', '.join(safe_nodes)}")
        
        # Log récent
        lines.append("\nJOURNAL (10 derniers messages):")
        lines.extend(self.log_messages[-10:])
        
        # Aide
        lines.append("\nCOMMANDES:")
        lines.append("c: Créer nouveau graphe | s: Démarrer simulation | x: Arrêter")
        lines.append("n: Pas manuel | a: Mode auto on/off | +/-: Vitesse | q: Quitter")
        
        return "\n".join(lines)

def auto_simulation_worker(sim):
    """Worker thread pour la simulation automatique"""
    while True:
        if sim.auto_mode and sim.is_running:
            sim.simulate_timestep()
            time.sleep(sim.simulation_speed)
        else:
            time.sleep(0.1)

def console_interface():
    """Interface console interactive"""
    sim = RealTimeSimulation()
    
    # Démarrer le thread de simulation automatique
    auto_thread = threading.Thread(target=auto_simulation_worker, args=(sim,), daemon=True)
    auto_thread.start()
    
    # Interface principal
    if CURSES_AVAILABLE:
        # Version avec curses (affichage dynamique)
        def curses_main(stdscr):
            curses.curs_set(0)  # Cacher le curseur
            stdscr.nodelay(1)   # Non-bloquant
            stdscr.timeout(100) # Timeout de 100ms
            
            while True:
                stdscr.clear()
                
                # Afficher l'état
                status = sim.get_status_display()
                try:
                    stdscr.addstr(0, 0, status)
                except curses.error:
                    pass  # Ignorer les erreurs d'affichage
                
                stdscr.refresh()
                
                # Gérer les entrées clavier
                key = stdscr.getch()
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    sim.create_graph()
                elif key == ord('s'):
                    if sim.edges_list:
                        edge_index = 0  # Première arête par défaut
                        sim.start_lock_simulation(edge_index)
                elif key == ord('x'):
                    sim.stop_simulation()
                elif key == ord('n'):
                    sim.simulate_timestep()
                elif key == ord('a'):
                    sim.auto_mode = not sim.auto_mode
                    sim.log_message(f"Mode automatique: {'ON' if sim.auto_mode else 'OFF'}")
                elif key == ord('+') or key == ord('='):
                    sim.simulation_speed = max(0.1, sim.simulation_speed - 0.2)
                    sim.log_message(f"Vitesse: {sim.simulation_speed:.1f}s")
                elif key == ord('-'):
                    sim.simulation_speed = min(3.0, sim.simulation_speed + 0.2)
                    sim.log_message(f"Vitesse: {sim.simulation_speed:.1f}s")
                
                time.sleep(0.05)
        
        curses.wrapper(curses_main)
    
    else:
        # Version basique sans curses
        sim.create_graph()
        
        while True:
            # Effacer l'écran
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Afficher l'état
            print(sim.get_status_display())
            
            # Attendre une commande
            try:
                cmd = input("\nCommande (h pour aide): ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'h':
                    print("\nCommandes disponibles:")
                    print("c: Créer nouveau graphe")
                    print("s: Démarrer simulation sur première arête")
                    print("x: Arrêter simulation")
                    print("n: Exécuter un pas manuel")
                    print("a: Basculer mode automatique")
                    print("q: Quitter")
                    input("\nAppuyez sur Entrée pour continuer...")
                elif cmd == 'c':
                    sim.create_graph()
                elif cmd == 's':
                    if sim.edges_list:
                        sim.start_lock_simulation(0)
                        sim.auto_mode = True
                elif cmd == 'x':
                    sim.stop_simulation()
                elif cmd == 'n':
                    sim.simulate_timestep()
                elif cmd == 'a':
                    sim.auto_mode = not sim.auto_mode
                    sim.log_message(f"Mode automatique: {'ON' if sim.auto_mode else 'OFF'}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break

def main():
    """Point d'entrée principal"""
    print("Démarrage de la simulation d'algorithme de verrouillage décentralisé...")
    print("Interface en temps réel")
    
    if CURSES_AVAILABLE:
        print("Interface avec curses détectée - affichage temps réel activé")
    else:
        print("Interface basique - mode commande")
    
    time.sleep(1)
    
    try:
        console_interface()
    except KeyboardInterrupt:
        print("\nSimulation interrompue par l'utilisateur")
    except Exception as e:
        print(f"\nErreur: {e}")
    
    print("Fin de la simulation")

if __name__ == "__main__":
    main()
