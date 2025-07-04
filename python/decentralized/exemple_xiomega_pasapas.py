import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from drone import Drone


def print_step(step, drones, all_messages):
    print(f"\n=== ÉTAPE {step} ===")
    print("Messages envoyés :")
    id_to_idx = {id(d): i for i, d in enumerate(drones)}
    for i, drone in enumerate(drones):
        msg = all_messages[drone]
        if msg:
            for neighbor, content in msg.items():
                j = drones.index(neighbor)
                # Correction ici : on affiche les indices des drones connus à partir des IDs
                known = sorted([id_to_idx[d_id] for d_id in content['known_drones_ids'] if d_id in id_to_idx and d_id != id(drone)])
                # Pour xi et omega, on affiche aussi les indices
                xi_idx = {id_to_idx.get(d_id, d_id): v for d_id, v in content['xi'].items() if d_id in id_to_idx and d_id != id(drone)}
                omega_idx = {id_to_idx.get(d_id, d_id): v for d_id, v in content['omega'].items() if d_id in id_to_idx and d_id != id(drone) and v != float('inf')}
                print(f"  Drone {i} -> Drone {j} | known: {known} | xi: {xi_idx} | omega: {omega_idx}")
    print("\nÉtat des drones :")
    for i, drone in enumerate(drones):
        known = [drones.index(d) for d in drone.known_drones if d != drone]
        xi = {drones.index(d): v for d, v in drone.xi.items() if d != drone}
        omega = {drones.index(d): v for d, v in drone.omega.items() if d != drone and v != float('inf')}
        print(f"  Drone {i}: connaît {known}, xi={xi}, omega={omega}")


def main():
    # 3 drones en ligne : 0 <-> 1 <-> 2
    positions = [np.array([0, 0]), np.array([1, 0]), np.array([2, 0])]
    drones = [Drone(position=pos, target=None) for pos in positions]
    # Connexions manuelles
    drones[0].add_connection(drones[1])
    drones[1].add_connection(drones[2])

    # Initialisation
    print("État initial :")
    for i, drone in enumerate(drones):
        print(f"  Drone {i}: position {drone.position}, connexions {[drones.index(d) for d in drone.connections]}")

    # Simulation pas à pas
    for step in range(6):
        # Phase 1 : chaque drone prépare ses messages
        all_messages = {}
        for drone in drones:
            all_messages[drone] = drone.prepare_messages()
        # Phase 2 : distribution des messages
        for sender, messages in all_messages.items():
            for receiver, message in messages.items():
                receiver.receive_message(sender, message)
        # Phase 3 : mise à jour
        for drone in drones:
            drone.update_xi_omega()
        # Affichage
        print_step(step+1, drones, all_messages)
        # Arrêt si tout le monde a convergé
        if all(drone.has_converged() for drone in drones):
            print("\n*** CONVERGENCE ATTEINTE ***")
            break

if __name__ == "__main__":
    main()
