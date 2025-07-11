import numpy as np

class Node:
    def __init__(self, node_id, num_nodes, neighbors):
        self.node_id = node_id  # Index du nœud (de 0 à n-1 pour la commodité Python)
        self.num_nodes = num_nodes
        self.neighbors = neighbors # Liste des IDs des voisins directs

        # Initialisation de xi et omega pour le nœud courant
        self.xi = np.zeros(num_nodes)
        self.omega = np.full(num_nodes, np.inf)

        # Le nœud est initialement conscient de lui-même
        self.xi[self.node_id] = 1
        self.omega[self.node_id] = 0

        # Pour stocker les valeurs précédentes pour les mises à jour
        self._xi_prev = np.copy(self.xi)
        self._omega_prev = np.copy(self.omega)

    def update_xi_omega(self, all_nodes_xi_prev, all_nodes_omega_prev):
        """
        Exécute une étape des mises à jour distribuées de xi et omega.
        Nécessite les états xi et omega de tous les nœuds de l'étape précédente.
        """
        # Mettre à jour _prev avec les valeurs actuelles avant de calculer les nouvelles
        self._xi_prev = np.copy(self.xi)
        self._omega_prev = np.copy(self.omega)

        # Calcul de xi_new (Équation 1)
        # Inclure le nœud lui-même dans l'ensemble des "voisins" pour le max
        # Car xi_l,j(k) est utilisé pour l'index l.
        # all_nodes_xi_prev est une liste de tableaux numpy, où l'index correspond à l'ID du nœud
        max_xi_val = self._xi_prev[self.node_id] # Commencer avec la propre valeur du nœud
        for neighbor_id in self.neighbors:
            max_xi_val = np.maximum(max_xi_val, all_nodes_xi_prev[neighbor_id])
        self.xi = max_xi_val # Mise à jour de xi pour l'itération k+1


        # Calcul de omega_new (Équation 2)
        # Ici, nous ne pouvons pas simplement copier omega_prev si xi_new est le même
        # Nous devons itérer sur chaque j
        for j in range(self.num_nodes):
            if self.xi[j] > self._xi_prev[j]: # Si la joignabilité a changé (0 -> 1)
                min_omega_neighbor = np.inf
                for neighbor_id in self.neighbors:
                    # Assurez-vous que le voisin a une distance finie au nœud j
                    if all_nodes_omega_prev[neighbor_id][j] != np.inf:
                        min_omega_neighbor = min(min_omega_neighbor, all_nodes_omega_prev[neighbor_id][j] + 1)
                if min_omega_neighbor != np.inf: # Seulement si un chemin a été trouvé
                    self.omega[j] = min_omega_neighbor
            else: # Si la joignabilité est la même (0->0 ou 1->1)
                self.omega[j] = self._omega_prev[j]


class Network:
    def __init__(self, num_nodes, adjacency_list):
        self.num_nodes = num_nodes
        self.nodes = [Node(i, num_nodes, adjacency_list[i]) for i in range(num_nodes)]

    def run_distributed_discovery(self):
        """
        Exécute l'algorithme de découverte de réseau distribué pendant n étapes.
        """
        for k in range(self.num_nodes): # S'exécute pour n étapes (k=0 à n-1)
            # Rassembler les états actuels de xi et omega de tous les nœuds
            # pour la prochaine itération (simule la communication locale)
            all_nodes_xi_current = [node.xi for node in self.nodes]
            all_nodes_omega_current = [node.omega for node in self.nodes]

            print(f"\n--- Étape {k} ---")
            for i, node in enumerate(self.nodes):
                print(f"Nœud {i}: xi={node.xi}, omega={node.omega}")

            # Chaque nœud met à jour ses propres états
            for node in self.nodes:
                node.update_xi_omega(all_nodes_xi_current, all_nodes_omega_current)

        print(f"\n--- Après {self.num_nodes} étapes (k={self.num_nodes-1}) ---")
        for i, node in enumerate(self.nodes):
            print(f"Nœud {i}: xi={node.xi}, omega={node.omega}")


    def calculate_delta(self):
        """
        Calcule la mesure Delta pour chaque arête (i, l).
        Doit être appelée après l'exécution de run_distributed_discovery().
        """
        delta_values = {}
        # Récupérer les valeurs finales d'omega après n étapes
        final_omegas = [node.omega for node in self.nodes]

        for i in range(self.num_nodes):
            for l in self.nodes[i].neighbors:
                # Éviter de calculer la même arête deux fois (par exemple, (0,1) et (1,0))
                if (l, i) in delta_values:
                    continue

                delta_il_for_all_j = np.zeros(self.num_nodes)
                for j in range(self.num_nodes):
                    # omega_i_j(n) et omega_l_j(n)
                    delta_il_for_all_j[j] = final_omegas[i][j] - final_omegas[l][j]
                delta_values[(i, l)] = delta_il_for_all_j
        return delta_values

    def identify_critical_edges(self, delta_values):
        """
        Identifie les arêtes critiques en utilisant le Théorème 2 et la mesure Delta.
        Ceci est une version simplifiée basée sur la description du théorème.
        """
        critical_edges = []
        for i in range(self.num_nodes):
            for l in self.nodes[i].neighbors:
                # Nous traitons chaque arête une fois (i, l)
                if (l, i) in [edge for edge, _ in critical_edges]: # Check if reverse edge already added
                    continue

                is_critical = True
                delta_il_for_all_j = delta_values.get((i, l)) or delta_values.get((l, i))

                if delta_il_for_all_j is None:
                    continue

                # Condition (9) du Théorème 2:
                # pour chaque nœud j ET pour tous les nœuds adjacents i' dans N_i et l' dans N_l,
                # la condition suivante est remplie : Delta_i,j^(il) != 0 ET {Delta_i,j^(ii'), Delta_l,j^(ll')} != {1,1}.
                # L'implémentation de la partie {Delta_i,j^(ii'), Delta_l,j^(ll')} != {1,1} nécessite des Delta imbriqués
                # qui ne peuvent pas être directement calculés sans itérations supplémentaires ou une structure de données plus complexe.
                # Pour cet exemple, nous nous concentrerons sur la condition Delta_i,j^(il) != 0 comme indicateur primaire
                # de l'absence de chemin alternatif simple (équidistance) tel que décrit dans le Lemme 1.
                # Une implémentation complète nécessiterait de recalculer des deltas pour chaque (i,i') et (l,l')
                # et de vérifier toutes les combinaisons, ce qui est très complexe à simuler dans ce cadre.
                # Donc, nous simplifions en nous basant sur la condition de distance non nulle pour le moment.

                # Si Delta_i,j^(il) = 0 pour au moins un j, cela implique un chemin alternatif (Lemme 1, Condition 1)
                # Donc, si tous les Delta_i,j^(il) sont non nuls, cela indique qu'il n'y a pas de nœud équidistant
                # de i et l (sans passer par l'arête e_il).
                # Ceci est une *simplification* de l'interprétation de la condition de criticité du papier
                # La preuve du Théorème 2 indique "Condition (9) explicitly negates the scenarios described in (7) and (8) from Lemma 1".
                # Condition (7) est Delta_i,j^(il)=0. Donc si pour TOUT j, Delta_i,j^(il) != 0, alors le scenario (7) est nié.
                # La condition (8) est plus complexe et nécessite des boucles supplémentaires pour i' et l'.
                # Pour une implémentation simplifiée et la première partie de la condition du Théorème 2:
                if np.any(delta_il_for_all_j == 0):
                    is_critical = False # Il y a un chemin alternatif via un nœud équidistant
                    break # Pas besoin de vérifier le reste des j pour cette arête

                # La seconde partie de la condition (9) est plus complexe:
                # "et {Delta_i,j^(ii'), Delta_l,j^(ll')} != {1,1}"
                # Pour implémenter cela, nous aurions besoin de:
                # 1. Calculer Delta_i,j^(ii') pour chaque voisin i' de i.
                # 2. Calculer Delta_l,j^(ll') pour chaque voisin l' de l.
                # 3. Vérifier si une combinaison donne {1,1}. Si oui, alors l'arête n'est pas critique.
                # Cela impliquerait des appels récursifs ou une structure de données pour stocker
                # les deltas de toutes les paires de nœuds possibles, pas seulement (i,l).
                # Pour l'instant, cette partie est omise pour la concision de l'exemple.
                # La présence de 0 dans delta_il_for_all_j est le cas le plus simple d'un chemin alternatif.

                if is_critical: # Si aucune condition de non-criticité n'a été trouvée
                    critical_edges.append(((i, l), delta_il_for_all_j))
        return critical_edges


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Exemple de G1 du papier (Fig. 1a)
    # Les ID des nœuds sont 1-8 dans le papier, nous les mapperons à 0-7 pour Python.
    # Adjacence list:
    # Node 0 (1): [1, 2, 3]
    # Node 1 (2): [0, 3, 4]
    # Node 2 (3): [0, 1]
    # Node 3 (4): [0, 1, 4]
    # Node 4 (5): [1, 3, 5]
    # Node 5 (6): [4, 6]
    # Node 6 (7): [5, 7]
    # Node 7 (8): [6]

    num_nodes_g1 = 8
    adjacency_list_g1 = {
        0: [1, 2, 3],
        1: [0, 3, 4],
        2: [0, 1],
        3: [0, 1, 4],
        4: [1, 3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6]
    }

    print("Simulating Network G1 (Fig. 1a) - Initial State:")
    network_g1 = Network(num_nodes_g1, adjacency_list_g1)
    network_g1.run_distributed_discovery()

    print("\nCalculating Delta for G1:")
    delta_vals_g1 = network_g1.calculate_delta()
    for edge, delta_vec in delta_vals_g1.items():
        print(f"Delta_{edge[0]+1},{edge[1]+1} (for node {edge[0]+1} w.r.t edge to {edge[1]+1}): {delta_vec}")

    print("\nIdentifying Critical Edges for G1 (Simplified):")
    critical_edges_g1 = network_g1.identify_critical_edges(delta_vals_g1)
    if critical_edges_g1:
        for edge, _ in critical_edges_g1:
            print(f"Critical Edge Detected: ({edge[0]+1}, {edge[1]+1})")
    else:
        print("No critical edges detected based on simplified criteria.")

    print("\n--- Simulating Network G2 (Fig. 1b) ---")
    # G2 (Fig. 1b) a une arête supplémentaire (1,8) ou (0,7) dans notre mapping
    adjacency_list_g2 = {
        0: [1, 2, 3, 7], # Nœud 1 est maintenant connecté au nœud 8
        1: [0, 3, 4],
        2: [0, 1],
        3: [0, 1, 4],
        4: [1, 3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 0] # Nœud 8 est maintenant connecté au nœud 1
    }
    network_g2 = Network(num_nodes_g1, adjacency_list_g2)
    network_g2.run_distributed_discovery()

    print("\nCalculating Delta for G2:")
    delta_vals_g2 = network_g2.calculate_delta()
    for edge, delta_vec in delta_vals_g2.items():
        print(f"Delta_{edge[0]+1},{edge[1]+1} (for node {edge[0]+1} w.r.t edge to {edge[1]+1}): {delta_vec}")

    print("\nIdentifying Critical Edges for G2 (Simplified):")
    critical_edges_g2 = network_g2.identify_critical_edges(delta_vals_g2)
    if critical_edges_g2:
        for edge, _ in critical_edges_g2:
            print(f"Critical Edge Detected: ({edge[0]+1}, {edge[1]+1})")
    else:
        print("No critical edges detected based on simplified criteria.")