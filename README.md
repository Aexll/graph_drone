# Using graphx

## Install

```bash
./install_graphx.sh
```

## Usage

```python
import graphx as gx

print(gx.version())
```

## Modifications

run this command to build the module, it will automatically build the module, no need to reinstall the module after each modification
```bash
./build_graphx.sh
```

## Documentation

print all the functions available in the module with the following command
```python
print(dir(gx))
```

- `Version() -> None` 

print the version of the module with the following command
```python
print(gx.version())
```

- `distance(a: np.array, b: np.array) -> float`
    - `a` and `b` are numpy arrays of shape (2,)
    - return the distance between the two points
    - the distance is computed using the euclidean distance

- `is_connected(a: np.array, b: np.array, dist_threshold: float) -> bool`
    - `a` et `b` sont des numpy arrays de forme (2,)
    - Retourne True si la distance euclidienne entre `a` et `b` est inférieure à `dist_threshold`.

- `get_adjacency_matrix(nodes: np.array, dist_threshold: float) -> np.array`
    - `nodes` est un numpy array de forme (n, 2)
    - Retourne la matrice d'adjacence du graphe (array d'entiers de forme (n, n)), où chaque entrée vaut 1 si les noeuds sont connectés, 0 sinon.

- `get_node_contact_array(nodes: np.array, start_idx: int, dist_threshold: float) -> list[int]`
    - Retourne la liste des indices des noeuds, ordonnée par le nombre d'arrêts minimum pour atteindre chaque noeud depuis `start_idx`.

- `chose_node_near_node_weighted(nodes: np.array, start_node: np.array, dist_threshold: float, sigma: float) -> int`
    - Sélectionne aléatoirement un noeud parmi `nodes`, avec une probabilité favorisant les noeuds proches de `start_node`.
    - Retourne l'indice du noeud choisi.

- `random_points_in_disk_with_attraction_point(disk_center: np.array, radius: float, attract_point: np.array, sigma: float) -> np.array`
    - Génère un point aléatoire dans un disque de centre `disk_center` et de rayon `radius`, avec une probabilité favorisant la proximité de `attract_point`.
    - Retourne un numpy array de forme (2,).

- `safe_mutate_nodes(nodes: np.array, start_idx: int, radius: float, sigma: float) -> np.array`
    - Génère une nouvelle configuration de noeuds connectés, en partant du noeud `start_idx` et en mutant les autres noeuds dans des disques de rayon `radius`.
    - Retourne un numpy array de forme (n, 2).

- `is_graph_connected_bfs(nodes: np.array, dist_threshold: float) -> bool`
    - Vérifie si le graphe formé par `nodes` est connexe (tous les noeuds sont accessibles).

- `cout_graph_p2(nodes: np.array, targets: np.array) -> float`
    - Calcule le coût (somme des distances au carré) entre chaque noeud et sa cible dans `targets`.

- `mutate_nodes(nodes: np.array, stepsize: float) -> np.array`
    - Retourne une nouvelle configuration de noeuds en ajoutant un bruit gaussien de variance `stepsize` à chaque coordonnée.

- `optimize_nodes(nodes, targets, dist_threshold, stepsize, n, failure_stop_enabled=False) -> np.array`
    - Optimise la position des noeuds pour minimiser le coût par rapport à `targets`, tout en gardant le graphe connexe.
    - `n` est le nombre d'itérations.
    - Retourne la meilleure configuration trouvée.

- `optimize_nodes_parallel_hybrid`, `optimize_nodes_parallel`, `optimize_nodes_history`, `optimize_nodes_history_parallel`
    - Fonctions avancées pour l'optimisation parallèle ou avec historique. Voir le code source pour plus de détails.

- `get_shape(nodes: np.array, dist_threshold: float) -> tuple`
    - Retourne un tuple représentant les arêtes du graphe (paires d'indices connectés).

- `get_shape_distance(shape1: tuple, shape2: tuple) -> int`
    - Retourne le nombre d'arêtes différentes entre deux graphes (distance de forme).

- `get_shape_string(shape: tuple) -> str`
    - Retourne une représentation textuelle compacte d'une forme (utile pour l'analyse d'évolution de graphe).

- `get_shape_string_transition_history(history: list[np.array], dist_threshold: float) -> set[tuple[str, str]]`
    - Retourne l'ensemble des transitions de formes observées dans un historique de graphes.

- `decompose_history_by_shape(history: list[np.array], targets: np.array, dist_threshold: float) -> dict`
    - Décompose un historique de graphes en un dictionnaire associant à chaque forme la meilleure configuration et son score.

- `optimize_nodes_genetic(nodes, targets, dist_threshold, stepsize, n, population_size, keep_best_ratio)`
    - Optimisation par algorithme génétique (avancé).

## Exemples d'utilisation

```python
import numpy as np
import graphx as gx

# Exemple : calcul de la matrice d'adjacence
nodes = np.array([[0, 0], [1, 0], [0, 1]])
adj = gx.get_adjacency_matrix(nodes, dist_threshold=1.5)
print(adj)

# Exemple : optimisation simple
nodes = np.random.rand(5, 2)
targets = np.random.rand(5, 2)
optimized = gx.optimize_nodes(nodes, targets, dist_threshold=1.0, stepsize=0.1, n=100)
print(optimized)
```

Pour plus de détails sur chaque fonction, consultez le code source ou utilisez `help(gx.nom_de_fonction)` dans Python.





