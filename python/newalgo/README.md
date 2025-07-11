# Algorithme de Verrouillage Décentralisé pour Suppression d'Arêtes

## Description

Cet algorithme permet à des nœuds dans un graphe connexe de supprimer des arêtes de manière coordonnée et décentralisée, sans casser la connectivité locale du graphe. Chaque nœud fonctionne de manière indépendante et ne connaît que ses voisins directs.

## Principe de Fonctionnement

### État des Nœuds

Chaque nœud maintient :
- `is_locked` : État de verrouillage (True/False)
- `locked_edge` : L'arête qui sera supprimée (si verrouillé)
- `unlock_status` : Dictionnaire associant chaque voisin à son état de débloquage

### Types de Messages

1. **Message de Verrouillage (LOCK)** : Informe les voisins qu'une arête va être supprimée
2. **Message de Débloquage (UNLOCK)** : Confirme qu'un nœud accepte la suppression de l'arête

### Algorithme

À chaque pas de temps :

1. **Nœud non verrouillé reçoit un message LOCK** :
   - Se verrouille sur l'arête spécifiée
   - Met tous ses voisins en état "non débloqué"
   - Propage le message LOCK à tous ses voisins

2. **Nœud verrouillé reçoit un message LOCK** :
   - Renvoie un message UNLOCK à l'expéditeur

3. **Nœud verrouillé reçoit un message UNLOCK** :
   - Si l'arête correspond à son arête verrouillée
   - Marque l'expéditeur comme "débloqué"
   - Si tous les voisins sont débloqués → peut supprimer l'arête en sécurité

## Garanties

- **Exclusion mutuelle** : Un seul nœud peut supprimer une arête à la fois
- **Coordination** : Tous les nœuds concernés sont informés avant la suppression
- **Sécurité** : La connectivité locale est préservée

## Utilisation

```python
# Créer le graphe et les nœuds
G = create_graph()
nodes = create_nodes_from_graph(G)

# Initier un verrouillage
test_edge = (0, 5)
nodes[0].initiate_lock(test_edge)

# Simuler l'algorithme
while has_pending_messages():
    simulate_timestep(nodes)
```

## Visualisation

Le programme inclut des fonctions de visualisation pour :
- Afficher l'état initial du graphe
- Visualiser la propagation du verrouillage
- Montrer quels nœuds peuvent supprimer des arêtes en sécurité

## Tests

Le programme teste automatiquement :
- Propagation simple du verrouillage
- Tentatives de verrouillage simultanées
- Coordination entre tous les nœuds du graphe connexe
