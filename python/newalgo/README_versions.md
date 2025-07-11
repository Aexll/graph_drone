# Algorithme de Verrouillage Décentralisé - Versions disponibles

## 📝 Description

Implémentation d'un algorithme de verrouillage décentralisé permettant aux nœuds d'un graphe de supprimer des arêtes de manière coordonnée sans casser la connectivité.

## 🚀 Versions disponibles

### 1. Version Console Classique (`main.py`)
```bash
python3 main.py
```
- Version originale avec sortie console détaillée
- Affichage des graphiques avec matplotlib
- Tests de scénarios multiples
- Idéal pour comprendre l'algorithme étape par étape

### 2. Version Temps Réel Console (`main_realtime.py`)
```bash
python3 main_realtime.py
```
- Interface console interactive avec curses
- Affichage en temps réel qui se met à jour automatiquement
- Contrôles clavier pour piloter la simulation
- Symboles visuels pour représenter l'état des nœuds

**Contrôles clavier :**
- `c` : Créer nouveau graphe
- `s` : Démarrer simulation
- `a` : Basculer mode automatique
- `n` : Pas manuel
- `x` : Arrêter simulation
- `+/-` : Ajuster vitesse
- `q` : Quitter

### 3. Version Interface Graphique (`main_interactive.py`)
```bash
python3 main_interactive.py
```
- Interface graphique moderne avec DearPyGUI
- Visualisation graphique du réseau
- Contrôles boutons et menus déroulants
- Simulation automatique avec thread séparé
- Légende visuelle des couleurs

**Fonctionnalités GUI :**
- 🔄 Génération de nouveaux graphes
- ▶️ Contrôles de simulation intuitifs
- 🎨 Visualisation colorée des états
- 📊 Statut en temps réel
- 📝 Journal des événements

## 🧠 Algorithme

### Principe
1. **Verrouillage** : Un nœud initie un verrouillage sur une arête
2. **Propagation** : Le verrouillage se propage à tout le graphe connexe
3. **Coordination** : Les nœuds s'envoient des messages de confirmation
4. **Sécurité** : Suppression uniquement quand tous les voisins confirment

### Types de messages
- **LOCK** : Informe du verrouillage d'une arête
- **UNLOCK** : Confirme l'acceptation du verrouillage

### États des nœuds
- **🔵 Libre** : Aucun verrouillage actif
- **🔴 Verrouillé (cible)** : Verrouillé sur l'arête à supprimer
- **🟢 Peut supprimer** : Tous les voisins ont confirmé
- **🟠 Verrouillé (autre)** : Verrouillé sur une autre arête

## 📊 Exemples d'utilisation

### Test simple
```python
# Créer un graphe et des nœuds
G = create_graph()
nodes = create_nodes_from_graph(G)

# Initier un verrouillage
edge = (0, 1)
nodes[0].initiate_lock(edge)

# Simuler jusqu'à la fin
while has_messages():
    simulate_timestep(nodes)
```

### Utilisation interactive
1. Lancez `main_interactive.py`
2. Cliquez sur "Nouveau graphe"
3. Sélectionnez une arête dans la liste
4. Cliquez sur "Démarrer" 
5. Observez la propagation du verrouillage
6. Utilisez "Auto OFF/ON" pour la simulation automatique

## 🔧 Installation

```bash
# Dépendances de base
pip install networkx numpy matplotlib

# Pour l'interface graphique
pip install dearpygui

# Pour l'interface console avancée (optionnel)
# curses est généralement déjà installé sur Linux/Mac
```

## 🏗️ Structure du projet

```
python/newalgo/
├── main.py                 # Version console classique
├── main_realtime.py        # Version temps réel console
├── main_interactive.py     # Version interface graphique
├── main_gui.py            # Version GUI avancée (expérimentale)
└── README.md              # Cette documentation
```

## 🧪 Tests et validation

L'algorithme a été testé avec :
- ✅ Propagation simple du verrouillage
- ✅ Gestion des conflits simultanés  
- ✅ Coordination complète du graphe
- ✅ Auto-déverrouillage sécurisé
- ✅ Préservation de la connectivité

## 🎯 Cas d'usage

- **Réseaux de drones** : Coordination pour suppression sécurisée de liens
- **Systèmes distribués** : Gestion décentralisée des ressources
- **Réseaux de communication** : Maintenance sans interruption de service
- **Algorithmes de consensus** : Synchronisation dans graphes dynamiques
