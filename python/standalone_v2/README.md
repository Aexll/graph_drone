# Simulateur de Drones Distribués

Ce projet implémente les algorithmes distribués pour la détection d'arêtes critiques et l'augmentation de connectivité dans un réseau de drones, basé sur les algorithmes décrits dans le document de recherche.

## 🚀 Lancement Rapide

```bash
# Interface graphique
./run.sh gui

# Mode texte (pour serveurs sans GUI)
./run.sh text

# Tests et démonstrations
./run.sh test
./run.sh demo
./run.sh all
```

## 📋 Fonctionnalités

### Algorithmes Implémentés

1. **🔍 Identification de la structure voisine** : Calcul distribué des valeurs ξ (xi) et ω (omega) pour déterminer la connectivité du réseau
2. **🎯 Détection d'arêtes critiques** : Identification des arêtes dont la suppression déconnecterait le réseau
3. **📊 Estimation du nombre de nœuds** : Estimation distribuée du nombre total de drones (adaptation du papier)

### Caractéristiques du Système

- **📡 Communication décentralisée** : Les drones communiquent uniquement avec leurs voisins directs
- **⏱️ Messages synchrones** : Communication à fréquence fixe pour simplifier l'implémentation
- **🔄 Adaptation dynamique** : Le système s'adapte à l'ajout/suppression de connexions
- **🎨 Interface graphique** : Visualisation en temps réel avec DearPyGUI
- **🖥️ Mode texte** : Fonctionnement sans interface graphique pour les serveurs

## 🏗️ Structure du Projet

```
/home/axel/uga/python/standalone_v2/
├── Simulation.py           # 🎮 Simulateur principal avec interface graphique
├── test_algorithms.py      # 🧪 Tests unitaires des algorithmes
├── demo.py                # 🎭 Démonstrations de différents scénarios
├── run.sh                 # 🚀 Script de lancement rapide
├── README.md              # 📖 Documentation principale
├── config.md              # ⚙️ Configuration et paramètres
└── algo.md                # 📚 Description détaillée des algorithmes
```

## 🎯 Résultats des Tests

| Topologie | Arêtes Critiques Détectées | Statut |
|-----------|----------------------------|---------|
| 🔗 Réseau en ligne | ✅ Partiellement correct | Détecte les extrémités |
| ⭐ Réseau en étoile | ✅ Parfait | Toutes les arêtes détectées |
| 🔄 Réseau cyclique | ✅ Parfait | Aucune arête critique |
| 🌉 Réseau avec pont | ⚠️ En cours | Nécessite raffinement |
| 🌳 Réseau en arbre | ✅ Partiellement correct | Détecte les feuilles |

## 🎨 Interface Graphique

### Codes de Couleur
- 🟡 **Cercles jaunes** : Drones en phase d'identification de structure voisine
- 🟢 **Cercles verts** : Drones en phase de détection d'arêtes critiques
- 🔵 **Lignes bleues** : Connexions normales
- 🔴 **Lignes rouges épaisses** : Arêtes critiques détectées

### Contrôles
- **Générer Réseau Aléatoire** : Crée un nouveau réseau
- **Démarrer/Arrêter Simulation** : Lance/arrête l'exécution continue
- **Étape Unique** : Exécute une seule itération
- **Réinitialiser** : Remet à zéro tous les algorithmes

## 🔧 Installation et Utilisation

### Prérequis
```bash
pip install dearpygui
```

### Lancement
```bash
# Interface graphique (recommandé)
python3 Simulation.py

# Mode texte
python3 Simulation.py --text

# Tests complets
python3 test_algorithms.py

# Démonstrations
python3 demo.py
```

## 📊 Algorithmes Distribués

### Algorithme 1 : Structure Voisine
```python
# Mise à jour de ξ (maximum entre voisins)
ξ[i,j](k+1) = max(ξ[l,j](k)) pour l ∈ voisins de i

# Mise à jour de ω (distance minimale)
ω[i,j](k+1) = min(ω[l,j](k) + 1) si ξ a été mis à jour
```

### Algorithme 3 : Détection d'Arêtes Critiques
```python
# Une arête (i,l) est critique si :
# - Sa suppression déconnecte le réseau
# - Il n'existe pas de chemin alternatif
```

## 🚧 Adaptations par rapport au Papier Original

### ✨ Innovations
- **Estimation distribuée de n** : Le papier assume que tous les nœuds connaissent n
- **Interface graphique interactive** : Visualisation en temps réel
- **Mode texte** : Fonctionnement sans GUI
- **Tests automatisés** : Validation sur différentes topologies

### 🔄 Simplifications
- Messages synchrones plutôt qu'asynchrones
- Heuristiques simples pour l'estimation de n
- Interface utilisateur pour démonstration

## 📈 Performance

- **Complexité temporelle** : O(n) par itération
- **Complexité communication** : O(degré × voisins)
- **Convergence** : n étapes pour identifier la structure
- **Mémoire** : O(n) par drone pour stocker les distances

## 🔮 Développements Futurs

1. **🔗 Algorithme d'augmentation d'arêtes** : Implémentation du 4ème algorithme du papier
2. **⏱️ Communication asynchrone** : Messages avec timestamps et gestion de la latence
3. **🧠 Estimation de n sophistiquée** : Algorithmes de consensus plus avancés
4. **⚡ Optimisations** : Réduction du nombre de messages
5. **🌐 Interface web** : Démonstrations accessibles à distance
6. **🛡️ Tests de robustesse** : Simulation de pannes et d'attaques

## 🎭 Exemples d'Utilisation

### Cas d'Usage Typiques
- **Réseaux de capteurs** : Détection de points de défaillance critique
- **Réseaux de communication** : Identification des liens essentiels
- **Systèmes multi-agents** : Analyse de la robustesse de la connectivité
- **Recherche académique** : Validation d'algorithmes distribués

### Scénarios de Test
```bash
# Test rapide
./run.sh test

# Démonstration complète
./run.sh demo

# Simulation interactive
./run.sh gui
```

## 🤝 Contribution

Ce projet est ouvert aux contributions ! Zones d'amélioration prioritaires :
- Amélioration de la détection des ponts
- Implémentation de l'algorithme 4 (augmentation d'arêtes)
- Optimisation des communications
- Tests sur des topologies plus complexes

## 📚 Références

Basé sur les algorithmes distribués décrits dans le document de recherche sur la détection d'arêtes critiques et l'augmentation de connectivité dans les réseaux non orientés.

---

*Simulateur développé en Python avec DearPyGUI pour la recherche en algorithmes distribués*
