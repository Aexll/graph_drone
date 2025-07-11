# Configuration du Simulateur de Drones Distribués

## Installation

```bash
pip install dearpygui
```

## Utilisation

### Interface Graphique
```bash
python3 Simulation.py
```

### Mode Texte (pour tests)
```bash
python3 Simulation.py --text
```

### Tests des Algorithmes
```bash
python3 test_algorithms.py
```

### Démonstrations
```bash
python3 demo.py
```

## Structure du Projet

```
/home/axel/uga/python/standalone_v2/
├── Simulation.py           # Simulateur principal
├── test_algorithms.py      # Tests unitaires
├── demo.py                # Démonstrations
├── README.md              # Documentation
├── algo.md                # Description des algorithmes
└── config.md              # Ce fichier
```

## Paramètres de Simulation

### Interface Graphique

- **Nombre de drones** : 1-50 (défaut: 8)
- **Probabilité de connexion** : 0.0-1.0 (défaut: 0.3)
- **Fréquence de simulation** : Temps réel contrôlé par l'utilisateur

### Algorithmes

- **Phase 1** : Identification structure voisine (n étapes)
- **Phase 2** : Détection arêtes critiques (continue)
- **Estimation de n** : Basée sur la découverte de nœuds

## Codes de Couleur (Interface Graphique)

- 🟡 **Jaune** : Drone en phase d'identification de structure
- 🟢 **Vert** : Drone en phase de détection d'arêtes critiques
- 🔵 **Ligne bleue** : Connexion normale
- 🔴 **Ligne rouge épaisse** : Arête critique détectée

## Performance

- **Complexité temporelle** : O(n) par itération
- **Complexité communication** : O(degré × voisins)
- **Convergence** : n étapes pour la structure, continue pour détection

## Limitations Connues

1. **Estimation de n** : Méthode simple, peut être améliorée
2. **Détection de ponts** : Fonctionne bien pour topologies simples
3. **Communication** : Synchrone uniquement
4. **Topologies dynamiques** : Adaptation basique

## Améliorations Futures

1. **Algorithme d'augmentation d'arêtes** (Algorithme 4 du papier)
2. **Communication asynchrone** avec timestamps
3. **Estimation de n plus sophistiquée** (consensus distribué)
4. **Optimisations de performance**
5. **Interface web** pour démonstrations distantes

## Dépannage

### Problème : Interface graphique ne s'ouvre pas
**Solution** : Utiliser le mode texte `--text` ou vérifier l'environnement graphique

### Problème : Tests échouent
**Solution** : Vérifier les dépendances et relancer `python3 test_algorithms.py`

### Problème : Détection incorrecte d'arêtes critiques
**Solution** : Certaines topologies complexes nécessitent plus d'itérations

## Contact et Contribution

Ce projet implémente les algorithmes distribués décrits dans le papier de recherche
sur la détection d'arêtes critiques et l'augmentation de connectivité dans les réseaux.

Pour contribuer :
1. Tester avec différentes topologies
2. Améliorer l'estimation de n
3. Implémenter l'algorithme 4 (augmentation d'arêtes)
4. Ajouter des métriques de performance
