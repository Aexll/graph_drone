#!/bin/bash
# Script de lancement rapide pour le simulateur de drones distribués

echo "======================================"
echo "Simulateur de Drones Distribués"
echo "Implémentation selon precisions.txt"
echo "======================================"
echo

# Fonction d'aide
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options disponibles:"
    echo "  gui         Lancer l'interface graphique (par défaut)"
    echo "  text        Mode texte simple"
    echo "  interactive Mode interactif en console"
    echo "  test        Tests de performance" 
    echo "  validate    Tests de validation de l'algorithme"
    echo "  demo        Démonstration des fonctionnalités"
    echo "  help        Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  $0          # Interface graphique"
    echo "  $0 text     # Simulation en mode texte"
    echo "  $0 validate # Valider l'algorithme"
}

# Vérifier que python3 est disponible
if ! command -v python3 &> /dev/null; then
    echo "Erreur: python3 n'est pas installé ou non trouvé dans le PATH"
    exit 1
fi

# Traiter les arguments
case "${1:-gui}" in
    gui)
        echo "Lancement de l'interface graphique..."
        python3 main.py
        ;;
    text)
        echo "Lancement en mode texte..."
        python3 main.py --text
        ;;
    interactive)
        echo "Lancement en mode interactif..."
        python3 main.py --interactive
        ;;
    test)
        echo "Exécution des tests de performance..."
        python3 main.py --test
        ;;
    validate)
        echo "Validation de l'algorithme..."
        python3 test_critical_edges.py
        ;;
    demo)
        echo "Démonstration des fonctionnalités..."
        python3 demo_features.py
        ;;
    all)
        echo "Exécution complète..."
        echo ""
        echo "1. Tests de validation:"
        python3 test_critical_edges.py
        echo ""
        echo "2. Démonstration:"
        python3 demo_features.py
        echo ""
        echo "3. Tests de performance:"
        python3 main.py --test
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Option inconnue: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "Terminé!"
