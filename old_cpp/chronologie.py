"""

dessine une frise chronologique de l'historique des shapes


"""

import numpy as np
import matplotlib.pyplot as plt
import graphx as gx # type: ignore
from getimg import get_mini_graph_image



def generate_chronologie(history, steps, get_image_func, targets, dist_threshold, figsize=(12, 3), bar_height=0.5, img_size=1, img_zoom=0.7):
    """
    Dessine une frise chronologique horizontale de l'évolution des shapes dans l'historique.
    - history : liste de graphes (np.ndarray)
    - steps : nombre d'étapes (longueur de l'historique)
    - get_image_func : fonction pour obtenir l'image d'un graphe
    - targets : positions cibles
    - dist_threshold : seuil pour les arêtes
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np
    import graphx as gx # type: ignore

    # Décomposer l'historique en périodes de shapes
    shapes_dict = gx.decompose_history_by_shape(history, targets, dist_threshold)
    # Pour chaque shape, récupérer age_min et age_max
    periods = []  # (shape_key, age_min, age_max, info)
    for shape_key, info in shapes_dict.items():
        age_min = info['age_min']
        age_max = info['age_max']
        periods.append((shape_key, age_min, age_max, info))
    # Trier par ordre d'apparition
    periods.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=figsize)
    y = 0
    colors = plt.get_cmap('tab10')
    for idx, (shape_key, age_min, age_max, info) in enumerate(periods):
        color = colors(idx % 10)
        # Dessiner la barre horizontale
        ax.barh(y, age_max - age_min + 1, left=age_min, height=bar_height, color=color, alpha=0.6, edgecolor='black')
        # Générer l'image de la shape
        img = get_image_func(info['graph'], targets, dist_threshold, size=img_size, skin="constellation")
        # Positionner l'image au centre de la barre
        x_img = (age_min + age_max) / 2
        imagebox = OffsetImage(img, zoom=img_zoom)
        ab = AnnotationBbox(imagebox, (x_img, y), frameon=True, pad=0, bboxprops=dict(edgecolor='black', boxstyle='round,pad=0.2'))
        ax.add_artist(ab)
        # Afficher le nom de la shape au-dessus
        ax.text(x_img, y + bar_height/2 + 0.1, str(shape_key), ha='center', va='bottom', fontsize=9, rotation=30)
    # Mise en forme
    ax.set_xlim(0, steps)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('Étape')
    ax.set_title('Frise chronologique des shapes')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from getimg import get_mini_graph_image
    from floating import histories_to_shapes_dict_and_transition_history, filter_shapes_dict

    targets = np.array([
        np.array([0,0]),
        np.array([3,0]),
        np.array([0,3]),
        np.array([3,4]),
        np.array([3.2,3.4]),
        np.array([1.5,2.2]),
        np.array([3.8,1.2]),
    ]).astype(np.float32)

    # nodes at the barycenter of the targets
    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))

    dist_threshold = 1.1

    graphs_histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, 0.1, 10000,1,False)

    # gd,gh = histories_to_shapes_dict_and_transition_history(graphs_histories, targets, dist_threshold)
    # gd,gh = filter_shapes_dict(gd, gh, lambda shape_key, info: info['score'] < 3)

    generate_chronologie(graphs_histories[0], len(graphs_histories[0]), get_mini_graph_image, targets, dist_threshold)
