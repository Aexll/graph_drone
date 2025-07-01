import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import time
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import graphx as gx # type: ignore
from matplotlib.patches import FancyArrowPatch


# for a single graph history
def plot_history_gif(history, targets, save_path, gif_name="history.gif", dist_threshold=1.1, scale_nodes=10, steps_per_frame=1):
    """
    Génère un gif de l'évolution du graphe (positions des noeuds + arêtes) au cours de l'historique.
    Visuellement identique au plot de generate_and_plot.
    steps_per_frame : nombre d'étapes de l'historique par frame du gif (pour accélérer ou alléger le gif)
    """
    import errorcalc as ec
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    os.makedirs(save_path, exist_ok=True)
    images = []
    nframes = len(history)
    n_nodes = targets.shape[0]

    # On ne prend qu'une frame toutes les steps_per_frame étapes
    frame_indices = list(range(0, nframes, steps_per_frame))
    if frame_indices[-1] != nframes-1:
        frame_indices.append(nframes-1)  # Toujours inclure la dernière étape

    for idx, t in enumerate(frame_indices):
        nodes = history[t]
        fig, ax = plt.subplots(figsize=(6, 6))
        canvas = FigureCanvas(fig)
        # Affichage des cibles
        ax.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets', zorder=3)
        # Affichage des arêtes
        adj = ec.nodes_to_matrix(nodes, dist_threshold)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj[i, j] == 1:
                    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
        # Affichage des noeuds
        ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
        ax.set_title(f"Evolution du graphe - étape {t+1}/{nframes}")
        ax.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
        ax.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        # Sauvegarde temporaire de la frame
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
        images.append(image)
        plt.close(fig)
    # Sauvegarde du gif
    gif_path = os.path.join(save_path, gif_name)
    imageio.mimsave(gif_path, images, duration=0.08)
    print(f"GIF sauvegardé dans {gif_path}")


def plot_history_trace(history, targets, save_path, img_name="history.png", dist_threshold=1.1, scale_nodes=10, steps_per_frame=1):
    """
    Génère une image de l'évolution du graphe (positions des noeuds + arêtes) à la dernière étape de l'historique.
    avec une ligne qui trace le chemin de chaque noeud de l'historique.
    """
    import errorcalc as ec
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    os.makedirs(save_path, exist_ok=True)
    nframes = len(history)
    n_nodes = targets.shape[0]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_nodes)]  # 10 couleurs distinctes

    fig, ax = plt.subplots(figsize=(6, 6))
    # Affichage des cibles
    ax.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets', zorder=3)
    # Trace le chemin de chaque noeud
    history_arr = np.array(history)  # shape: (nframes, n_nodes, 2)
    for i in range(n_nodes):
        ax.plot(history_arr[:, i, 0], history_arr[:, i, 1], color=colors[i], linewidth=1, alpha=0.7, zorder=2)
    # Affichage des arêtes à la dernière étape
    nodes = history[-1]
    adj = ec.nodes_to_matrix(nodes, dist_threshold)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj[i, j] == 1:
                ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
    # Affichage des noeuds à la dernière étape
    ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
    ax.set_title(f"Evolution du graphe (trace des noeuds)")
    ax.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
    ax.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    # Sauvegarde de l'image
    img_path = os.path.join(save_path, img_name)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Image sauvegardée dans {img_path}")


def plot_history_trace_tilemap(histories, targets, save_path, img_name="history.png", dist_threshold=1.1, scale_nodes=10, steps_per_frame=1, sorted_by_error=False):
    """
    générer une mosaique de l'évolution de tous les graphes de l'historique (trace des noeuds)
    """
    import errorcalc as ec
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    os.makedirs(save_path, exist_ok=True)
    n_graphs = len(histories)
    n_nodes = targets.shape[0]
    img_size = 300  # pixels par graphe
    # Calcul des erreurs finales si tri demandé
    if sorted_by_error:
        errors = [ec.cout_snt(h[-1], targets) for h in histories]
        order = np.argsort(errors)
    else:
        order = np.arange(n_graphs)

    # Détermine la taille de la grille
    n_cols = math.ceil(math.sqrt(n_graphs))
    n_rows = math.ceil(n_graphs / n_cols)

    # Génère toutes les images en mémoire
    images = []
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_nodes)]
    for idx in order:
        history = histories[idx]
        history_arr = np.array(history)
        nodes = history_arr[-1]
        fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
        canvas = FigureCanvas(fig)
        # Cibles
        ax.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3)
        # Traces
        for i in range(n_nodes):
            ax.plot(history_arr[:, i, 0], history_arr[:, i, 1], color=colors[i], linewidth=1, alpha=0.7, zorder=2)
        # Arêtes
        adj = ec.nodes_to_matrix(nodes, dist_threshold)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj[i, j] == 1:
                    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
        # Noeuds finaux
        ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
        ax.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
        ax.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout(pad=0)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(Image.fromarray(img[..., :3]))  # Convert RGBA to RGB
        plt.close(fig)

    # Crée la mosaïque
    tilemap = Image.new('RGB', (n_cols * img_size, n_rows * img_size), (255, 255, 255))
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        tilemap.paste(img, (col * img_size, row * img_size))

    out_path = os.path.join(save_path, img_name)
    tilemap.save(out_path)
    print(f"Mosaïque sauvegardée dans {out_path}")


def plot_history_gif_tilemap(histories, targets, save_path, img_name="tilemap.gif", 
    dist_threshold=1.1, 
    scale_nodes=10, 
    steps_per_frame=1, 
    sorted_by_error=False, 
    duration=0.08,
    show_trace=True,
):
    """
    Génère une mosaïque animée (GIF) de l'évolution de tous les graphes de l'historique (trace des noeuds).
    """
    import errorcalc as ec
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import math
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    os.makedirs(save_path, exist_ok=True)
    n_graphs = len(histories)
    n_nodes = targets.shape[0]
    img_size = 200  # pixels par graphe
    # Calcul des erreurs finales si tri demandé
    if sorted_by_error:
        errors = [ec.cout_snt(h[-1], targets) for h in histories]
        order = np.argsort(errors)
    else:
        order = np.arange(n_graphs)

    # Détermine la taille de la grille
    n_cols = math.ceil(math.sqrt(n_graphs))
    n_rows = math.ceil(n_graphs / n_cols)

    # Génère toutes les séquences d'images en mémoire
    all_frames = []  # all_frames[g][f] = image PIL
    max_len = 0
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_nodes)]
    for idx in order:
        history = histories[idx]
        history_arr = np.array(history)
        nframes = len(history_arr)
        frame_indices = list(range(0, nframes, steps_per_frame))
        if frame_indices[-1] != nframes-1:
            frame_indices.append(nframes-1)
        frames = []
        for t in frame_indices:
            nodes = history_arr[t]
            fig, ax = plt.subplots(figsize=(img_size/100, img_size/100), dpi=100)
            canvas = FigureCanvas(fig)
            # Cibles
            ax.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3)
            # Traces jusqu'à t
            if show_trace:
                for i in range(n_nodes):
                    ax.plot(history_arr[:t+1, i, 0], history_arr[:t+1, i, 1], color=colors[i], linewidth=1, alpha=0.7, zorder=2)
            # Arêtes
            adj = ec.nodes_to_matrix(nodes, dist_threshold)
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    if adj[i, j] == 1:
                        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
            # Noeuds finaux
            ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
            ax.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
            ax.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
            ax.set_aspect('equal')
            ax.axis('off')
            fig.tight_layout(pad=0)
            canvas.draw()
            img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(Image.fromarray(img[..., :3]))  # Convert RGBA to RGB
            plt.close(fig)
        all_frames.append(frames)
        if len(frames) > max_len:
            max_len = len(frames)

    # Complète les séquences pour qu'elles aient toutes la même longueur
    for i in range(len(all_frames)):
        while len(all_frames[i]) < max_len:
            all_frames[i].append(all_frames[i][-1])

    # Pour chaque frame, assemble la mosaïque
    tilemap_frames = []
    for f in range(max_len):
        tile = Image.new('RGB', (n_cols * img_size, n_rows * img_size), (255, 255, 255))
        for idx, frames in enumerate(all_frames):
            row = idx // n_cols
            col = idx % n_cols
            tile.paste(frames[f], (col * img_size, row * img_size))
        tilemap_frames.append(tile)

    out_path = os.path.join(save_path, img_name)
    tilemap_frames[0].save(out_path, save_all=True, append_images=tilemap_frames[1:], duration=int(duration*1000), loop=0)
    print(f"Mosaïque GIF sauvegardée dans {out_path}")




# def get_shapes_from_histories(histories) -> list[list[int]]:
#     """
#     dans un historique il se trouve plieurs graphes ayant les mêmes "formes"
#     la forme d'un graphe est définie par le nombre de connections de chaque noeud
#     on peut donc définir une forme comme une liste d'entiers qui représente le nombre de connections de chaque noeud
#     (l'indice de la liste correspond au noeud et la valeur à son nombre de connections)
#     """
#     shapes = []
#     for history in histories:
#         shapes.append(tuple(gx.get_shape(history[-1], dist_threshold)))
#     return shapes



def shape_histogram(shapes):
    """
    crée un histogramme des formes (nombre de graphes ayant une forme donnée)
    """
    # Convert shapes to strings for proper counting since tuples can't be directly counted
    shape_strings = [str(shape) for shape in shapes]
    
    # Count occurrences of each shape
    from collections import Counter
    shape_counts = Counter(shape_strings)
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(shape_counts)), list(shape_counts.values()))
    plt.xlabel("Forme")
    plt.ylabel("Nombre de graphes")
    plt.title("Histogramme des formes")
    
    # Set x-axis labels to show the actual shapes
    plt.xticks(range(len(shape_counts)), list(shape_counts.keys()), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_shape_error_histograms_with_best_graph(graphs, targets, dist_threshold, save_path=None, scale_nodes=20, figsize_per_row=(7, 4), 
    n_bins=20, 
    multicolor_nodes=False, 
    target_marker='x',
    target_color=(1,0,0,0.3),
    target_scale=10,
    node_use_cmap=False,
    node_cmap='Spectral',
    node_marker='o',
    target_same_color=False,
    ):
    """
    Affiche toutes les formes dans une seule grande figure :
      - chaque ligne = une forme
      - à gauche : image du meilleur graphe (erreur minimale)
      - à droite : histogramme des erreurs pour cette forme
    Si save_path est fourni, sauvegarde la figure complète.
    """
    import errorcalc as ec
    import graphx as gx # type: ignore
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image

    # Collecte des infos pour chaque historique
    graph_infos = []
    for graph in graphs:
        shape = gx.get_shape_string(gx.get_shape(graph, dist_threshold))
        error = ec.cout_snt(graph, targets)
        graph_infos.append({'nodes': graph, 'shape': shape, 'error': error})

    # Regroupement par forme
    from collections import defaultdict
    shape_to_graphs = defaultdict(list)
    for info in graph_infos:
        shape_to_graphs[info['shape']].append(info)

    # sort shapes by error
    for shape_list in shape_to_graphs.values():
        shape_list.sort(key=lambda x: x['error'])

    n_shapes = len(shape_to_graphs)
    if n_shapes == 0:
        print("Aucune forme trouvée.")
        return

    # Taille de la figure : une ligne par forme
    figsize = (figsize_per_row[0], figsize_per_row[1] * n_shapes)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_shapes, 2, width_ratios=[1, 2], wspace=0.3, hspace=0.4)

    # sort shapes by error
    shape_to_graphs_item_sorted = sorted(shape_to_graphs.items(), key=lambda x: x[1][0]['error'])


    for row_idx, (shape, graphs) in enumerate(shape_to_graphs_item_sorted):
        errors = [g['error'] for g in graphs]
        best_idx = int(np.argmin(errors))
        best_graph = graphs[best_idx]['nodes']
        n_nodes = targets.shape[0]
        hsv_colors = [plt.get_cmap(node_cmap)(i / n_nodes) for i in range(n_nodes)] 
        colors = [tuple(c for c in color[:3]) for color in hsv_colors]

        # Partie gauche : image du meilleur graphe
        ax_img = fig.add_subplot(gs[row_idx, 0])
        if target_same_color:
            ax_img.scatter(targets[:, 0], targets[:, 1], color=colors, zorder=3,marker=target_marker,s=target_scale)
        else:
            ax_img.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3,marker=target_marker,s=target_scale)
        adj = ec.nodes_to_matrix(best_graph, dist_threshold)
        for i1 in range(n_nodes):
            for j1 in range(i1+1, n_nodes):
                if adj[i1, j1] == 1:
                    ax_img.plot([best_graph[i1, 0], best_graph[j1, 0]], [best_graph[i1, 1], best_graph[j1, 1]], color='gray', linewidth=1, zorder=1)
        if not node_use_cmap:
            ax_img.scatter(best_graph[:, 0], best_graph[:, 1], color='blue', s=scale_nodes, zorder=4,marker=node_marker)
        else:
            ax_img.scatter(best_graph[:, 0], best_graph[:, 1], color=colors, s=scale_nodes, zorder=4,marker=node_marker,edgecolors='black',linewidths=0.2)
        ax_img.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
        ax_img.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
        ax_img.set_aspect('equal')
        ax_img.axis('off')
        ax_img.set_title(f"Forme: {shape}\nErreur min: {errors[best_idx]:.2f}")

        # Partie droite : histogramme des erreurs
        # On veut un histogramme avec l'axe x = erreur (même échelle pour tous), y = nombre de graphes
        # On calcule les bins sur toutes les erreurs de toutes les formes
        if row_idx == 0:
            # On calcule les bins globaux une seule fois
            all_errors = [g['error'] for graphs in shape_to_graphs.values() for g in graphs]
            min_error = min(all_errors)
            max_error = max(all_errors)
            min_number_of_graphs = 0
            max_number_of_graphs = max(len(graphs) for graphs in shape_to_graphs.values())
            bins = np.linspace(min_error, max_error, n_bins + 1)
        ax_hist = fig.add_subplot(gs[row_idx, 1])
        ax_hist.hist(errors, bins=bins, color='#03A9F4', edgecolor='black')
        ax_hist.set_xlabel("Erreur")
        ax_hist.set_ylabel("Nombre de graphes")
        ax_hist.set_title("Histogramme des erreurs")
        ax_hist.set_xlim(min_error, max_error)
        ax_hist.set_ylim(min_number_of_graphs, max_number_of_graphs)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"formes_histos.png")
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def shape_evolution_plot(history, targets, dist_threshold, save_path=None, scale_nodes=20, figsize_per_shape=(4, 4), show_error=True, only_first_occurrence=True):
    """
    Affiche l'évolution des shapes d'un historique de graphes :
      - chaque colonne = une shape rencontrée (dans l'ordre d'apparition ou d'amélioration)
      - pour chaque shape, affiche le graphe correspondant (meilleur ou premier)
      - en dessous, le nom de la shape et l'erreur
    """
    import graphx as gx # type: ignore
    import errorcalc as ec
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    n_steps = len(history)
    n_nodes = targets.shape[0]

    # Pour chaque étape, calcule la shape et l'erreur
    step_infos = []
    for i, nodes in enumerate(history):
        shape = gx.get_shape_string(gx.get_shape(nodes, dist_threshold))
        error = ec.cout_snt(nodes, targets)
        step_infos.append({'step': i, 'nodes': nodes, 'shape': shape, 'error': error})

    # Sélectionne pour chaque shape la meilleure (ou la première)
    shape_to_info = {}
    for info in step_infos:
        shape = info['shape']
        if shape not in shape_to_info:
            shape_to_info[shape] = info
        elif not only_first_occurrence:
            # Si on veut la meilleure, on garde celle avec l'erreur minimale
            if info['error'] < shape_to_info[shape]['error']:
                shape_to_info[shape] = info

    # Trie les shapes par ordre d'apparition (ou d'amélioration)
    ordered_shapes = []
    seen = set()
    for info in step_infos:
        shape = info['shape']
        if shape not in seen and shape in shape_to_info:
            ordered_shapes.append(shape)
            seen.add(shape)

    n_shapes = len(ordered_shapes)
    if n_shapes == 0:
        print("Aucune forme trouvée dans l'historique.")
        return

    figsize = (figsize_per_shape[0] * n_shapes, figsize_per_shape[1])
    fig, axes = plt.subplots(1, n_shapes, figsize=figsize, squeeze=False)
    axes = axes[0]  # 1D array

    for idx, shape in enumerate(ordered_shapes):
        info = shape_to_info[shape]
        nodes = info['nodes']
        error = info['error']
        ax = axes[idx]
        # Affichage des cibles
        ax.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3, label='Targets')
        # Affichage des arêtes
        adj = ec.nodes_to_matrix(nodes, dist_threshold)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if adj[i, j] == 1:
                    ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
        # Affichage des noeuds
        ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
        ax.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
        ax.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        # Titre avec shape et erreur
        title = f"Shape: {shape}"
        if show_error:
            title += f"\nErreur: {error:.2f}"
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"shape_evolution.png")
        plt.savefig(fname, bbox_inches='tight')
        print(f"Frise des shapes sauvegardée dans {fname}")
    plt.show()
    plt.close(fig)





# def draw_shape_graph(ax, nodes, adj, node_color='blue', target_color='red', scale_nodes=20):
#     # Dessine un petit graphe sur un axe donné
#     ax.scatter(nodes[:, 0], nodes[:, 1], color=target_color, zorder=3)
#     n_nodes = nodes.shape[0]
#     for i in range(n_nodes):
#         for j in range(i+1, n_nodes):
#             if adj[i, j] == 1:
#                 ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
#     ax.scatter(nodes[:, 0], nodes[:, 1], color=node_color, s=scale_nodes, zorder=4)
#     ax.axis('off')
#     ax.set_aspect('equal')

def draw_mini_graph(ax, nodes, targets, dist_threshold, size=1.5, scale_nodes=20, scale_targets=20, offset=0.2):
    # import graphx as gx # type: ignore
    adj = gx.get_adjacency_matrix(nodes, dist_threshold)
    for i in range(nodes.shape[0]):
        for j in range(i+1, nodes.shape[0]):
            if adj[i, j] == 1:
                ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], color='gray', linewidth=1, zorder=1)
    ax.scatter(nodes[:, 0], nodes[:, 1], color='blue', s=scale_nodes, zorder=4)
    ax.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3, s=scale_targets)
    ax.set_xlim(targets[:, 0].min() - offset, targets[:, 0].max() + offset)
    ax.set_ylim(targets[:, 1].min() - offset, targets[:, 1].max() + offset)
    ax.axis('off')
    ax.set_aspect('equal')

def get_graph_image(nodes, targets, dist_threshold, size=1.5):
    # Crée une image matplotlib du graphe et la retourne comme array
    fig, ax = plt.subplots(figsize=(size, size))
    draw_mini_graph(ax, nodes, targets, dist_threshold)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8') # type: ignore
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert ARGB to RGBA for matplotlib OffsetImage
    image = image[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
    plt.close(fig)
    return image


def get_mini_graph_image(nodes, targets, dist_threshold, size=1.5, skin="default", error=None):
    nodes_scale = 20
    node_zorder = 4
    nodes_marker = 'o'
    scale_targets = 20
    offset = 0.2
    nodes_color = 'blue'
    targets_color = 'red'
    line_width = 1
    line_color = 'gray'
    line_zorder = 1
    line_style = '-'
    target_lines = False
    target_lines_width = 0.5
    target_lines_color = (1,0,0,0.5)
    targets_marker = 'o'
    background_color = 'white'
    text_color = 'black'
    if skin == "small":
        nodes_scale = 10
        scale_targets = 10
        offset = 0.1
        nodes_color = 'blue'
        targets_color = 'red'
        line_width = 0.5
    elif skin == "stick":
        nodes_color = 'black'
        nodes_scale = 10
        node_zorder = 4
        nodes_marker = ' '
        targets_color = 'white'
        target_lines = True
        target_lines_width = 0.5
        target_lines_color = (0,0,0,0.2)
        line_width = 2
        line_color = 'black'
        line_zorder = 4
    elif skin == "black":
        nodes_color = 'black'
        nodes_scale = 30
        node_zorder = 4
        nodes_marker = 'num'
        targets_color = 'white'
        line_width = 0.5
        line_color = 'black'
        line_zorder = 4
    elif skin == "constellation":
        nodes_color = 'yellow'
        nodes_scale = 30
        nodes_marker = '*'
        node_zorder = 4
        targets_color = 'black'
        targets_marker = ' '
        line_width = 0.5
        line_color = 'yellow'
        line_style = '--'
        line_zorder = 4
        background_color = (0.1,0,0.5,1)
        text_color = 'yellow'

    fig, ax = plt.subplots(figsize=(size, size))
    adj = gx.get_adjacency_matrix(nodes, dist_threshold)
    for i in range(nodes.shape[0]):
        for j in range(i+1, nodes.shape[0]):
            if adj[i, j] == 1:
                ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]], 
                color=line_color, 
                linewidth=line_width, 
                zorder=line_zorder, 
                linestyle=line_style,
                )
    if target_lines:
        for i in range(nodes.shape[0]):
            ax.plot([nodes[i, 0], targets[i, 0]], [nodes[i, 1], targets[i, 1]], color=target_lines_color, linewidth=target_lines_width, zorder=line_zorder)
    
    if nodes_marker == "num":
        for i in range(nodes.shape[0]):
            txt_size = nodes_scale/3
            ax.text(
                nodes[i, 0], nodes[i, 1], str(i),
                fontsize=txt_size,
                zorder=node_zorder+1,
                ha='center',
                va='center',
            )
        nodes_marker = 'o'
        nodes_color = 'white'
    ax.scatter(nodes[:, 0], nodes[:, 1], color=nodes_color, s=nodes_scale, zorder=node_zorder, marker=nodes_marker)
    ax.scatter(targets[:, 0], targets[:, 1], color=targets_color, zorder=3, s=scale_targets, marker=targets_marker)
    ax.set_xlim(targets[:, 0].min() - offset, targets[:, 0].max() + offset)
    ax.set_ylim(targets[:, 1].min() - offset, targets[:, 1].max() + offset)

    if error is not None:
        ax.text(
            (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2, ax.get_ylim()[0] - offset, f"{error:.2f}",
            fontsize=size*6,
            zorder=node_zorder+1,
            ha='center',
            va='bottom',
            transform=ax.transData,
            color=text_color,
        )

    ax.axis('off')
    ax.set_aspect('equal')
    fig.set_facecolor(background_color)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8') # type: ignore
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert ARGB to RGBA for matplotlib OffsetImage
    image = image[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
    plt.close(fig)
    return image







def plot_shape_transition_graph_with_mini_graphs(graphs_histories, targets, dist_threshold, shapes_dict=None, figsize=(12, 8), mini_graph_size=1, mini_graph_zoom=0.5, seed=42, spring_k=2.0):
    """
    Construit et affiche le graphe dirigé des transitions de shapes, avec pour chaque noeud une miniature du graphe correspondant (meilleur score).
    - graphs_histories : liste d'historiques de graphes (list of list of np.ndarray)
    - targets : positions cibles
    - dist_threshold : seuil de distance pour les arêtes
    - shapes_dict : dictionnaire optionnel {shape_key: {'graph': nodes, 'score': float}}
    - spring_k : paramètre d'espacement du layout spring_layout (plus grand = plus espacé)
    """
    import graphx as gx # type: ignore
    import networkx as nx
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute all transitions from each graph's history
    full_transitions = set()
    shapes = {} if shapes_dict is None else dict(shapes_dict)
    for graph_history in graphs_histories:
        full_transitions.update(gx.get_shape_string_transition_history(graph_history, dist_threshold))
        new_shapes = gx.decompose_history_by_shape(graph_history, targets, dist_threshold)
        for shape_key in new_shapes.keys():
            if shape_key not in shapes:
                shapes[shape_key] = new_shapes[shape_key]
            else:
                if new_shapes[shape_key]['score'] < shapes[shape_key]['score']:
                    shapes[shape_key] = new_shapes[shape_key]

    # Build a directed graph from the transitions
    G = nx.DiGraph()
    for from_shape, to_shape in full_transitions:
        G.add_edge(from_shape, to_shape)

    # Draw the graph
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed, k=spring_k)
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color='white', ax=ax, edge_color='gray', arrowsize=20)

    for shape_key, (node, (x, y)) in zip(shapes.keys(), pos.items()):
        nodes = shapes[shape_key]['graph']
        img = get_graph_image(nodes, targets, dist_threshold, size=mini_graph_size)
        imagebox = OffsetImage(img, zoom=mini_graph_zoom)
        # Add a white outline by setting frameon=True and customizing the bboxprops
        ab = AnnotationBbox(
            imagebox, (x, y),
            frameon=True,
            bboxprops=dict(edgecolor='black', linewidth=1, boxstyle='round,pad=0.2', facecolor='yellow')
        )
        ax.add_artist(ab)

    plt.title("Shape String Transition Graph (with mini-graphs)")
    plt.tight_layout()
    plt.show()



def filter_shapes_dict(shapes_dict, transition_history, filter_func=lambda shape_key, info: True):

        new_shapes_dict = {}
        new_shapes_transition_history = set()

        for shape_key, info in shapes_dict.items():
            if filter_func(shape_key, info):
                new_shapes_dict[shape_key] = info

        for transition in transition_history:
            if transition[0] in new_shapes_dict and transition[1] in new_shapes_dict:
                new_shapes_transition_history.add(transition)

        return new_shapes_dict, new_shapes_transition_history

def histories_to_shapes_dict_and_transition_history(histories, targets, dist_threshold):
        shapes_dict = {}
        transition_history = set()
        for history in histories:
            gh = gx.get_shape_string_transition_history(history, dist_threshold)
            gd = gx.decompose_history_by_shape(history, targets, dist_threshold)
            for shape_key, info in gd.items():
                if shape_key not in shapes_dict:
                    shapes_dict[shape_key] = info
                else:
                    if info['score'] < shapes_dict[shape_key]['score']:
                        shapes_dict[shape_key] = info
            transition_history.update(gh)
        return shapes_dict, transition_history

def plot_shape_frise_with_transitions(
    shapes_dict,  # dict: shape_key -> {'graph': nodes, 'score': float}
    transitions,  # set of (shape_key_from, shape_key_to)
    targets,      # positions cibles (pour dessiner les mini-graphes)
    dist_threshold,
    figsize=(12, 4),
    mini_graph_size=1.5,
    mini_graph_zoom=0.5,
    y=0,  # position verticale fixe pour tous les shapes
    show_error=True,
    margin=0.5,  # espace horizontal minimal entre deux shapes (sera recalculé dynamiquement)
    font_size=10,
    arrow_style='-|>',
    arrow_color='gray',
    node_outline_color='black',
    node_outline_width=1,
    node_box_color='yellow',
    error_fmt='{:.2f}',
    zigzag_amplitude=0.4,  # amplitude du zigzag vertical
    zigzag=True,           # activer le zigzag
):
    """
    Dessine une frise horizontale de mini-graphes ordonnés par erreur, avec transitions entre shapes.
    - shapes_dict : {shape_key: {'graph': nodes, 'score': float}}
    - transitions : set de (shape_key_from, shape_key_to)
    - targets : positions cibles
    - dist_threshold : seuil pour les arêtes
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import numpy as np

    # 1. Trier les shapes par erreur croissante
    sorted_shapes = sorted(shapes_dict.items(), key=lambda x: x[1]['score'])
    n = len(sorted_shapes)
    if n == 0:
        print("Aucune shape à afficher.")
        return

    # 2. Calculer la taille réelle d'un minigraphe (en inch)
    # On crée une image temporaire pour estimer la taille
    nodes_sample = sorted_shapes[0][1]['graph']
    img_sample = get_graph_image(nodes_sample, targets, dist_threshold, size=mini_graph_size)
    dpi = 100  # valeur typique pour matplotlib
    img_px = img_sample.shape[0]  # hauteur en pixels (carré)
    img_inch = img_px / dpi * mini_graph_zoom
    # Espacement horizontal minimal = taille image + marge
    min_margin = img_inch * 1.2  # 20% de marge autour

    # 3. Déterminer les positions horizontales (x)
    xs = np.arange(n) * min_margin

    # 4. Décalage vertical (zigzag ou non)
    if zigzag:
        ys = np.array([y + ((-1)**i) * zigzag_amplitude for i in range(n)])
    else:
        ys = np.full(n, y)

    # 5. Associer chaque shape_key à sa position (x, y)
    shape_to_pos = {shape_key: (x, y_) for (shape_key, _), x, y_ in zip(sorted_shapes, xs, ys)}

    # 6. Ajuster la taille de la figure si besoin
    total_width = xs[-1] - xs[0] + 2 * min_margin
    fig_height = max(figsize[1], 2 * (img_inch + zigzag_amplitude))
    fig, ax = plt.subplots(figsize=(total_width, fig_height))
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(xs[0] - min_margin, xs[-1] + min_margin)
    ax.axis('off')

    # 7. Dessiner les mini-graphes
    for (shape_key, info), x, y_ in zip(sorted_shapes, xs, ys):
        nodes = info['graph']
        img = get_graph_image(nodes, targets, dist_threshold, size=mini_graph_size)
        imagebox = OffsetImage(img, zoom=mini_graph_zoom)
        ab = AnnotationBbox(
            imagebox, (x, y_),
            frameon=True,
            bboxprops=dict(edgecolor=node_outline_color, linewidth=node_outline_width, boxstyle='round,pad=0.2', facecolor=node_box_color)
        )
        ax.add_artist(ab)
        # Afficher l'erreur sous le minigraphe
        if show_error:
            ax.text(x, y_ - (img_inch * 0.7), error_fmt.format(info['score']), ha='center', va='top', fontsize=font_size)
        # Afficher la shape string au-dessus
        ax.text(x, y_ + (img_inch * 0.7), str(shape_key), ha='center', va='bottom', fontsize=font_size-2, rotation=30)

    # 8. Dessiner les transitions (arrows)
    for idx, (from_shape, to_shape) in enumerate(transitions):
        if from_shape in shape_to_pos and to_shape in shape_to_pos:
            x0, y0 = shape_to_pos[from_shape]
            x1, y1 = shape_to_pos[to_shape]
            if abs(x1 - x0) < 1e-6 and abs(y1 - y0) < 1e-6:
                continue  # pas de flèche sur place
            # Rayon de la courbe : positif si x1>x0, négatif sinon, et alterne selon l'indice
            rad = 0.4 * (1 if x1 > x0 else -1)
            # Pour alterner le sens des arcs si plusieurs transitions similaires
            if idx % 2 == 1:
                rad = -rad
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle=arrow_style,
                color=arrow_color,
                lw=1.5,
                mutation_scale=15,
            )
            ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig("shape_frise_with_transitions.png")
    # plt.show()




if __name__ == "__main__" and True:

    targets = np.array([
        np.array([0,0]),
        np.array([3,0]),
        np.array([0,3]),
        np.array([3,4]),
        np.array([3.2,3.4]),
        np.array([1.5,2.2]),
        np.array([3.8,1.2]),
    ]).astype(np.float32)

    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))

    dist_threshold = 1.1

    nodes_histories = gx.optimize_nodes_parallel_hybrid(nodes, targets, dist_threshold, 0.1, 10000, 100)

    # testing get_mini_graph_image
    img_default = get_mini_graph_image(nodes_histories[0], targets, dist_threshold, size=1.5, skin="default", error=gx.cout_graph_p2(nodes_histories[0], targets))
    img_stick = get_mini_graph_image(nodes_histories[0], targets, dist_threshold, size=1.5, skin="stick", error=gx.cout_graph_p2(nodes_histories[0], targets))
    img_black = get_mini_graph_image(nodes_histories[0], targets, dist_threshold, size=1.5, skin="black", error=gx.cout_graph_p2(nodes_histories[0], targets))
    img_constellation = get_mini_graph_image(nodes_histories[0], targets, dist_threshold, size=1.5, skin="constellation", error=gx.cout_graph_p2(nodes_histories[0], targets))

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].imshow(img_default)
    axs[0].set_title("default")
    axs[1].imshow(img_stick)
    axs[1].set_title("stick")
    axs[2].imshow(img_black)
    axs[2].set_title("black")
    axs[3].imshow(img_constellation)
    axs[3].set_title("constellation")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__" and False:
    # targets = np.array([
    #     np.array([0,0]),
    #     np.array([3,0]),
    #     np.array([0,3]),
    #     np.array([3,4]),
    #     np.array([3.2,3.4]),
    #     np.array([1.5,2.2]),
    #     np.array([3.8,1.2]),
    # ]).astype(np.float32)
    
    targets = np.array([
        np.array([0,0]),
        np.array([3,0]),
        np.array([0,3]),
        np.array([3,4]),
    ]).astype(np.float32)

    # nodes at the barycenter of the targets
    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))

    dist_threshold = 1.1

    graphs_histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, 0.1, 10000,10,False)

    gd,gh = histories_to_shapes_dict_and_transition_history(graphs_histories, targets, dist_threshold)
    gd,gh = filter_shapes_dict(gd, gh, lambda shape_key, info: info['score'] < 3.2)



    plot_shape_frise_with_transitions(gd, gh, targets, dist_threshold, 
    figsize=(12, 4), 
    mini_graph_size=1.5, 
    mini_graph_zoom=0.5, 
    y=0, 
    show_error=True, 
    margin=0.5, 
    font_size=10, 
    arrow_style='-|>', 
    arrow_color='gray', 
    node_outline_color='black', 
    node_outline_width=1, 
    node_box_color='yellow', 
    error_fmt='{:.2f}',
    zigzag_amplitude=0.8,
    zigzag=True,
    )