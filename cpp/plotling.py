import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import time
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image




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




def get_shapes_from_histories(histories) -> list[list[int]]:
    """
    dans un historique il se trouve plieurs graphes ayant les mêmes "formes"
    la forme d'un graphe est définie par le nombre de connections de chaque noeud
    on peut donc définir une forme comme une liste d'entiers qui représente le nombre de connections de chaque noeud
    (l'indice de la liste correspond au noeud et la valeur à son nombre de connections)
    """
    shapes = []
    for history in histories:
        shapes.append(tuple(gx.get_shape(history[-1], dist_threshold)))
    return shapes



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


def plot_shape_error_histograms_with_best_graph(histories, targets, dist_threshold, save_path=None, scale_nodes=20, figsize_per_row=(7, 4)):
    """
    Affiche toutes les formes dans une seule grande figure :
      - chaque ligne = une forme
      - à gauche : image du meilleur graphe (erreur minimale)
      - à droite : histogramme des erreurs pour cette forme
    Si save_path est fourni, sauvegarde la figure complète.
    """
    import errorcalc as ec
    import graphx as gx
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image

    # Collecte des infos pour chaque historique
    graph_infos = []
    for history in histories:
        nodes = history[-1]
        shape = tuple(gx.get_shape(nodes, dist_threshold))
        error = ec.cout_snt(nodes, targets)
        graph_infos.append({'nodes': nodes, 'shape': shape, 'error': error, 'history': history})

    # Regroupement par forme
    from collections import defaultdict
    shape_to_graphs = defaultdict(list)
    for info in graph_infos:
        shape_to_graphs[info['shape']].append(info)

    n_shapes = len(shape_to_graphs)
    if n_shapes == 0:
        print("Aucune forme trouvée.")
        return

    # Taille de la figure : une ligne par forme
    figsize = (figsize_per_row[0], figsize_per_row[1] * n_shapes)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_shapes, 2, width_ratios=[1, 2], wspace=0.3, hspace=0.4)

    for row_idx, (shape, graphs) in enumerate(shape_to_graphs.items()):
        errors = [g['error'] for g in graphs]
        best_idx = int(np.argmin(errors))
        best_graph = graphs[best_idx]['nodes']

        # Partie gauche : image du meilleur graphe
        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.scatter(targets[:, 0], targets[:, 1], color='red', zorder=3)
        adj = ec.nodes_to_matrix(best_graph, dist_threshold)
        n_nodes = targets.shape[0]
        for i1 in range(n_nodes):
            for j1 in range(i1+1, n_nodes):
                if adj[i1, j1] == 1:
                    ax_img.plot([best_graph[i1, 0], best_graph[j1, 0]], [best_graph[i1, 1], best_graph[j1, 1]], color='gray', linewidth=1, zorder=1)
        ax_img.scatter(best_graph[:, 0], best_graph[:, 1], color='blue', s=scale_nodes, zorder=4)
        ax_img.set_xlim(targets[:, 0].min() - 1, targets[:, 0].max() + 1)
        ax_img.set_ylim(targets[:, 1].min() - 1, targets[:, 1].max() + 1)
        ax_img.set_aspect('equal')
        ax_img.axis('off')
        ax_img.set_title(f"Forme: {shape}\nErreur min: {errors[best_idx]:.2f}")

        # Partie droite : histogramme des erreurs
        ax_hist = fig.add_subplot(gs[row_idx, 1])
        ax_hist.bar(range(len(errors)), sorted(errors), color='#03A9F4')
        ax_hist.set_xlabel("Graphes de cette forme")
        ax_hist.set_ylabel("Erreur")
        ax_hist.set_title("Histogramme des erreurs")
        ax_hist.set_xticks([])

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fname = os.path.join(save_path, f"formes_histos.png")
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close(fig)




if __name__ == "__main__":

    import errorcalc as ec

    targets = np.array([
        np.array([0,0]),
        np.array([3,0]),
        np.array([0,3]),
        np.array([3,4]),
        np.array([3.2,3.4]),
        np.array([1.5,2.2]),
        np.array([3.8,1.2]),
    ]).astype(np.float32)

    dist_threshold = 1.1



    # results, histories = ec.multicalc_optimal_graph(targets, dist_threshold, ec.cout_snt, 2, 
    #     ngraphs=100, 
    #     steps=10000, 
    #     mutation_stepsize=0.01,
    #     sampling_size=1,
    #     use_genetic_sampling=True,
    # )
    # # plot_history_trace(histories[0], targets, "results/history/trace.png",steps_per_frame=30)
    # # plot_history_trace_tilemap(histories, targets, "results/history/tilemap.png",sorted_by_error=True)

    import graphx as gx # type: ignore


    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))


    # history = gx.optimize_nodes_history(nodes, targets, dist_threshold, 0.01, 100000, False, True)
    # plot_history_gif(history, targets, "results/history/gif","map.gif",0.01,steps_per_frame=1000)
    # plot_history_trace(history, targets, "results/history/trace","trace.png",steps_per_frame=1)


    # histories = gx.optimize_nodes_history_parallel_old(nodes, targets, dist_threshold, 0.001, 100000, 100)
    # plot_history_gif_tilemap(histories, targets, "results/history/gif","tilemap.gif",sorted_by_error=True,steps_per_frame=10000,show_trace=False)

    histories = gx.optimize_nodes_history_parallel_old(nodes, targets, dist_threshold, 0.01, 100000, 1000)
    # print(gx.get_shape(histories[0][-1], dist_threshold))
    # shapes = get_shapes_from_histories(histories)
    # shape_histogram(shapes)
    plot_shape_error_histograms_with_best_graph(histories, targets, dist_threshold, "results/history/shapes_2", scale_nodes=20, figsize_per_row=(7, 4))
    

