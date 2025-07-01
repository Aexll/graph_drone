import errorcalc as ec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import graphx as gx # type: ignore


"""

this program will generate some random targets and then will compute the multicalc_optimal_graph for them
plot the results and save the results in a file




"""


def generate_targets(nb_nodes, box_size):
    return np.random.uniform(0, box_size, (nb_nodes, 2)).astype(np.float32)


# print(list(generate_targets(20, 10)))

# quit()

# targets = np.array([[0,0],[3,0],[0,3],[3,4], [0,4]]).astype(np.float32)
targets = np.array([
    np.array([0,0]),
    np.array([3,0]),
    np.array([0,4]),
    np.array([2,1]),
    np.array([2.3,3.2]),
    np.array([4,1]),
    np.array([1,2.5]),
    np.array([3.5,5]),
    ]).astype(np.float32)





# # Paramètres
# NB_NODES = 6
# BOX_SIZE = 10
# NGRAPHS = 100
# STEPS = 20000
# MUTATION_STEPSIZE = 0.4
# SAMPLING_SIZE = 3
# USE_GENETIC_SAMPLING = True
# SCALE_NODES = 10
# DIST_THRESHOLD = 1.1

# Génération des cibles aléatoires
# targets = np.random.uniform(0, BOX_SIZE, (NB_NODES, 2)).astype(np.float32)

def generate_and_plot(
    targets,
    ngraphs=100,
    steps=20000,
    mutation_stepsize=0.4,
    sampling_size=3,
    use_genetic_sampling=True,
    scale_nodes=10,
    dist_threshold=1.1,
    save_prefix="results",
    expo=2,
    save_npz=True,
    save_png=True,
    show_plot=True
):
    print("targets", targets)
    # results, histories = ec.multicalc_optimal_graph(
    #     targets, dist_threshold, ec.cout_snt, expo,
    #     ngraphs=ngraphs,
    #     steps=steps,
    #     mutation_stepsize=mutation_stepsize,
    #     sampling_size=sampling_size,
    #     use_genetic_sampling=use_genetic_sampling
    # )
    nodes = np.array([[np.mean(targets[:, 0]), np.mean(targets[:, 1])]] * len(targets))
    results = gx.optimize_nodes_history_parallel(nodes, 
    targets, 
    dist_threshold, 
    mutation_stepsize,
    steps,
    ngraphs,False)
    # Use the last element of each history as the final result
    final_results = [history[-1] for history in results]
    errors = np.array([ec.cout_snt(result, targets) for result in final_results])
    if save_npz:
        np.savez(f"{save_prefix}.npz", results=final_results, targets=targets, errors=errors)
        print(f"Résultats sauvegardés dans {save_prefix}.npz")
    norm = mcolors.Normalize(vmin=errors.min(), vmax=errors.max())
    cmap = plt.colormaps['viridis']
    colors = [cmap(norm(error)) for error in errors]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax1.scatter(targets[:, 0], targets[:, 1], label='Targets', color='red')
    for result, color in zip(final_results, colors):
        ax1.scatter(result[:, 0], result[:, 1], color=color, s=scale_nodes)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax1, label='Error')
    ax1.set_title('Positions des noeuds colorées par erreur')
    ax2.hist(errors, bins=30, color='gray', edgecolor='black')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogramme des erreurs')
    plt.tight_layout()
    if show_plot:
        plt.show()
    if save_png:
        plt.savefig(f"{save_prefix}.png")
        print(f"Figure sauvegardée dans {save_prefix}.png")
        plt.close(fig)



# Exemple d'utilisation :
if __name__ == "__main__":
    # targets = generate_targets(6, 10)
    # targets = np.array([
    #     np.array([0,0]),
    #     np.array([3,0]),
    #     np.array([0,3]),
    #     np.array([3,4]),
    #     np.array([3.2,3.4]),
    #     np.array([1.5,2.2]),
    #     np.array([3.8,1.2]),
    # ]).astype(np.float32)

    # targets = generate_targets(4,10)



    generate_and_plot(
        targets=targets,
        ngraphs=10,
        steps=1000000,
        mutation_stepsize=0.1,
        sampling_size=10,
        use_genetic_sampling=True,
        scale_nodes=10,
        dist_threshold=1.1,
        save_prefix=f"results7_N1",
        expo=1,
        save_npz=False,
        save_png=False,
        show_plot=True
    )    
    
    # generate_and_plot(
    #     targets=targets,
    #     ngraphs=100,
    #     steps=10000,
    #     mutation_stepsize=0.01,
    #     sampling_size=1,
    #     use_genetic_sampling=True,
    #     scale_nodes=10,
    #     dist_threshold=1.1,
    #     save_prefix=f"results7_N2",
    #     expo=2,
    #     save_png=False,
    #     show_plot=False
    # )