"""

permet de générer des images de graphes facilement

"""


import matplotlib.pyplot as plt
import numpy as np
import graphx as gx # type: ignore
import imageio


SKINS = [
    "default", 
    "stick_dark",
    "stick",
    "dark",
    "black", 
    "constellation",
]

def get_mini_graph_image_from_dict(shapes_dict, targets, dist_threshold, size=1.5, skin="default", error=None):

    nodes = shapes_dict['graph']
    error = shapes_dict['score']
    age_min = shapes_dict['age_min']
    age_max = shapes_dict['age_max']
    age = shapes_dict['age']
    shape_key = shapes_dict['shape']
    return get_mini_graph_image(nodes, targets, dist_threshold, size=size, 
            skin=skin, 
            error=error, 
            age_min=age_min, 
            age_max=age_max, 
            age=age, 
            shape_key=shape_key,
        )




def get_mini_graph_image(nodes, targets, dist_threshold, 
    size=1.5, 
    skin="default", 
    error=None, 
    age_min=None, 
    age_max=None, 
    age=None,
    shape_key=None,
):
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
        offset = 0.2
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
    elif skin == "stick_dark":
        nodes_color = 'white'
        nodes_scale = 10
        node_zorder = 4
        nodes_marker = ' '
        targets_color = 'black'
        targets_marker = ' '
        target_lines = True
        target_lines_width = 0.5
        target_lines_color = (0.5,0.5,0.5,0.5)
        line_width = 2
        line_color = 'white'
        line_zorder = 4
        background_color = (0.1,0.1,0.1,1)
        text_color = 'white'
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
    ax.scatter(nodes[:, 0], nodes[:, 1], color=nodes_color, s=nodes_scale*size, zorder=node_zorder, marker=nodes_marker)
    ax.scatter(targets[:, 0], targets[:, 1], color=targets_color, zorder=3, s=scale_targets, marker=targets_marker)
    ax.set_xlim(targets[:, 0].min() - offset, targets[:, 0].max() + offset)
    ax.set_ylim(targets[:, 1].min() - offset, targets[:, 1].max() + offset)

    if error is not None:
        ax.text(
            (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2, ax.get_ylim()[0], f"{error:.2f}",
            fontsize=size*6,
            zorder=node_zorder+1,
            ha='center',
            va='bottom',
            transform=ax.transData,
            color=text_color,
        )
    if age is not None:
        ax.text(
            (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2, ax.get_ylim()[0] * 2.5, f"{age}",
            fontsize=size*4,
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
    image = image[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA (ce format trop bizare)
    plt.close(fig)
    return image

def save_mini_graph_gif(historique, targets, dist_threshold, size=1.5, skin="default", frames=100, save_path="results.gif", duration=0.08):
    images = []
    len_hist = len(historique)
    for i in range(frames):
        images.append(get_mini_graph_image(historique[i*len_hist//frames], targets, dist_threshold, size=size, skin=skin))
    
    imageio.mimsave(save_path, images, duration=duration)

    return save_path





TEST_IMAGE = True
TEST_GIF = False

if __name__ == "__main__" and TEST_IMAGE:

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

    nodes_histories = gx.optimize_nodes_history(nodes, targets, dist_threshold, 0.1, 10000,False,True)
    best = gx.get_shape_string(gx.get_shape(nodes_histories[-1],dist_threshold))
    dict_nodes_histories = gx.decompose_history_by_shape(nodes_histories, targets, dist_threshold)

    print(dict_nodes_histories[best])
    # testing get_mini_graph_image
    SIZE = 2
    img_default = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE, skin="default")
    img_stick = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE, skin="stick")
    img_black = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE, skin="black")
    img_constellation = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE, skin="constellation")

    img_default_small = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE/2, skin="default")
    img_default_tiny = get_mini_graph_image_from_dict(dict_nodes_histories[best], targets, dist_threshold, size=SIZE/4, skin="default")

    fig, axs = plt.subplots(1, 6, figsize=(12, 12))
    axs[0].imshow(img_default)
    axs[0].set_title("default")
    axs[1].imshow(img_stick)
    axs[1].set_title("stick")
    axs[2].imshow(img_black)
    axs[2].set_title("black")
    axs[3].imshow(img_constellation)
    axs[3].set_title("constellation")
    axs[4].imshow(img_default_small)
    axs[4].set_title("default small")
    axs[5].imshow(img_default_tiny)
    axs[5].set_title("default tiny")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__" and TEST_GIF:
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

    nodes_histories = gx.optimize_nodes_history(nodes, targets, dist_threshold, 0.1, 1000,False,True)

    save_mini_graph_gif(nodes_histories, targets, dist_threshold, size=1.5, skin="default", frames=100, save_path="images/gif/test_gif.gif", duration=0.08)

    print("Gif generated")