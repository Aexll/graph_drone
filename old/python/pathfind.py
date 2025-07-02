import numpy as np
import dearpygui.dearpygui as pg
import time
import algograph as ag

# targets = np.array([
#     [0, 0], 
#     [1, 0], 
#     [0, 1.5], 
#     [1, 1.5],
#     [2.2, 0.4],
#     [0.3, 1.5],
# ])

dist_threshold = 1.1
# targets = np.array([
#     np.array([0,0]),
#     np.array([3,0]),
#     np.array([0,3]),
#     np.array([3,4]),
#     np.array([0,4]),
#     np.array([4,0]),
#     ]).astype(np.float32)

# targets = np.array([[0,0],[3,0],[0,3],[3,4], [0,4]]).astype(np.float32)


# targets = np.array([
#     np.array([0,0]),
#     np.array([3,0]),
#     np.array([0,3]),
#     np.array([3,4]),
#     np.array([3.2,3.4]),
#     np.array([1.5,2.2]),
#     np.array([2.5,2.2]),
# ]).astype(np.float32)

targets = np.array([
    np.array([0,0]),
    np.array([3,0]),
    np.array([0,3]),
    np.array([3,4]),
    np.array([3.2,3.4]),
    np.array([1.5,2.2]),
    np.array([3.8,1.2]),
]).astype(np.float32)

targets = np.dot(targets ,np.array([[1,0],[0,-1]]))


barycenter = np.mean(targets, axis=0)

nodes = np.array([barycenter  for _ in range(len(targets))]) 
# nodes -= np.array([[0,1]])



def distance(p1, p2):
    return np.linalg.norm(p1 - p2)



def neighbors(nodes,idx, threshold=dist_threshold):
    # return the set of nodes that are within distof the node at idx
    return set([i for i in range(len(nodes)) if i != idx and distance(nodes[idx], nodes[i]) < threshold])

def is_connected(nodes, p1, p2, threshold=dist_threshold):
    if p2 in neighbors(nodes, p1, threshold):
        return True
    else:
        for n in neighbors(nodes, p1, threshold):
            if is_connected(nodes, n, p2, threshold):
                return True
        return False


def is_fully_connected(nodes, threshold=dist_threshold):
    """
    return ture if there exists a path between any two nodes
    """
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not is_connected(nodes, i, j, threshold=threshold):
                return False
    return True

def edge_count(nodes, idx1, idx2, n, threshold=dist_threshold) -> int:
    """
    count the number of edges between idx1 and idx2 at the n-th step
    """
    if n == 0:
        return 0
    if idx1 == idx2:
        return 0
    if is_connected(nodes, idx1, idx2, threshold=threshold):
        return 1
    else: 
        return 1 + edge_count(nodes, idx1, idx2, n-1, threshold=threshold)



NODE_SPEED = 0.01

def node_step(nodes, idx, target, threshold=dist_threshold, speed=NODE_SPEED):
    """
    move the node at idx towards the target without breaking the connection
    """
    movement =  (target - nodes[idx]) / np.linalg.norm(target - nodes[idx]) * speed 
    for i in neighbors(nodes, idx, threshold):
        if is_critical(nodes, i, idx, threshold):
            if distance(nodes[idx] + movement, nodes[i]) > threshold:
                return nodes[idx]
    return nodes[idx] + movement

def draw_graph(nodes, targets, parent="graph", node_color=(255, 0, 0), target_color=(0, 255, 0), edge_color=(100, 100, 255), node_radius=10, target_radius=10, threshold=dist_threshold):
    """
    Draws the graph of nodes and targets, including edges between connected nodes.
    """
    # Clear previous drawings
    pg.delete_item(parent, children_only=True)

    # Scale and center
    scale = 400
    margin = 100
    win_w = pg.get_viewport_width()
    win_h = pg.get_viewport_height()
    center = np.array([win_w / 2, win_h / 2])

    # Normalize and scale nodes/targets to fit window
    all_points = np.vstack((nodes, targets))
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-5)
    norm_nodes = (nodes - min_xy) / span
    norm_targets = (targets - min_xy) / span

    new_nodes = norm_nodes * (scale) + center - scale/2
    new_targets = norm_targets * (scale) + center - scale/2

    # Draw edges between connected nodes
    for i in range(len(new_nodes)):
        for j in range(i + 1, len(new_nodes)):
            if distance(nodes[i], nodes[j]) < threshold:
                # Check if the edge is critical
                if is_critical(nodes, i, j, threshold):
                    color = (255, 255, 0)  # yellow for critical edge
                else:
                    color = edge_color
                pg.draw_line((new_nodes[i][0], new_nodes[i][1]), (new_nodes[j][0], new_nodes[j][1]), color=color, thickness=2, parent=parent)

    # Draw nodes and targets
    for i in range(len(new_nodes)):
        # Optionally, draw a line from node to its target
        pg.draw_line((new_nodes[i][0], new_nodes[i][1]), (new_targets[i][0], new_targets[i][1]), color=(100, 100, 100), thickness=1, parent=parent)
        # draw targets
        pg.draw_circle((new_nodes[i][0], new_nodes[i][1]), node_radius, color=node_color, fill=node_color, parent=parent)
        pg.draw_circle((new_targets[i][0], new_targets[i][1]), target_radius, color=target_color, fill=target_color, parent=parent)

    # Optionally, label nodes
    for i in range(len(new_nodes)):
        pg.draw_text((new_nodes[i][0]+12, new_nodes[i][1]-12), f"N{i}", color=(255,0,0), size=14, parent=parent)
        pg.draw_text((new_targets[i][0]+12, new_targets[i][1]-12), f"T{i}", color=(0,255,0), size=14, parent=parent)

def is_critical(nodes, i, j, threshold=dist_threshold):
    """
    Returns True if the edge (i, j) is critical for the connectivity of the graph.
    That is, if removing the edge (i, j) disconnects the graph.
    """
    # Build adjacency list without edge (i, j)
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and distance(nodes[u], nodes[v]) < threshold:
                if (u == i and v == j) or (u == j and v == i):
                    continue  # skip the edge to be removed
                adj[u].add(v)
    # BFS to check connectivity
    visited = [False] * n
    queue = [0]
    visited[0] = True
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return not all(visited)

if __name__ == "__main__":
    pg.create_context()
    pg.create_viewport(title="Pathfind", width=1000, height=1000)
    pg.setup_dearpygui()
    with pg.window(label="graph", tag="graph", width=1000, height=1000):
        pass  # Window is created, drawing will be handled below

    pg.show_viewport()

    # Initial draw
    draw_graph(nodes, targets)

    while pg.is_dearpygui_running():
        # Move nodes towards targets
        for i in range(len(nodes)):
            nodes[i] = node_step(nodes, i, targets[i])
            # pass
        draw_graph(nodes, targets)
        pg.render_dearpygui_frame()
        time.sleep(1/60)  # Add delay for ~60 FPS

    pg.destroy_context()


