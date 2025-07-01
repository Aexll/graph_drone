import graphx as gx
import numpy as np

LOG = True
DETAILS = False

def print_log(message,color="white",force=False, end="\n"):
    if LOG or force:
        if color == "white":
            print(message, end=end)
        elif color == "green":
            print(f"\033[92m{message}\033[0m", end=end)
        elif color == "red":
            print(f"\033[91m{message}\033[0m", end=end)
        elif color == "yellow":
            print(f"\033[93m{message}\033[0m", end=end)
        elif color == "blue":
            print(f"\033[94m{message}\033[0m", end=end)
        elif color == "purple":
            print(f"\033[95m{message}\033[0m", end=end)
        elif color == "cyan":
            print(f"\033[96m{message}\033[0m", end=end)

def assert_log(condition,message,color="red"):
    if not condition:
        print_log(message,color=color)
        return False
    return True

def test_distance():
    print_log("testing distance > ", end="" if not DETAILS else "\n")
    a = np.array([1, 2])
    b = np.array([3, 4])
    if DETAILS:
        print_log(f"distance between {a} and {b} is {gx.distance(a, b)}",color="green")
    return assert_log(gx.distance(a, b) == np.linalg.norm(a - b),"distance function is not working")

def test_is_connected():
    print_log("testing is_connected > ", end="" if not DETAILS else "\n")
    a = np.array([1, 2])
    b = np.array([3, 4])
    t = 3
    if DETAILS:
        print_log(f"is_connected between {a} and {b} with dist_threshold {t} is {gx.is_connected(a, b, t)}",color="green")
    return assert_log(gx.is_connected(a, b, t),"is_connected function is not working")

def test_get_node_contact_array():
    print_log("testing get_node_contact_array > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    t = 1.1
    if DETAILS:
        print_log(f"get_node_contact_array for {nodes} with dist_threshold {t} at start index 0 is {gx.get_node_contact_array(nodes, 0, t)}",color="green")
    return assert_log(gx.get_node_contact_array(nodes, 0, t) == [0, 2, 3,1],"get_node_contact_array function is not working")

def test_get_adjacency_matrix():
    print_log("testing get_adjacency_matrix > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    t = 1.1
    if DETAILS:
        print_log(f"get_adjacency_matrix for \n{nodes} with dist_threshold {t} is \n{gx.get_adjacency_matrix(nodes, t)}",color="green")
    return assert_log(np.array_equal(gx.get_adjacency_matrix(nodes, t),
    np.array([
        [1, 0, 1, 1], 
        [0, 1, 1, 1], 
        [1, 1, 1, 0], 
        [1, 1, 0, 1]])),
    "get_adjacency_matrix function is not working")


def test_is_graph_connected_bfs():
    print_log("testing is_graph_connected_bfs > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    t = 1.1
    if DETAILS:
        print_log(f"is_graph_connected_bfs for \n{nodes} with dist_threshold {t} is {gx.is_graph_connected_bfs(nodes, t)}",color="green")
    return assert_log(gx.is_graph_connected_bfs(nodes, t),"is_graph_connected_bfs function is not working")

def test_cout_graph_p2():
    print_log("testing cout_graph_p2 > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    targets = np.array([[0, 0], [2, 2], [0, 2], [2, 0]])
    if DETAILS:
        print_log(f"cout_graph_p2 for \n{nodes} and \n{targets} is {gx.cout_graph_p2(nodes, targets)}",color="green")
    return assert_log(gx.cout_graph_p2(nodes, targets) == 2,"cout_graph_p2 function is not working")

def test_mutate_nodes():
    print_log("testing mutate_nodes > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    mutated = gx.mutate_nodes(nodes, 0.0)
    # Avec stepsize=0, les noeuds ne doivent pas bouger
    return assert_log(np.allclose(mutated, nodes), "mutate_nodes function is not working for stepsize=0")

def test_get_shape():
    print_log("testing get_shape > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    t = 1.1
    shape = gx.get_shape(nodes, t)
    # On attend un tuple d'arêtes, on vérifie juste le type et la taille
    return assert_log(isinstance(shape, tuple) and len(shape) > 0, "get_shape function is not working")

def test_get_shape_string():
    print_log("testing get_shape_string > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    t = 1.1
    shape = gx.get_shape(nodes, t)
    shape_str = gx.get_shape_string(shape)
    return assert_log(isinstance(shape_str, str) and len(shape_str) > 0, "get_shape_string function is not working")

def test_get_shape_distance():
    print_log("testing get_shape_distance > ", end="" if not DETAILS else "\n")
    nodes1 = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    nodes2 = np.array([[0, 0], [1, 1], [0, 1], [2, 0]])
    t = 1.1
    shape1 = gx.get_shape(nodes1, t)
    shape2 = gx.get_shape(nodes2, t)
    dist = gx.get_shape_distance(shape1, shape2)
    return assert_log(isinstance(dist, int), "get_shape_distance function is not working")

def test_chose_node_near_node_weighted():
    print_log("testing chose_node_near_node_weighted > ", end="" if not DETAILS else "\n")
    nodes = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    start_node = np.array([0, 0])
    idx = gx.chose_node_near_node_weighted(nodes, start_node, 1.1, 1.0)
    return assert_log(isinstance(idx, int) and 0 <= idx < len(nodes), "chose_node_near_node_weighted function is not working")

def test_random_points_in_disk_with_attraction_point():
    print_log("testing random_points_in_disk_with_attraction_point > ", end="" if not DETAILS else "\n")
    center = np.array([0, 0])
    attract = np.array([1, 1])
    pt = gx.random_points_in_disk_with_attraction_point(center, 1.0, attract, 1.0)
    return assert_log(isinstance(pt, np.ndarray) and pt.shape == (2,), "random_points_in_disk_with_attraction_point function is not working")

if __name__ == "__main__":
    error = 0
    if test_distance(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_is_connected(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_get_node_contact_array(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_get_adjacency_matrix(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_is_graph_connected_bfs(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_cout_graph_p2(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_mutate_nodes(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_get_shape(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_get_shape_string(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_get_shape_distance(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_chose_node_near_node_weighted(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if test_random_points_in_disk_with_attraction_point(): print_log("Ok",color="green")
    else: print_log("Error",color="red"); error += 1
    if error == 0:
        print_log("all tests passed",color="green",force=True)
    else:
        print_log(f"all tests passed with {error} errors",color="red",force=True)




