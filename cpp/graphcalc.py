import graphx  # type: ignore
import numpy as np
import time
import errorcalc as ec

TEST_SIZE = 1000000
TEST_NODES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
TEST_TARGETS = np.array([[0, 1], [2, 0], [0, 2], [2, 2]])
TEST_DIST_THRESHOLD = 1.1

TEST_FOR_DISTANCE = False
TEST_FOR_IS_CONNECTED = False
TEST_FOR_IS_GRAPH_CONNECTED = False
TEST_FOR_COUT_GRAPH = False
TEST_FOR_MUTATE_NODES = False
TEST_FOR_OPTIMIZE_NODES = False
TEST_FOR_OPTIMIZE_NODES_HISTORY = False
TEST_FOR_OPTIMIZE_NODES_HISTORY_PARALLEL = False

# distance
def test_distance():
    print("distance: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.distance(np.array([[0, 0]]), np.array([[1, 1]]))
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    start_time_B = time.time()
    for i in range(TEST_SIZE):
        np.linalg.norm(np.array([0, 0]) - np.array([1, 1]))
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for numpy")
    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")

# is_connected
def test_is_connected():
    print("is_connected: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.is_connected(np.array([[0, 0]]), np.array([[1, 1]]), 1.1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    start_time_B = time.time()
    for i in range(TEST_SIZE):
        np.linalg.norm(np.array([0, 0]) - np.array([1, 1])) < 1.1
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for numpy")
    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")

# is_graph_connected
def test_is_graph_connected():
    print("is_graph_connected: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.is_graph_connected_bfs(TEST_NODES, TEST_DIST_THRESHOLD)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

# cout_graph
def test_cout_graph():
    print("cout_graph: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

# mutate_nodes
def test_mutate_nodes():
    print("mutate_nodes: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.mutate_nodes(TEST_NODES, 0.1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

# optimize_nodes
def test_optimize_nodes():
    print("optimize_nodes: ")
    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.optimize_nodes_cooling(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

# optimize_nodes_history
def test_optimize_nodes_history():
    print("optimize_nodes_history: ")
    nodes = TEST_NODES.copy()
    targets = TEST_TARGETS.copy()
    dist_threshold = TEST_DIST_THRESHOLD
    stepsize = 0.1
    n = 10
    history = graphx.optimize_nodes_history(nodes, targets, dist_threshold, stepsize, n)
    print(f"History length: {len(history)}")

# optimize_nodes_history_parallel
def test_optimize_nodes_history_parallel():
    print("optimize_nodes_history_parallel: ")
    nodes = TEST_NODES.copy()
    targets = TEST_TARGETS.copy()
    dist_threshold = TEST_DIST_THRESHOLD
    stepsize = 0.1
    n = 10
    n_threads = 2
    all_histories = graphx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, stepsize, n, n_threads)
    print(f"Number of threads: {len(all_histories)}")
    print(f"History length for thread 0: {len(all_histories[0])}")

if __name__ == "__main__":
    if TEST_FOR_DISTANCE:
        test_distance()
    if TEST_FOR_IS_CONNECTED:
        test_is_connected()
    if TEST_FOR_IS_GRAPH_CONNECTED:
        test_is_graph_connected()
    if TEST_FOR_COUT_GRAPH:
        test_cout_graph()
    if TEST_FOR_MUTATE_NODES:
        test_mutate_nodes()
    if TEST_FOR_OPTIMIZE_NODES:
        test_optimize_nodes()
    if TEST_FOR_OPTIMIZE_NODES_HISTORY:
        test_optimize_nodes_history()
    if TEST_FOR_OPTIMIZE_NODES_HISTORY_PARALLEL:
        test_optimize_nodes_history_parallel()














# place nodes at the barycenter of the targets
NODES = np.array([[np.mean(TEST_TARGETS[:, 0]), np.mean(TEST_TARGETS[:, 1])]])

print(graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS))

# optimized_nodes = graphx.optimize_nodes_cooling(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 10000000)
# optimized_nodes = graphx.optimize_nodes_history_parallel(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 1000, 2)

# print(len(optimized_nodes))


optimized_nodes_history = graphx.optimize_nodes_history(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 10000000)

print(len(optimized_nodes_history))
print(optimized_nodes_history)



import matplotlib.pyplot as plt

# plos the trajectory of the nodes along the history


# for i in range(len(optimized_nodes)):
#     plt.scatter(optimized_nodes[i][:, 0], optimized_nodes[i][:, 1], color='green', label='Optimized Nodes')
# plt.show()

# plt.scatter(TEST_TARGETS[:, 0], TEST_TARGETS[:, 1], color='red', label='Targets')
# plt.scatter(optimized_nodes[:, 0], optimized_nodes[:, 1], color='green', label='Optimized Nodes')
# plt.show()




# mutated_nodes = graphx.mutate_nodes(TEST_NODES, 0.1)

# print(graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS))
# print(graphx.cout_graph_p2(optimized_nodes, TEST_TARGETS))
# print(graphx.cout_graph_p2(mutated_nodes, TEST_TARGETS))

# # plot the graph
# import matplotlib.pyplot as plt

# plt.scatter(mutated_nodes[:, 0], mutated_nodes[:, 1], color='yellow', label='Mutated Nodes')
# plt.show()