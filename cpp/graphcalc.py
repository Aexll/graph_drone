import graphx # type: ignore
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

# distance
if TEST_FOR_DISTANCE:
    print("distance: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.distance(np.array([0, 0]), np.array([1, 1]))
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")


    start_time_B = time.time()
    for i in range(TEST_SIZE):
        np.linalg.norm(np.array([0, 0]) - np.array([1, 1]))
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for numpy")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")


# is_connected
if TEST_FOR_IS_CONNECTED:
    print("is_connected: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.is_connected(np.array([0, 0]), np.array([1, 1]), 1.1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")


    start_time_B = time.time()
    for i in range(TEST_SIZE):
        np.linalg.norm(np.array([0, 0]) - np.array([1, 1])) < 1.1
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for numpy")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")


# is_graph_connected
if TEST_FOR_IS_GRAPH_CONNECTED:
    print("is_graph_connected: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.is_graph_connected_bfs(TEST_NODES, TEST_DIST_THRESHOLD)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")


    start_time_B = time.time()
    for i in range(TEST_SIZE):
        ec.is_graph_connected(TEST_NODES, TEST_DIST_THRESHOLD)
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for nx")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")


# cout_graph
if TEST_FOR_COUT_GRAPH:
    print("cout_graph: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    start_time_B = time.time()
    for i in range(TEST_SIZE):
        ec.cout_snt(TEST_NODES, TEST_TARGETS)
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for nx")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")

# mutate_nodes
if TEST_FOR_MUTATE_NODES:
    print("mutate_nodes: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.mutate_nodes(TEST_NODES, 0.1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    start_time_B = time.time()
    for i in range(TEST_SIZE):
        ec.mutate_nodes(TEST_NODES, TEST_TARGETS, 0.1, 0.1, ec.cout_snt)
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for nx")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")


# optimize_nodes
if TEST_FOR_OPTIMIZE_NODES:
    print("optimize_nodes: ")

    start_time_A = time.time()
    for i in range(TEST_SIZE):
        graphx.optimize_nodes(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 1)
    end_time_A = time.time()
    print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    start_time_B = time.time()
    for i in range(TEST_SIZE):
        ec.mutate_nodes(TEST_NODES, TEST_TARGETS, 0.1, 0.1, ec.cout_snt)
    end_time_B = time.time()
    print(f"Time taken: {end_time_B - start_time_B} seconds for nx")

    print(f"Ratio: {(end_time_B - start_time_B) / (end_time_A - start_time_A)}")

    # for n in [1, 10, 100]:
    #     print(f"For n={n}")
    #     start_time_A = time.time()
    #     for i in range(TEST_SIZE):
    #         graphx.optimize_nodes(TEST_NODES, TEST_DIST_THRESHOLD, 0.1, n)
    #     end_time_A = time.time()
    #     print(f"Time taken: {end_time_A - start_time_A} seconds for graphx")

    #     start_time_B = time.time()



# print(graphx.optimize_nodes(TEST_NODES, TEST_DIST_THRESHOLD, 0.1, 1))

# place nodes at the barycenter of the targets
# NODES = np.array([[np.mean(TEST_TARGETS[:, 0]), np.mean(TEST_TARGETS[:, 1])]])

# print(graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS))

optimized_nodes = graphx.optimize_nodes_history(TEST_NODES, TEST_TARGETS, TEST_DIST_THRESHOLD, 0.1, 10000000)

print(len(optimized_nodes))


import matplotlib.pyplot as plt

# plos the trajectory of the nodes along the history


plt.scatter(TEST_TARGETS[:, 0], TEST_TARGETS[:, 1], color='red', label='Targets')
for i in range(len(optimized_nodes)):
    plt.scatter(optimized_nodes[i][:, 0], optimized_nodes[i][:, 1], color='green', label='Optimized Nodes')
plt.show()






# mutated_nodes = graphx.mutate_nodes(TEST_NODES, 0.1)

# print(graphx.cout_graph_p2(TEST_NODES, TEST_TARGETS))
# print(graphx.cout_graph_p2(optimized_nodes, TEST_TARGETS))
# print(graphx.cout_graph_p2(mutated_nodes, TEST_TARGETS))

# # plot the graph
# import matplotlib.pyplot as plt

# plt.scatter(TEST_NODES[:, 0], TEST_NODES[:, 1], color='blue', label='Nodes')
# plt.scatter(optimized_nodes[:, 0], optimized_nodes[:, 1], color='green', label='Optimized Nodes')
# plt.scatter(mutated_nodes[:, 0], mutated_nodes[:, 1], color='yellow', label='Mutated Nodes')
# plt.show()