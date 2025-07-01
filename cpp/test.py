from tkinter import N
import graphx as gx # type: ignore
import numpy as np
import time
import matplotlib.pyplot as plt
import getimg

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

# noeuds au barycentre des targets, autant de noeuds que de targets
nodes = np.array([
    np.mean(targets, axis=0)
    for _ in range(len(targets))
]).astype(np.float32)

opti_nodes = gx.optimize_nodes_genetic(nodes, targets, dist_threshold, 0.1, 10000, 20, 0.1)



print("________________________")
print("chose_node_near_node_weighted :")
point = np.array([1.5,2.2])
dist = np.array([gx.distance(np.array([nodes[i]]),np.array([point])) for i in range(len(nodes))])
choses = []
for _ in range(100000):
    choses.append(gx.chose_node_near_node_weighted(nodes, point, dist_threshold, 0.5))
choses = np.array(choses)
print(dist)
# plt.hist(choses, bins=len(nodes))
# plt.bar(dist, np.zeros(len(dist)))
# plt.xlim(0, len(nodes))
# plt.show()
count = []
for i in range(len(nodes)):
    count.append(np.sum(choses == i))
for i in range(len(nodes)):
    print(f"node {i} : {0*count[i]} \t {dist[i]:.2f} \t {(count[i]/len(choses)*100):.2f}%")





print("________________________")
print("get_adjacency_matrix :")
print(gx.get_adjacency_matrix(nodes, dist_threshold))


print("________________________")
print("get_node_contact_array :")
print(gx.get_node_contact_array(nodes, 0, dist_threshold))
# img = getimg.get_mini_graph_image(nodes, targets, dist_threshold,skin="black")
# plt.imshow(img)
# plt.show()


print("________________________")
# nodes = np.array([[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[2,3],[3,2],[3,3]])
print("mutate_nodes :")
time_start = time.perf_counter()
fails=0
nodes = opti_nodes.copy()
for _ in range(10000):
    done=False
    while not done:
        new_nodes= gx.mutate_nodes(nodes, 0.1)
        if gx.is_graph_connected_bfs(new_nodes, dist_threshold):
            nodes = new_nodes
            done=True
        else:
            fails+=1
time_end = time.perf_counter()
print(f"mutate_nodes time: {time_end - time_start} seconds")
print(f"fails: {fails}")

print("________________________")
print("safe_mutate_nodes :")
nodes = opti_nodes.copy()
# nodes = np.array([[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[2,3],[3,2],[3,3]])
im1 = getimg.get_mini_graph_image(nodes, targets, dist_threshold,skin="default")
time_start = time.perf_counter()
for _ in range(10000):
    nodes = gx.safe_mutate_nodes(nodes, 0, dist_threshold, 0.1)
time_end = time.perf_counter()
print(f"safe_mutate_nodes time: {time_end - time_start} seconds")
im2 = getimg.get_mini_graph_image(nodes, targets, dist_threshold,skin="default")

plt.subplot(1,2,1)
plt.imshow(im1)
plt.subplot(1,2,2)
plt.imshow(im2)
plt.show()





quit()

print("________________________")
print("Optimized :")
print("initial error:",gx.cout_graph_p2(nodes, targets))

opti_1 = gx.optimize_nodes(nodes, targets, dist_threshold, 0.01, 100000,False)

print("optimized error:",gx.cout_graph_p2(opti_1, targets))


print("")
print("________________________")
print("Parallel :")
parallel_opti = gx.optimize_nodes_parallel(nodes, targets, dist_threshold, 0.01, 10000,10,False)
# sort by error
sorted_parallel_opti = sorted(parallel_opti, key=lambda x: gx.cout_graph_p2(x, targets))
print("min error:",gx.cout_graph_p2(sorted_parallel_opti[0], targets))
print("mean error:",np.mean([gx.cout_graph_p2(x, targets) for x in sorted_parallel_opti]))
print("max error:",gx.cout_graph_p2(sorted_parallel_opti[-1], targets))

print("")

print("________________________")
print("History :")
start_time = time.perf_counter()
history = gx.optimize_nodes_history(nodes, targets, dist_threshold, 0.01, 100000,False,False)
print("start error:",gx.cout_graph_p2(history[0], targets))
print("mean error:",np.mean([gx.cout_graph_p2(x, targets) for x in history]))
print("last error:",gx.cout_graph_p2(history[-1], targets))


print("")
print("________________________")
print("Parallel history " , end="")
start_time = time.perf_counter()
histories = gx.optimize_nodes_history_parallel(nodes, targets, dist_threshold, 0.01, 10000,10,False)
end_time = time.perf_counter()
print(f": {end_time - start_time} seconds")
print("graph 0, first :",gx.cout_graph_p2(histories[0][0], targets))
print("graph 0, last :",gx.cout_graph_p2(histories[0][-1], targets))
print("graph 1, first :",gx.cout_graph_p2(histories[1][0], targets))
print("graph 1, last :",gx.cout_graph_p2(histories[1][-1], targets))
print("graph 2, first :",gx.cout_graph_p2(histories[2][0], targets))
print("graph 2, last :",gx.cout_graph_p2(histories[2][-1], targets))
print("graph 3, first :",gx.cout_graph_p2(histories[3][0], targets))
print("graph 3, last :",gx.cout_graph_p2(histories[3][-1], targets))
print("Averages :")
print("at index 0 :", np.mean([gx.cout_graph_p2(history[0], targets) for history in histories]))
print("at index -1 :", np.mean([gx.cout_graph_p2(history[-1], targets) for history in histories]))
# print("len(histories):",len(histories))
# print("--- at historyindex 0 ---")
# print("shape:",histories[0][-1].shape)
# print("average : ",np.mean([gx.cout_graph_p2(x, targets) for x in histories[x][0]]))



print("________________________")
print("shape :")
nodes_shaped_121 = np.array([[0,0],[1,0],[2,0]])
print("shape of (1,2,1) :", gx.get_shape(nodes_shaped_121, dist_threshold))
nodes_shaped_1311 = np.array([[0,0],[1,0],[2,0],[1,1]])
print("shape of (1,3,1,1) :", gx.get_shape(nodes_shaped_1311, dist_threshold))
nodes_shaped_1221 = np.array([[0,0],[1,0],[2,0],[3,0]])
print("shape of (1,2,2,1) :", gx.get_shape(nodes_shaped_1221, dist_threshold))

print("shape of (1,2,1) :", gx.get_shape_string(gx.get_shape(nodes_shaped_121, dist_threshold)))
print("shape of (1,3,1,1) :", gx.get_shape_string(gx.get_shape(nodes_shaped_1311, dist_threshold)))
print("shape of (1,2,2,1) :", gx.get_shape_string(gx.get_shape(nodes_shaped_1221, dist_threshold)))

print("shape of optimized : ", gx.get_shape(opti_1, dist_threshold))
print("shape of optimized string : ", gx.get_shape_string(gx.get_shape(opti_1, dist_threshold)))

print("distance between (1,3,1,1) and (1,2,2,1) :", 
gx.get_shape_distance(gx.get_shape(nodes_shaped_1311, dist_threshold), gx.get_shape(nodes_shaped_1221, dist_threshold)))


print("________________________")
print("shape string history :")
print(gx.get_shape_string_transition_history(history, dist_threshold))

# shapes_info = gx.decompose_history_by_shape(history, targets, dist_threshold)
# for shape_str, info in shapes_info.items():
#     print("Shape:", shape_str)
#     print("Best score:", info['score'])
#     print("Best graph:", info['graph'])

print("________________________")
print("genetic :")
genetic_opti = gx.optimize_nodes_genetic(nodes, targets, dist_threshold, 0.1, 100000, 20, 0.1)
print("genetic error:",gx.cout_graph_p2(genetic_opti, targets))




