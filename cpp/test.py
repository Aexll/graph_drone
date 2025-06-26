import graphx as gx # type: ignore
import numpy as np
import time


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