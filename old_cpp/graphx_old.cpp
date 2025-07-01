#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <thread>
#include <mutex>
#include <iostream>



// distance entre deux points
double distance(pybind11::array_t<double> a, pybind11::array_t<double> b) {
    pybind11::buffer_info a_info = a.request();
    pybind11::buffer_info b_info = b.request();

    // Check that both arrays are 1D and of size 2
    if (a_info.ndim != 1 || b_info.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (a_info.size != 2 || b_info.size != 2) {
        throw std::runtime_error("Input arrays must have size 2");
    }

    // Cast pointers to double*
    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* b_ptr = static_cast<double*>(b_info.ptr);

    double dx = a_ptr[0] - b_ptr[0];
    double dy = a_ptr[1] - b_ptr[1];
    return std::sqrt(dx * dx + dy * dy);
}


bool is_connected(pybind11::array_t<double> a, pybind11::array_t<double> b, double dist_threshold) {
    // ne pas utiliser distance, car on ne veut pas calculer faire de racine carrée
    pybind11::buffer_info a_info = a.request();
    pybind11::buffer_info b_info = b.request();

    // Check that both arrays are 1D and of size 2
    if (a_info.ndim != 1 || b_info.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (a_info.size != 2 || b_info.size != 2) {
        throw std::runtime_error("Input arrays must have size 2");
    }

    double* a_ptr = static_cast<double*>(a_info.ptr);
    double* b_ptr = static_cast<double*>(b_info.ptr);

    double dx = a_ptr[0] - b_ptr[0];
    double dy = a_ptr[1] - b_ptr[1];
    return (dx * dx + dy * dy) < (dist_threshold * dist_threshold);
}

/*
L'algorithme is_graph_connected_bfs utilise une recherche en largeur (BFS) 
pour vérifier si tous les noeuds d'un graphe 
(défini par des positions de points et un seuil de distance) sont accessibles à partir du premier noeud. 
On marque les noeuds visités en partant du noeud 0, 
et on ajoute à la file tous les voisins (points à distance inférieure au seuil) non encore visités. 
Si à la fin tous les noeuds ont été visités, (visited[i] == true pour tout i)
le graphe est connexe.
*/
bool is_graph_connected_bfs(pybind11::array_t<double> nodes, double dist_threshold) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }
    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);

    // Visited array
    std::vector<bool> visited(n, false);
    std::vector<size_t> queue;
    queue.push_back(0);
    visited[0] = true;

    while (!queue.empty()) {
        size_t current = queue.front();
        queue.erase(queue.begin());

        double* current_ptr = nodes_ptr + current * 2;
        for (size_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                double* other_ptr = nodes_ptr + i * 2;
                double dx = current_ptr[0] - other_ptr[0];
                double dy = current_ptr[1] - other_ptr[1];
                double dist2 = dx * dx + dy * dy;
                if (dist2 < dist_threshold * dist_threshold) {
                    visited[i] = true;
                    queue.push_back(i);
                }
            }
        }
    }

    // Check if all nodes are visited
    for (size_t i = 0; i < n; ++i) {
        if (!visited[i]) return false;
    }
    return true;
}





/*
fonction de cout
racines carrée de la somme des distances entre tous les noeuds et tous les targets
*/
double cout_graph_p2(pybind11::array_t<double> nodes, pybind11::array_t<double> targets) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }

    pybind11::buffer_info targets_info = targets.request();
    if (targets_info.ndim != 2 || targets_info.shape[1] != 2) {
        throw std::runtime_error("Input targets must be a 2D array with shape (m, 2)");
    }

    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);
    double* targets_ptr = static_cast<double*>(targets_info.ptr);

    // return raw_cout_graph_p2(nodes_ptr, targets_ptr, n, m);

    // on calcule la distance entre les noeuds et les targets
    double cout = 0;
    for (size_t i = 0; i < n; ++i) {
        double dx = nodes_ptr[i * 2] - targets_ptr[i * 2];
        double dy = nodes_ptr[i * 2 + 1] - targets_ptr[i * 2 + 1];
        cout += dx * dx + dy * dy;
    }
    return std::sqrt(cout);
}


/* COUT TOTAL déprécié 
double cout_graph_p2(pybind11::array_t<double> nodes, pybind11::array_t<double> targets) {
    // nodes: shape (n, 2)
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }

    pybind11::buffer_info targets_info = targets.request();
    if (targets_info.ndim != 2 || targets_info.shape[1] != 2) {
        throw std::runtime_error("Input targets must be a 2D array with shape (m, 2)");
    }

    size_t n = nodes_info.shape[0];
    size_t m = targets_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);
    double* targets_ptr = static_cast<double*>(targets_info.ptr);

    // return raw_cout_graph_p2(nodes_ptr, targets_ptr, n, m);

    // on calcule la distance entre les noeuds et les targets
    double cout = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double dx = nodes_ptr[i * 2] - targets_ptr[j * 2];
            double dy = nodes_ptr[i * 2 + 1] - targets_ptr[j * 2 + 1];
            cout += dx * dx + dy * dy;
        }
    }
    return std::sqrt(cout);
}
*/


// Optimized mutate_nodes: use a single random generator for all elements, and avoid repeated vector construction
pybind11::array_t<double> mutate_nodes(pybind11::array_t<double> nodes, double stepsize) {
    pybind11::buffer_info nodes_info = nodes.request();
    if (nodes_info.ndim != 2 || nodes_info.shape[1] != 2) {
        throw std::runtime_error("Input nodes must be a 2D array with shape (n, 2)");
    }
    size_t n = nodes_info.shape[0];
    double* nodes_ptr = static_cast<double*>(nodes_info.ptr);

    // Preallocate result array with the same shape as input
    pybind11::array_t<double> result(nodes_info.shape);
    pybind11::buffer_info result_info = result.request();
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Efficient random number generation: fill all at once
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> d(0.0, stepsize);

    // Mutate all coordinates in a single loop (2*n elements)
    for (size_t i = 0; i < n * 2; ++i) {
        result_ptr[i] = nodes_ptr[i] + d(gen);
    }
    return result;
}


/*
on mutate les noeuds n fois, on vérifie si le graph est connecté, sinon on garde les noeuds précédents
on répète alors n fois l'opération
*/
pybind11::array_t<double> optimize_nodes(pybind11::array_t<double> nodes, pybind11::array_t<double> targets, double dist_threshold, double stepsize, size_t n) {
    double best_error = cout_graph_p2(nodes, targets);
    for (size_t i = 0; i < n; ++i) {
        double current_stepsize = stepsize * (static_cast<double>(n - i) / n);
        pybind11::array_t<double> new_nodes = mutate_nodes(nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                nodes = new_nodes;
                best_error = new_error;
            }
        }
    }
    return nodes;
}

/*
Optimise les noeuds en gardant une historique des noeuds optimisés au fur et à mesure
*/
std::vector<pybind11::array_t<double>> optimize_nodes_history(pybind11::array_t<double> nodes, pybind11::array_t<double> targets, double dist_threshold, double stepsize, size_t n) {
    std::vector<pybind11::array_t<double>> history;
    double best_error = cout_graph_p2(nodes, targets);
    // Store the initial state
    history.push_back(nodes);
    for (size_t i = 0; i < n; ++i) {
        double current_stepsize = stepsize * (static_cast<double>(n - i) / n);
        pybind11::array_t<double> new_nodes = mutate_nodes(nodes, current_stepsize);
        double new_error = cout_graph_p2(new_nodes, targets);
        if (new_error < best_error) {
            if (is_graph_connected_bfs(new_nodes, dist_threshold)) {
                nodes = new_nodes;
                best_error = new_error;
                // Store a copy of the new nodes in the history
                history.push_back(nodes);
            }
        }
    }
    return history;
}

/*
Optimise les noeuds en lancant plusieurs threads de optimize_nodes_history en parallèle

retourne une liste d'historiques de noeuds optimisés
*/
std::vector<std::vector<pybind11::array_t<double>>> optimize_nodes_history_parallel(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads
) {
    std::vector<std::vector<pybind11::array_t<double>>> all_histories(n_threads);
    std::vector<std::thread> threads;

    pybind11::gil_scoped_release release;

    auto worker = [&](size_t thread_id) {
        try {
            pybind11::gil_scoped_acquire acquire;
            std::cout << "Thread " << thread_id << " start history\n";
            auto history = optimize_nodes_history(nodes, targets, dist_threshold, stepsize, n);
            std::cout << "Thread " << thread_id << " end history\n";
            all_histories[thread_id] = std::move(history);
        } catch (const std::exception& e) {
            std::cout << "Exception in thread " << thread_id << ": " << e.what() << std::endl;
        }
    };

    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    std::cout << "All threads launched\n";
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "All threads joined\n";
    return all_histories;
}



// Version optimisée de la fonction parallèle
std::vector<std::vector<pybind11::array_t<double>>> optimize_nodes_history_parallel_native(
    pybind11::array_t<double> nodes,
    pybind11::array_t<double> targets,
    double dist_threshold,
    double stepsize,
    size_t n,
    size_t n_threads
) {
    // Convertir les données en C++ natif AVANT de lancer les threads
    NodeData nodes_native(nodes);
    NodeData targets_native(targets);
    
    std::vector<std::vector<NodeData>> all_histories_native(n_threads);
    std::vector<std::thread> threads;

    // Maintenant on peut relâcher le GIL car on travaille avec du C++ pur
    pybind11::gil_scoped_release release;

    auto worker = [&](size_t thread_id) {
        try {
            std::cout << "Thread " << thread_id << " start history\n";
            auto history = optimize_nodes_history_native(
                nodes_native, targets_native, dist_threshold, stepsize, n
            );
            std::cout << "Thread " << thread_id << " end history\n";
            all_histories_native[thread_id] = std::move(history);
        } catch (const std::exception& e) {
            std::cout << "Exception in thread " << thread_id << ": " << e.what() << std::endl;
        }
    };

    for (size_t i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    std::cout << "All threads launched\n";
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "All threads joined\n";

    // Reconvertir en pybind11::array_t<double> APRÈS les threads
    std::vector<std::vector<pybind11::array_t<double>>> all_histories(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        all_histories[i].reserve(all_histories_native[i].size());
        for (const auto& node_data : all_histories_native[i]) {
            all_histories[i].push_back(node_data.to_array());
        }
    }
    
    return all_histories;
}


PYBIND11_MODULE(graphx, m) {
    m.def("distance", &distance, "Distance entre deux points");
    m.def("is_connected", &is_connected, "Vérifie si deux points sont connectés");
    m.def("is_graph_connected_bfs", &is_graph_connected_bfs, "Vérifie si un graph est connecté");
    m.def("cout_graph_p2", &cout_graph_p2, "Calcul le cout d'un graph");
    m.def("mutate_nodes", &mutate_nodes, "Muter les noeuds");
    m.def("optimize_nodes", &optimize_nodes, "Optimiser les noeuds");
    m.def("optimize_nodes_history", &optimize_nodes_history, "Optimiser les noeuds et garder l'historique");
    m.def("optimize_nodes_history_parallel", &optimize_nodes_history_parallel, "Optimiser les noeuds et garder l'historique en parallèle");
}